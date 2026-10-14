"""Regenerate the figures and numbers in strategy-explained.md.

Rebuilds the strategy's inputs -- the monthly top-20 dollar-volume universe and
the 60-day PCA -> OLS -> residual z-score -- from the same data folder LEAN
read, checks that they reproduce every rebalance LEAN logged, then draws the
figures.

    uv run --group pca run_local.py "PCA Stat Arbitrage" --data-folder data-sp500
    uv run --group pca run_local.py "PCA Stat Arbitrage (short)" --data-folder data-sp500
    uv run run_local.py "SPY Buy & Hold 2021-2026" --data-folder data-sp500
    uv run --group pca python research/pca_stat_arbitrage/explain_figures.py

Needs `data-sp500/` (see "Constituents universes" in README.md) and the three
backtests above: the picks, trades and equity curves come from their result
files, not from a re-simulation.
"""

import io
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data-sp500" / "equity" / "usa"
OUT = Path(__file__).resolve().parent
RESULTS = REPO / "results"

START, END = "2021-09-16", "2026-09-15"
LOOKBACK, COMPONENTS, ENTRY, UNIVERSE = 60, 3, -1.5, 20
EXAMPLE = pd.Timestamp("2024-11-01")

FONT_CANDIDATES = ["Microsoft JhengHei", "Noto Sans TC", "Microsoft YaHei",
                   "PingFang TC", "Heiti TC", "Noto Sans CJK TC", "DejaVu Sans"]
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e4e1", "#fcfcfb"
# Reference categorical palette, fixed order (dataviz skill, palette.md).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
PALE = "#cfd9e6"


def use_cjk_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [n for n in FONT_CANDIDATES if n in installed] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.sans-serif": chosen, "axes.unicode_minus": False,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "font.size": 10,
    })


# --------------------------------------------------------------------------
# data, the way LEAN sees it
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def read_bars(ticker):
    """Raw bars plus LEAN's default Adjusted close (raw x the factor of the
    first factor-file row dated on or after the bar)."""
    path = DATA / "daily" / f"{ticker.lower()}.zip"
    if not path.exists():
        return None
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(archive.namelist()[0]).decode()
    bars = pd.read_csv(io.StringIO(raw), header=None,
                       names=["date", "open", "high", "low", "close", "volume"])
    bars.index = pd.to_datetime(bars.pop("date").astype(str).str[:8], format="%Y%m%d")
    bars[["open", "high", "low", "close"]] /= 10000.0

    factors = pd.read_csv(DATA / "factor_files" / f"{ticker.lower()}.csv", header=None,
                          names=["date", "price", "split", "ref"])
    factors.index = pd.to_datetime(factors["date"].astype(str), format="%Y%m%d")
    scale = (factors["price"] * factors["split"]).sort_index()
    position = scale.index.searchsorted(bars.index, side="left")
    bars["adj"] = bars["close"] * scale.iloc[position].to_numpy()
    return bars


def constituents(file_date):
    path = DATA / "universes" / "etf" / "spy" / f"{file_date:%Y%m%d}.csv"
    rows = [line.split(",") for line in path.read_text().splitlines() if line.strip()]
    return [r[0] for r in rows if r[3]]


def select(day, files):
    """The selection that runs at 00:00 on `day`: the universe file and the
    bar of the previous session. Returns the top 20 by dollar volume."""
    file_date = files[files < day][-1]
    rows = {}
    for ticker in constituents(file_date):
        bars = read_bars(ticker)
        if bars is None:
            continue
        before = bars.loc[:day - pd.Timedelta(days=1)]
        if before.empty:
            continue
        bar = before.iloc[-1]
        if bar.close > 5 and bar.close * bar.volume > 0:
            rows[ticker] = bar.close * bar.volume
    return pd.Series(rows).sort_values().iloc[-UNIVERSE:][::-1]


def closes(tickers, day):
    """What self.history(symbols, 60, DAILY).close.unstack(0) returns at 08:00."""
    frame = {}
    for t in tickers:
        adj = read_bars(t)["adj"].loc[:day - pd.Timedelta(days=1)]
        frame[t] = adj.iloc[-LOOKBACK:]
    return pd.DataFrame(frame)


def pca_model(day, tickers):
    """The original's _compute_weights_by_symbol, keeping the intermediates."""
    history = closes(tickers, day)
    sample = np.log(history.dropna(axis=1))
    sample = sample - sample.mean()
    pca = PCA().fit(sample)
    factors = np.dot(sample, pca.components_.T)[:, :COMPONENTS]
    design = sm.add_constant(factors)
    models = {t: sm.OLS(sample[t], design).fit() for t in sample.columns}
    residuals = pd.DataFrame({t: m.resid for t, m in models.items()}, index=sample.index)
    z = (residuals - residuals.mean()) / residuals.std()
    selected = z.iloc[-1][z.iloc[-1] < ENTRY]
    weights = selected.abs() / selected.abs().sum()
    fitted = pd.DataFrame({t: m.fittedvalues for t, m in models.items()}, index=sample.index)
    return dict(history=history, sample=sample, pca=pca, factors=factors, models=models,
                residuals=residuals, z=z, last=z.iloc[-1], weights=weights, fitted=fitted)


def lean_rebalances(folder="pca_stat_arbitrage"):
    log = (RESULTS / folder / "PcaStatArbitrageAlgorithm-log.txt").read_text(encoding="utf-8")
    out = {}
    for date, body in re.findall(r"^(\d{4}-\d{2}-\d{2}) 08:00:00 REBALANCE ?(.*)$", log, re.M):
        pairs = [p.split(":") for p in body.split(",") if p]
        out[pd.Timestamp(date)] = {t: -float(w) for t, w in pairs}
    return out


def load_result(folder):
    path = next(p for p in (RESULTS / folder).glob("*.json")
                if not p.name.endswith(("summary.json", "order-events.json"))
                and not p.name.startswith("data-monitor"))
    return json.loads(path.read_text(encoding="utf-8"))


def equity(result):
    values = result["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
    series = pd.Series({pd.Timestamp(v[0], unit="s"): v[4] for v in values})
    return series.groupby(series.index.normalize()).last()


def orders(result):
    rows = []
    for order in result["orders"].values():
        rows.append(dict(ticker=order["symbol"]["value"],
                         time=pd.Timestamp(order["time"]).tz_convert("America/New_York").tz_localize(None),
                         qty=order["quantity"], price=order["price"], status=order["status"],
                         value=abs(order["quantity"] * order["price"])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# reproduce the decisions
# --------------------------------------------------------------------------

def build():
    files = pd.DatetimeIndex(sorted(pd.Timestamp(p.stem) for p in
                                    (DATA / "universes" / "etf" / "spy").glob("*.csv")))
    spy_days = read_bars("SPY").loc[START:END].index
    month_starts = pd.Series(spy_days, index=spy_days).groupby(spy_days.to_period("M")).first()
    # The first rebalance (2021-09-16, end of warm-up) trades on the
    # 2021-09-01 selection; every later one on its own month start.
    selection_days = [pd.Timestamp("2021-09-01")] + [d for d in month_starts if d > pd.Timestamp(START)]
    rebalance_days = [pd.Timestamp(START)] + selection_days[1:]
    universes = {d: select(d, files) for d in selection_days}
    models = {r: pca_model(r, universes[s].index) for r, s in zip(rebalance_days, selection_days)}
    return spy_days, selection_days, rebalance_days, universes, models


def verify(models, lean):
    mismatches = 0
    worst = 0.0
    for day, m in models.items():
        theirs = lean.get(day, {})
        mine = m["weights"].to_dict()
        if set(mine) != set(theirs):
            mismatches += 1
            print("mismatch", day.date(), sorted(mine), sorted(theirs))
            continue
        for t in mine:
            worst = max(worst, abs(mine[t] - theirs[t]))
    print(f"verified {len(models)} rebalances, {mismatches} mismatches, "
          f"largest weight difference {worst:.4f}")


def forward_returns(models, rebalance_days, spy_days):
    """For every (month, stock): its z-score and the next month's return,
    open-to-open as LEAN trades it, relative to the 20-name average."""
    rows = []
    for i, day in enumerate(rebalance_days):
        stop = rebalance_days[i + 1] if i + 1 < len(rebalance_days) else None
        m = models[day]
        for t, z in m["last"].items():
            bars = read_bars(t)
            entry = bars.loc[day:]
            if entry.empty:
                continue
            # adj open = raw open x the close's adjustment factor
            factor = bars["adj"] / bars["close"]
            p0 = entry["open"].iloc[0] * factor.loc[entry.index[0]]
            if stop is None:
                last = bars.loc[:END].iloc[-1]
                p1 = last["adj"]
            else:
                nxt = bars.loc[stop:]
                p1 = nxt["open"].iloc[0] * factor.loc[nxt.index[0]]
            rows.append(dict(day=day, ticker=t, z=z, ret=p1 / p0 - 1))
    frame = pd.DataFrame(rows)
    frame["excess"] = frame["ret"] - frame.groupby("day")["ret"].transform("mean")
    return frame


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_universe(day, universe):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    u = universe[::-1] / 1e9
    ax.barh(range(len(u)), u, color=BLUE, height=0.7)
    ax.set_yticks(range(len(u)))
    ax.set_yticklabels(u.index, color=INK)
    for i, v in enumerate(u):
        ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("前一個交易日的成交金額（十億美元）")
    ax.set_title(f"① {day:%Y-%m-%d} 的股票池：成交金額最大的 20 檔", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-universe.png", dpi=130)
    plt.close(fig)


def fig_factors(day, m):
    """What the three factors look like, next to an equal-weight average."""
    sample, pca = m["sample"], m["pca"]
    f = pd.DataFrame(m["factors"], index=sample.index, columns=["PC1", "PC2", "PC3"])
    # Signs from PCA are arbitrary; flip PC1 so it rises with the market.
    mkt = sample.mean(axis=1)
    if np.corrcoef(f.PC1, mkt)[0, 1] < 0:
        f["PC1"] *= -1
    ratio = pca.explained_variance_ratio_
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 2.2]})
    k = np.arange(1, len(ratio) + 1)
    left.bar(k, ratio * 100, color=[BLUE] * COMPONENTS + [PALE] * (len(ratio) - COMPONENTS))
    for i in range(COMPONENTS):
        left.text(k[i], ratio[i] * 100 + 1.5, f"{ratio[i] * 100:.0f}%", ha="center", color=INK, fontsize=9)
    left.set_xticks(k)
    left.set_xticklabels([str(i) if i <= 6 or i % 5 == 0 else "" for i in k], fontsize=8)
    left.set_xlabel("第幾個主成分")
    left.set_ylabel("解釋的變異（%）")
    left.grid(axis="x", visible=False)
    left.set_title("前 3 個主成分（深藍）被當成因子", loc="left", color=INK, fontsize=11)
    for col, c in zip(f.columns, [BLUE, ORANGE, AQUA]):
        right.plot(f.index, f[col], color=c, lw=2, label=col)
    right.plot(mkt.index, mkt * np.sqrt(sample.shape[1]), color=MUTED, lw=1.2, ls="--",
               label="20 檔平均（同尺度）")
    right.axhline(0, color=MUTED, lw=0.8)
    right.legend(frameon=False, ncol=4, loc="upper left")
    right.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    right.set_title(f"三個因子在 60 天裡的走勢：PC1 和 20 檔平均很像（相關 {np.corrcoef(f.PC1, mkt)[0, 1]:.2f}）", loc="left", color=INK, fontsize=11)
    fig.suptitle(f"② {day:%Y-%m-%d} 換股：PCA 從 20 檔股票裡抽出的共同因子", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2-factors.png", dpi=130)
    plt.close(fig)
    return ratio[:COMPONENTS], np.corrcoef(f.PC1, mkt)[0, 1]


def fig_residual(day, m, ticker):
    sample, fitted, resid, z = m["sample"], m["fitted"], m["residuals"], m["z"]
    price = m["history"][ticker]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True,
                                      gridspec_kw={"height_ratios": [1.5, 1]})
    # Back to dollars so the reader sees prices, not log deviations.
    level = np.exp(np.log(price).mean())
    top.plot(price.index, price, color=INK, lw=2, label=f"{ticker} 實際收盤價")
    top.plot(price.index, level * np.exp(fitted[ticker]), color=BLUE, lw=2, ls="--",
             label="因子模型算出的「合理價」")
    top.fill_between(price.index, price, level * np.exp(fitted[ticker]), color=ORANGE, alpha=0.15, lw=0)
    p_last, f_last = price.iloc[-1], level * np.exp(fitted[ticker].iloc[-1])
    top.annotate(f"實際 {p_last:.2f}\n合理 {f_last:.2f}\n便宜 {(1 - p_last / f_last) * 100:.1f}%",
                 (price.index[-1], p_last), xytext=(10, -10), textcoords="offset points",
                 color=INK, fontsize=9, va="top")
    top.legend(frameon=False, loc="upper left")
    top.set_ylabel("還原權值收盤價（美元）")
    top.set_title(f"③ {ticker}：實際價格 vs. 3 個因子解釋得了的部分（{day:%Y-%m-%d} 換股前 60 天）",
                  loc="left", color=INK, fontsize=12)
    zz = z[ticker]
    bottom.bar(zz.index, zz, color=[ORANGE if v < ENTRY else PALE for v in zz], width=1.0)
    bottom.axhline(ENTRY, color=RED, lw=1.2, ls="--")
    bottom.text(zz.index[1], ENTRY - 0.15, "進場門檻 z = -1.5", color=RED, va="top", fontsize=9)
    bottom.axhline(0, color=MUTED, lw=0.8)
    bottom.annotate(f"最後一天 z = {zz.iloc[-1]:.2f}", (zz.index[-1], zz.iloc[-1]),
                    xytext=(-130, -4), textcoords="offset points", color=INK, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=MUTED))
    bottom.set_ylabel("殘差 z 分數")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    fig.savefig(OUT / "fig3-residual.png", dpi=130)
    plt.close(fig)
    return p_last, f_last


def fig_scores(day, m):
    last = m["last"].sort_values()
    w = m["weights"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = [ORANGE if v < ENTRY else PALE for v in last]
    ax.barh(range(len(last)), last, color=colors, height=0.7)
    ax.set_yticks(range(len(last)))
    ax.set_yticklabels(last.index, color=INK)
    ax.axvline(ENTRY, color=RED, lw=1.2, ls="--")
    ax.axvline(0, color=MUTED, lw=0.8)
    for i, (t, v) in enumerate(last.items()):
        label = f"z={v:.2f}  →  權重 {w[t] * 100:.1f}%" if t in w else f"{v:.2f}"
        ax.text(v - 0.08 if v < 0 else v + 0.08, i, label, va="center",
                ha="right" if v < 0 else "left", fontsize=8.5, color=INK)
    ax.set_xlim(min(last.min() - 2.6, -4.5), max(last.max() + 0.8, 2.5))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("最後一天的殘差 z 分數（越負 = 比因子模型便宜越多）")
    ax.set_title(f"④ {day:%Y-%m-%d}：20 檔的 z 分數，橘色 = 低於 -1.5，被選中", loc="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig4-scores.png", dpi=130)
    plt.close(fig)


def fig_equity(curves):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    for (label, s, c), dy in zip(curves, [6, -6, 0, 0]):
        s = s / s.iloc[0] * 100_000
        top.plot(s.index, s / 1000, color=c, lw=2, label=label)
        top.annotate(f"{s.iloc[-1] / 10000:.1f} 萬", (s.index[-1], s.iloc[-1] / 1000),
                     xytext=(6, dy), textcoords="offset points", va="center", color=INK, fontsize=9)
        dd = s / s.cummax() - 1
        bottom.plot(dd.index, dd * 100, color=c, lw=1.6)
    top.set_ylabel("資產（千美元）")
    top.legend(loc="upper left", frameon=False)
    top.set_title("⑥ 10 萬美元的走勢，與從最高點回落的幅度", loc="left", color=INK, fontsize=12)
    bottom.set_ylabel("回撤（%）")
    bottom.xaxis.set_major_locator(mdates.YearLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig6-equity.png", dpi=130)
    plt.close(fig)


def fig_signal(fwd):
    """Does a low z-score predict a better next month? Bucket every stock-month."""
    bins = [-np.inf, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, np.inf]
    labels = ["< -2", "-2 ~ -1.5", "-1.5 ~ -1", "-1 ~ -0.5", "-0.5 ~ 0",
              "0 ~ 0.5", "0.5 ~ 1", "1 ~ 1.5", "> 1.5"]
    fwd = fwd.assign(bucket=pd.cut(fwd.z, bins, labels=labels))
    g = fwd.groupby("bucket", observed=False).agg(mean=("excess", "mean"), median=("excess", "median"),
                                                  hit=("excess", lambda s: (s > 0).mean()), n=("excess", "size"))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = np.arange(len(g))
    ax.bar(x, g["mean"] * 100, color=[ORANGE] * 2 + [PALE] * (len(g) - 2), width=0.7)
    for xi, (_, r) in zip(x, g.iterrows()):
        v = r["mean"] * 100
        ax.text(xi, v + (0.15 if v >= 0 else -0.15), f"{v:+.2f}%\n(n={r.n})",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("換股當天的殘差 z 分數（橘色 = 策略會買的區間）")
    ax.set_ylabel("下個月報酬 - 20 檔平均（%）")
    lo, hi = (g["mean"] * 100).min(), (g["mean"] * 100).max()
    ax.set_ylim(min(lo - 1.2, -1), max(hi + 1.2, 1))
    ax.grid(axis="x", visible=False)
    ax.set_title("⑤ 訊號有沒有用？61 個月 × 20 檔，依 z 分數分組後，下個月平均贏大家多少",
                 loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig5-signal.png", dpi=130)
    plt.close(fig)
    return g


def fig_holdings(lean, rebalance_days):
    fig, ax = plt.subplots(figsize=(11, 3.4))
    days = rebalance_days
    counts = [len(lean.get(d, {})) for d in days]
    top = [max(lean[d].values()) if lean.get(d) else 0 for d in days]
    colors = [MUTED if n == 0 else (RED if n == 1 else BLUE) for n in counts]
    ax.bar(days, counts, width=22, color=colors)
    for d, n in zip(days, counts):
        if n == 0:
            ax.text(d, 0.15, "現金", ha="center", va="bottom", fontsize=8, color=MUTED, rotation=90)
    ax.set_ylabel("持有幾檔")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, RED, MUTED)]
    ax.legend(handles, ["2 檔以上", "只有 1 檔（100% 押一檔）", "0 檔（全現金）"],
              frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, max(counts) + 2.5)
    ax.set_title("⑦ 每個月被選中幾檔", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig7-holdings.png", dpi=130)
    plt.close(fig)
    return counts, top


def monthly_baselines(fwd, rebalance_days):
    """Close the loop on selection: compound, month by month, (a) all 20 names
    equal weight and (b) the picks at the strategy's weights. Open to open,
    no fees, no cash buffer -- a frictionless version of the same book."""
    ew, picks = [1.0], [1.0]
    for day in rebalance_days:
        month = fwd[fwd.day == day]
        ew.append(ew[-1] * (1 + month.ret.mean()))
        chosen = month[month.z < ENTRY]
        if chosen.empty:
            picks.append(picks[-1])
        else:
            w = chosen.z.abs() / chosen.z.abs().sum()
            picks.append(picks[-1] * (1 + (w * chosen.ret).sum()))
    return np.array(ew), np.array(picks)


def stats(s):
    years = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    return dict(end=s.iloc[-1], cagr=(s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1,
                dd=(1 - s / s.cummax()).max(), vol=r.std() * np.sqrt(252))


def yearly(s):
    y = s.groupby(s.index.year).last()
    first = pd.Series({s.index[0].year - 1: s.iloc[0]})
    return pd.concat([first, y]).pct_change().dropna() * 100


def main():
    use_cjk_font()
    spy_days, selection_days, rebalance_days, universes, models = build()
    lean = lean_rebalances()
    verify(models, lean)

    # --- the worked example -------------------------------------------------
    m = models[EXAMPLE]
    u = universes[EXAMPLE]
    print(f"\n{EXAMPLE.date()} universe:", ", ".join(f"{t}({v / 1e9:.1f}B)" for t, v in u.items()))
    ratio, corr = fig_factors(EXAMPLE, m)
    print("variance explained by PC1-3:", np.round(ratio * 100, 1), "sum", round(ratio.sum() * 100, 1),
          "PC1 vs average corr", round(corr, 3))
    loadings = pd.DataFrame(m["pca"].components_[:COMPONENTS].T, index=m["sample"].columns,
                            columns=["PC1", "PC2", "PC3"])
    print("loadings (largest |x|):")
    for c in loadings:
        print(" ", c, loadings[c].reindex(loadings[c].abs().sort_values(ascending=False).index).head(5).round(2).to_dict())
    print("60-day log-price std:", m["sample"].std().sort_values(ascending=False).round(3).head(6).to_dict())
    print("z last:", m["last"].sort_values().round(2).to_dict())
    print("weights:", m["weights"].round(4).to_dict())
    fig_universe(EXAMPLE, u)
    fig_scores(EXAMPLE, m)
    example_ticker = m["weights"].idxmax()
    p_last, f_last = fig_residual(EXAMPLE, m, example_ticker)
    model = m["models"][example_ticker]
    print(f"{example_ticker}: R2 {model.rsquared:.3f}, params {model.params.round(4).to_dict()}")
    print(f"  last resid {m['residuals'][example_ticker].iloc[-1]:.4f}, "
          f"resid std {m['residuals'][example_ticker].std():.4f}, "
          f"resid mean {m['residuals'][example_ticker].mean():.6f}, z {m['last'][example_ticker]:.3f}")
    print(f"  price {p_last:.2f} fair {f_last:.2f}")
    r2 = pd.Series({t: mm.rsquared for t, mm in m["models"].items()})
    print("R2 across the 20: median", round(r2.median(), 3), "min", round(r2.min(), 3))

    # --- the signal across the whole backtest ------------------------------
    fwd = forward_returns(models, rebalance_days, spy_days)
    g = fig_signal(fwd)
    print("\nforward excess by z bucket:\n", g.round(4))
    picked = fwd[fwd.z < ENTRY]
    print("picked stock-months", len(picked), "mean excess", round(picked.excess.mean() * 100, 2),
          "hit", round((picked.excess > 0).mean(), 3))
    print("corr(z, excess)", round(fwd[["z", "excess"]].corr().iloc[0, 1], 4))

    ew, picks = monthly_baselines(fwd, rebalance_days)
    years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
    print(f"frictionless: 20-name EW x{ew[-1]:.2f} ({ew[-1] ** (1 / years) - 1:.1%}/yr), "
          f"picks x{picks[-1]:.2f} ({picks[-1] ** (1 / years) - 1:.1%}/yr)")

    counts, top = fig_holdings(lean, rebalance_days)
    print("\npicks per month:", pd.Series(counts).value_counts().sort_index().to_dict())
    print("months with a 100% single name:", [(d.date(), list(lean[d])) for d in rebalance_days if len(lean.get(d, {})) == 1])

    # --- results ------------------------------------------------------------
    strat = equity(load_result("pca_stat_arbitrage")).loc[START:]
    short = equity(load_result("pca_stat_arbitrage_short")).loc[START:]
    spy = equity(load_result("spy_buy_and_hold_2021_2026")).loc[START:]
    ew_curve = pd.Series(ew[1:] * 100_000, index=[rebalance_days[i + 1] if i + 1 < len(rebalance_days)
                                                   else strat.index[-1] for i in range(len(rebalance_days))])
    ew_curve = pd.concat([pd.Series({strat.index[0]: 100_000.0}), ew_curve])
    fig_equity([("程式原樣（買進便宜的）", strat, BLUE), ("照描述放空", short, RED),
                ("SPY 買進持有", spy, ORANGE), ("20 檔等權持有（每月重設）", ew_curve, AQUA)])
    for name, s in [("strategy", strat), ("short", short), ("spy", spy)]:
        print(name, {k: round(v, 4) for k, v in stats(s).items()})
    print(pd.DataFrame({"strategy": yearly(strat), "short": yearly(short), "spy": yearly(spy)}).round(1))

    # monthly returns by rebalance period, to find the best and worst months
    rets = []
    for i, d in enumerate(rebalance_days):
        stop = rebalance_days[i + 1] if i + 1 < len(rebalance_days) else strat.index[-1] + pd.Timedelta(days=1)
        seg = strat.loc[:stop - pd.Timedelta(days=1)]
        before = strat.loc[:d - pd.Timedelta(days=1)]
        if before.empty:
            continue
        rets.append((d, seg.iloc[-1] / before.iloc[-1] - 1, list(lean.get(d, {}))))
    rets = sorted(rets, key=lambda r: r[1])
    print("\nworst months:", [(d.date(), round(r * 100, 1), t) for d, r, t in rets[:6]])
    print("best months:", [(d.date(), round(r * 100, 1), t) for d, r, t in rets[-6:]])

    o = orders(load_result("pca_stat_arbitrage"))
    print("\norders by status:", o.status.value_counts().to_dict())
    print(o[o.status != 3])


if __name__ == "__main__":
    main()
