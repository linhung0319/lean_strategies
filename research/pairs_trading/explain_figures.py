"""Regenerate the figures and numbers in strategy-explained.md.

Rebuilds every formation (12-month normalized prices -> SSD -> 20 pairs) from
the same data folder LEAN read, checks it against the pairs LEAN logged, then
draws the figures from the LEAN result files.

    uv run run_local.py "Pairs Trading GGR" --data-folder data-sp500
    uv run run_local.py "SPY Buy & Hold 2021-2026" --data-folder data-sp500
    uv run run_local.py "Pairs Trading GGR 1997-2026" --data-folder data-sp500-long
    uv run python research/pairs_trading/explain_figures.py
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

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data-sp500" / "equity" / "usa"
OUT = Path(__file__).resolve().parent
RESULTS = REPO / "results"
FORMATION, PAIRS, ENTRY = 252, 20, 2.0
SAME_COMPANY = [{"GOOG", "GOOGL"}, {"FOX", "FOXA"}, {"NWS", "NWSA"}, {"DISCA", "DISCK"},
                {"UA", "UAA"}, {"BF.A", "BF.B"}, {"LEN", "LEN.B"}, {"BRK.A", "BRK.B"},
                {"VIA", "VIAB"}, {"CMCSA", "CMCSK"}]

FONT_CANDIDATES = ["Microsoft JhengHei", "Noto Sans TC", "Microsoft YaHei", "DejaVu Sans"]
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e4e1", "#fcfcfb"
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
    """Raw bars plus LEAN's Adjusted close (raw x factor-file scale)."""
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


def universe_files():
    names = sorted(p.stem for p in (DATA / "universes" / "etf" / "spy").glob("*.csv"))
    return pd.DatetimeIndex(pd.to_datetime(names, format="%Y%m%d"))


def constituents(file_date):
    path = DATA / "universes" / "etf" / "spy" / f"{file_date:%Y%m%d}.csv"
    rows = [line.split(",") for line in path.read_text().splitlines() if line.strip()]
    return [r[0] for r in rows if r[3]]


def formation_closes(day, files):
    """What self.history(symbols, 252, DAILY).close.unstack(0).dropna(axis=1)
    returns at 00:00 on `day`."""
    tickers = constituents(files[files < day][-1])
    frame = {}
    for t in tickers:
        bars = read_bars(t)
        if bars is None:
            continue
        frame[t] = bars["adj"].loc[:day - pd.Timedelta(days=1)].iloc[-FORMATION:]
    closes = pd.DataFrame(frame)
    closes = closes.iloc[-FORMATION:]
    return closes.dropna(axis=1)


def ssd_matrix(closes):
    norm = closes.values / closes.values[0]
    sq = (norm ** 2).sum(axis=0)
    ssd = sq[:, None] + sq[None, :] - 2.0 * norm.T @ norm
    np.fill_diagonal(ssd, np.inf)
    index = {t: i for i, t in enumerate(closes.columns)}
    for group in SAME_COMPANY:
        members = [index[t] for t in group if t in index]
        for i in members:
            for j in members:
                ssd[i, j] = np.inf
    return norm, ssd


def form_pairs(closes):
    norm, ssd = ssd_matrix(closes)
    best = ssd.argmin(axis=1)
    cand = {}
    for i, j in enumerate(best):
        if np.isfinite(ssd[i, j]):
            cand[(min(i, j), max(i, j))] = ssd[i, j]
    chosen = sorted(cand.items(), key=lambda kv: kv[1])[:PAIRS]
    cols = list(closes.columns)
    return [(cols[i], cols[j], float(d), float((norm[:, i] - norm[:, j]).std())) for (i, j), d in chosen], ssd


# --------------------------------------------------------------------------
# LEAN results
# --------------------------------------------------------------------------

def lean_log(folder="pairs_trading_ggr"):
    return (RESULTS / folder / "PairsTrading-log.txt").read_text(encoding="utf-8")


def lean_pairs(log):
    cycles = {}
    for day, cycle, a, b, ssd, sigma in re.findall(
            r"^(\S+) \S+ PAIR cycle=(\d+) (\S+)/(\S+) ssd=([\d.]+) sigma=([\d.]+)", log, re.M):
        cycles.setdefault(pd.Timestamp(day), []).append((a, b, float(ssd), float(sigma)))
    return cycles


def lean_trades(log):
    opened, trades = {}, []
    for line in log.splitlines():
        m = re.match(r"(\S+) \S+ (OPEN|CLOSE) (\S+)/(\S+) (.*)", line)
        if not m:
            continue
        day, kind, a, b, rest = m.groups()
        px = re.search(r"px=([\d.]+)/([\d.]+)", rest)
        if kind == "OPEN":
            q = re.search(r"qty=(-?\d+)/(-?\d+)", rest)
            spread = float(re.search(r"spread=(-?[\d.]+)", rest)[1])
            limit = float(re.search(r"limit=([\d.]+)", rest)[1])
            opened[(a, b)] = dict(open=pd.Timestamp(day), qa=int(q[1]), qb=int(q[2]),
                                  pa0=float(px[1]), pb0=float(px[2]), spread0=spread, limit=limit)
        else:
            o = opened.pop((a, b))
            reason = re.search(r"reason=(.+?) spread", rest)[1]
            pa1, pb1 = float(px[1]), float(px[2])
            trades.append(dict(a=a, b=b, **o, close=pd.Timestamp(day), reason=reason,
                               pa1=pa1, pb1=pb1,
                               pnl=o["qa"] * (pa1 - o["pa0"]) + o["qb"] * (pb1 - o["pb0"])))
    return pd.DataFrame(trades)


def load_result(folder):
    path = next(p for p in (RESULTS / folder).glob("*.json")
                if not p.name.endswith(("summary.json", "order-events.json"))
                and not p.name.startswith("data-monitor"))
    return json.loads(path.read_text(encoding="utf-8"))


def equity(result):
    values = result["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
    series = pd.Series({pd.Timestamp(v[0], unit="s"): v[4] for v in values})
    return series.groupby(series.index.normalize()).last()


def stats(folder):
    path = next((RESULTS / folder).glob("*-summary.json"))
    return json.loads(path.read_text(encoding="utf-8"))["statistics"]


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def month_axis(ax, fmt="%Y-%m"):
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def fig_normalize(closes, a, b):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(closes.index, closes[a], color=BLUE, lw=2, label=a)
    ax.plot(closes.index, closes[b], color=ORANGE, lw=2, label=b)
    ax.set_title(f"① 原始股價（美元）：{a} 和 {b} 價位差很多，看不出像不像", loc="left", fontsize=10.5)
    ax.legend(frameon=False)
    month_axis(ax)
    ax = axes[1]
    na, nb = closes[a] / closes[a].iloc[0], closes[b] / closes[b].iloc[0]
    ax.fill_between(closes.index, na, nb, color=PALE, lw=0, label="兩條線的差距")
    ax.plot(closes.index, na, color=BLUE, lw=2, label=a)
    ax.plot(closes.index, nb, color=ORANGE, lw=2, label=b)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_title("② 都從 1 開始（＝每 1 塊錢變成多少）：幾乎疊在一起", loc="left", fontsize=10.5)
    ax.legend(frameon=False)
    month_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig1-normalize.png", dpi=130)
    plt.close(fig)


def fig_ssd(closes, examples):
    fig, axes = plt.subplots(1, len(examples), figsize=(12, 3.6), sharey=True)
    for ax, (a, b) in zip(axes, examples):
        na, nb = closes[a] / closes[a].iloc[0], closes[b] / closes[b].iloc[0]
        ssd = float(((na - nb) ** 2).sum())
        ax.fill_between(closes.index, na, nb, color=PALE, lw=0)
        ax.plot(closes.index, na, color=BLUE, lw=2, label=a)
        ax.plot(closes.index, nb, color=ORANGE, lw=2, label=b)
        ax.set_title(f"{a} vs {b}\nSSD = {ssd:.2f}", loc="left", fontsize=10.5)
        ax.legend(frameon=False, loc="upper left")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 9]))
        month_axis(ax)
    axes[0].set_ylabel("正規化價格（起點 = 1）")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig2-ssd.png", dpi=130)
    plt.close(fig)


def fig_trade(pair_trades, base_day, end_day, sigma, name, title):
    a, b = pair_trades.iloc[0]["a"], pair_trades.iloc[0]["b"]
    ba, bb = read_bars(a)["adj"], read_bars(b)["adj"]
    base_a = ba.loc[:base_day - pd.Timedelta(days=1)].iloc[-1]
    base_b = bb.loc[:base_day - pd.Timedelta(days=1)].iloc[-1]
    window = slice(base_day - pd.Timedelta(days=1), end_day)
    na, nb = ba.loc[window] / base_a, bb.loc[window] / base_b
    spread = (na - nb).dropna()
    limit = ENTRY * sigma

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(na.index, na, color=BLUE, lw=2, label=a)
    ax.plot(nb.index, nb, color=ORANGE, lw=2, label=b)
    ax.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel("正規化價格（交易期起點 = 1）")
    ax.legend(frameon=False, loc="lower left")
    ax = axes[1]
    ax.axhspan(-limit, limit, color=PALE, lw=0, label=f"±2σ = ±{limit:.3f}（不動區）")
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(spread.index, spread, color=INK, lw=2, label=f"價差 = {a} 減 {b}")
    ax.set_ylabel("價差")
    lines = []
    for n, (_, t) in enumerate(pair_trades.iterrows(), 1):
        for x in axes:
            x.axvspan(t["open"], t["close"], color=YELLOW, alpha=0.12, lw=0)
        y0, y1 = spread.asof(t["open"]), spread.asof(t["close"])
        ax.plot([t["open"]], [y0], "o", ms=9, color=RED, mec=PAPER, mew=2, zorder=5)
        ax.plot([t["close"]], [y1], "o", ms=9, color=GREEN, mec=PAPER, mew=2, zorder=5)
        ax.annotate(str(n), (t["open"], y0), xytext=(-8, 6 if y0 > 0 else -14),
                    textcoords="offset points", fontsize=9, color=INK, ha="right", fontweight="bold")
        why = "回到 0" if t["reason"] == "converged" else "到期強制平倉"
        lines.append(f"{n}. {t['open']:%m-%d} 進場（價差 {y0:+.3f}）→ {t['close']:%m-%d} {why}，"
                     f"損益 {t['pnl']:+,.0f} 美元")
    ax.text(0.99, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=INK, bbox=dict(boxstyle="round,pad=0.4", fc=PAPER, ec=GRID))
    ax.plot([], [], "o", color=RED, label="進場（紅）")
    ax.plot([], [], "o", color=GREEN, label="出場（綠）")
    ax.legend(frameon=False, loc="lower left", fontsize=9, ncol=2)
    month_axis(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def fig_outcomes(trades):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    bins = np.arange(-3000, 1001, 100)
    for ax, (reason, color, label) in zip(axes, [
            ("converged", BLUE, "價差回到 0（收斂）"), ("cycle end", ORANGE, "6 個月到期還沒收斂（強制平倉）")]):
        sub = trades[trades.reason == reason]
        ax.hist(sub.pnl.clip(-3000, 1000), bins=bins, color=color, edgecolor=PAPER, lw=1)
        ax.axvline(0, color=MUTED, lw=1)
        ax.axvline(sub.pnl.mean(), color=INK, lw=1.5, ls="--")
        ax.set_title(f"{label}\n{len(sub)} 筆，平均 {sub.pnl.mean():+,.0f}、合計 {sub.pnl.sum():+,.0f} 美元",
                     loc="left", fontsize=10.5)
        ax.set_xlabel("每筆配對交易的損益（美元，訊號價格計，未扣手續費）")
    axes[0].set_ylabel("交易筆數")
    fig.tight_layout()
    fig.savefig(OUT / "fig5-outcomes.png", dpi=130)
    plt.close(fig)


def fig_equity(curves, name, title, logy=False):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = [BLUE, ORANGE, AQUA]
    for (label, s), c in zip(curves.items(), colors):
        ax.plot(s.index, s / s.iloc[0] * 100, color=c, lw=2, label=label)
        ax.annotate(f"{s.iloc[-1] / s.iloc[0] * 100:,.0f}", (s.index[-1], s.iloc[-1] / s.iloc[0] * 100),
                    xytext=(4, 0), textcoords="offset points", color=INK, fontsize=9, va="center")
    ax.axhline(100, color=MUTED, lw=0.8, ls=":")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("資產（起點 = 100）")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)


def fig_yearly(series, name, title):
    yearly = series.groupby(series.index.year).last()
    first = series.iloc[0]
    returns = (yearly / yearly.shift(1).fillna(first) - 1) * 100
    fig, ax = plt.subplots(figsize=(11, 3.8))
    colors = [BLUE if r >= 0 else ORANGE for r in returns]
    ax.bar(returns.index, returns, color=colors, width=0.75)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axvspan(returns.index.min() - 0.5, 2002.5, color=PALE, alpha=0.5, lw=0, zorder=0)
    ax.text(returns.index.min() - 0.3, returns.max() * 0.95, "GGR 論文樣本\n涵蓋的年份", color=MUTED, fontsize=9, va="top")
    ax.set_ylabel("當年報酬（%）")
    ax.set_title(title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    return returns


# --------------------------------------------------------------------------

def main():
    use_cjk_font()
    log = lean_log()
    cycles = lean_pairs(log)
    files = universe_files()

    # 1. Reproduce every formation.
    mismatches = 0
    formations = {}
    for day, logged in cycles.items():
        closes = formation_closes(day, files)
        mine, ssd = form_pairs(closes)
        formations[day] = (closes, ssd)
        same = [(a, b) for a, b, *_ in mine] == [(a, b) for a, b, *_ in logged]
        worst = max(abs(m[2] - l[2]) for m, l in zip(mine, logged)) if same else float("nan")
        mismatches += not same
        print(f"{day:%Y-%m-%d}  stocks={closes.shape[1]}  pairs match={same}  max |SSD diff|={worst:.5f}")
    print("formations not matching:", mismatches)

    trades = lean_trades(log)
    trades.to_csv(OUT / "trades.csv", index=False)
    print(trades.groupby("reason").pnl.agg(["count", "mean", "sum", lambda s: (s > 0).mean()]))
    print(trades.assign(days=(trades.close - trades.open).dt.days).groupby("reason").days.median())
    print("open pairs per day: mean",
          np.mean([((trades.open <= d) & (trades.close > d)).sum()
                   for d in pd.bdate_range("2021-10-01", "2026-09-14")]))

    # 2. Example cycle: 2024-03-01.
    day = pd.Timestamp("2024-03-01")
    closes, ssd = formations[day]
    print("cycle 2024-03-01 pairs:", [(a, b, round(d, 3), round(s, 4)) for a, b, d, s in cycles[day]])
    finite = ssd[np.isfinite(ssd)]
    print("all-pair SSD median:", np.median(finite) / 1, " pairs:", finite.size // 2)
    fig_normalize(closes, "AME", "NDSN")
    fig_ssd(closes, [("ETR", "PPL"), ("AME", "NDSN"), ("AME", "NVDA")])
    a, b = closes["AME"], closes["NDSN"]
    print("AME/NDSN formation first/last:", a.iloc[0], a.iloc[-1], b.iloc[0], b.iloc[-1])

    win = trades[(trades.a == "AME") & (trades.b == "NDSN") & (trades.open >= day)
                 & (trades.open < "2024-09-01")]
    print("win trades:", win.to_dict("records"))
    fig_trade(win, day, pd.Timestamp("2024-09-03"), [s for x, y, _, s in cycles[day] if (x, y) == ("AME", "NDSN")][0],
              "fig3-trade-win.png", "成功的例子：AME / NDSN（2024 年 3–8 月交易期）")
    day7 = pd.Timestamp("2024-09-03")
    day7 = [d for d in cycles if d.year == 2024 and d.month == 9][0]
    loss = trades[(trades.a == "EIX") & (trades.b == "NI")]
    print("loss trade:", loss.to_dict("records"))
    fig_trade(loss, day7, pd.Timestamp("2025-03-03"), [s for x, y, _, s in cycles[day7] if (x, y) == ("EIX", "NI")][0],
              "fig4-trade-loss.png", "失敗的例子：EIX / NI（2024-09 → 2025-03 交易期）")
    fig_outcomes(trades)

    # 3. Equity.
    pairs = equity(load_result("pairs_trading_ggr"))
    spy = equity(load_result("spy_buy_and_hold_2021_2026"))
    fig_equity({"配對交易 GGR": pairs, "SPY 買進持有": spy}, "fig6-equity.png",
               "2021-09 → 2026-09：配對交易 vs SPY（10 萬美元起）")
    for folder in ("pairs_trading_ggr", "spy_buy_and_hold_2021_2026"):
        s = stats(folder)
        print(folder, {k: s[k] for k in ["End Equity", "Compounding Annual Return", "Drawdown",
                                         "Sharpe Ratio", "Annual Standard Deviation", "Total Fees", "Total Orders"]})

    long_dir = RESULTS / "pairs_trading_ggr_1997_2026"
    if long_dir.exists() and list(long_dir.glob("*-summary.json")):
        lp = equity(load_result("pairs_trading_ggr_1997_2026"))
        ls = equity(load_result("spy_buy_and_hold_1997_2026"))
        fig_equity({"配對交易 GGR": lp, "SPY 買進持有": ls}, "fig7-long-equity.png",
                   "1997 → 2026：同一套規則放到 30 年（對數刻度）", logy=True)
        r = fig_yearly(lp, "fig8-long-yearly.png", "配對交易每年報酬（1997–2026）")
        print("long yearly:", r.round(1).to_dict())
        for folder in ("pairs_trading_ggr_1997_2026", "spy_buy_and_hold_1997_2026"):
            s = stats(folder)
            print(folder, {k: s[k] for k in ["End Equity", "Compounding Annual Return", "Drawdown",
                                             "Sharpe Ratio", "Annual Standard Deviation", "Total Fees"]})
        for lo, hi in [(1997, 2002), (2003, 2008), (2009, 2014), (2015, 2020), (2021, 2026)]:
            seg = lp[(lp.index.year >= lo) & (lp.index.year <= hi)]
            yrs = (seg.index[-1] - seg.index[0]).days / 365.25
            print(f"  {lo}-{hi}: CAGR {(seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1:.2%}")


if __name__ == "__main__":
    main()
