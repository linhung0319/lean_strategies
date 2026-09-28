"""Regenerate the figures and numbers in strategy-explained.md.

Rebuilds the strategy's inputs -- the monthly price / dollar-volume / market-cap
funnel and the 252-day momentum -- from the same data folder LEAN read, checks
that they reproduce every rebalance LEAN logged, then draws the figures.

    uv run run_local.py "Momentum Effect Top5" --data-folder data-sp500
    uv run run_local.py "SPY Buy & Hold 2021-2026" --data-folder data-sp500
    uv run python research/momentum_effect/explain_figures.py

Needs `data-sp500/` (see "Constituents universes" in README.md) and both
backtests above: the picks, trades and equity curve come from their result
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

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data-sp500" / "equity" / "usa"
OUT = Path(__file__).resolve().parent
RESULTS = REPO / "results"

START, END = "2021-09-16", "2026-09-15"
LOOKBACK, LIQUID, UNIVERSE, PORTFOLIO = 252, 100, 50, 5
EXAMPLE = pd.Timestamp("2024-01-02")

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


def momentum(ticker):
    """LEAN's MOMP(252): percent change against the close 252 bars earlier."""
    return read_bars(ticker)["adj"].pct_change(LOOKBACK) * 100


def constituents(file_date):
    path = DATA / "universes" / "etf" / "spy" / f"{file_date:%Y%m%d}.csv"
    rows = [line.split(",") for line in path.read_text().splitlines() if line.strip()]
    return {r[0]: float(r[3]) for r in rows if r[3]}, len(rows)


def funnel(day, files):
    """The selection that runs at 00:00 on `day`: it sees the universe file
    and the bar of the previous session."""
    file_date = files[files < day][-1]
    weights, members = constituents(file_date)
    rows = []
    for ticker, weight in weights.items():
        bars = read_bars(ticker)
        if bars is None:
            continue
        before = bars.loc[:day - pd.Timedelta(days=1)]
        if before.empty:
            continue
        bar = before.iloc[-1]
        rows.append(dict(ticker=ticker, weight=weight, price=bar.close,
                         dollar_volume=bar.close * bar.volume))
    frame = pd.DataFrame(rows).set_index("ticker")
    above5 = frame[frame.price > 5]
    liquid = above5.sort_values("dollar_volume").iloc[-LIQUID:]
    large = liquid.sort_values("weight", ascending=False).iloc[:UNIVERSE]
    return dict(members=members, priced=len(frame), above5=above5, liquid=liquid, large=large)


def picks_on(day, universe):
    """Top-5 momentum among `universe`, using the close before `day`."""
    scores = {}
    for ticker in universe:
        m = momentum(ticker).loc[:day - pd.Timedelta(days=1)]
        if len(m) and not np.isnan(m.iloc[-1]):
            scores[ticker] = m.iloc[-1]
    return pd.Series(scores).sort_values(ascending=False)


def lean_rebalances():
    log = (RESULTS / "momentum_effect_top5" / "MomentumEffect-log.txt").read_text(encoding="utf-8")
    out = {}
    for date, body in re.findall(r"^(\d{4}-\d{2}-\d{2}) 08:00:00 REBALANCE (.+)$", log, re.M):
        out[pd.Timestamp(date)] = [(p.split(":")[0], float(p.split(":")[1])) for p in body.split(",")]
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
                         qty=order["quantity"], price=order["price"],
                         value=abs(order["quantity"] * order["price"])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# reproduce the decisions
# --------------------------------------------------------------------------

def build():
    files = pd.DatetimeIndex(sorted(pd.Timestamp(p.stem) for p in
                                    (DATA / "universes" / "etf" / "spy").glob("*.csv")))
    spy_days = read_bars("SPY").loc[START:END].index
    # Month starts inside the window, plus the 2021-09-01 selection that the
    # 45-day warm-up runs and the first (2021-09-16) rebalance trades on.
    month_starts = pd.Series(spy_days, index=spy_days).groupby(spy_days.to_period("M")).first()
    selection_days = [pd.Timestamp("2021-09-01")] + [d for d in month_starts if d > pd.Timestamp(START)]
    rebalance_days = [pd.Timestamp(START)] + selection_days[1:]
    universes = {d: funnel(d, files) for d in selection_days}
    return files, spy_days, selection_days, rebalance_days, universes


def verify(rebalance_days, selection_days, universes, lean):
    """Every LEAN rebalance must pick exactly the five names we recompute."""
    picks, mismatches = {}, 0
    for rebalance, selection in zip(rebalance_days, selection_days):
        ranked = picks_on(rebalance, universes[selection]["large"].index)
        mine = list(ranked.index[:PORTFOLIO])
        picks[rebalance] = ranked
        theirs = [t for t, _ in lean.get(rebalance, [])]
        if mine != theirs:
            mismatches += 1
            print("mismatch", rebalance.date(), mine, theirs)
    print(f"verified {len(rebalance_days)} rebalances, {mismatches} mismatches")
    return picks


def equal_weight_universe(rebalance_days, selection_days, universes, spy_days):
    """Hold all 50 names equally, reset monthly, close to close, no fees."""
    daily = []
    for i, (rebalance, selection) in enumerate(zip(rebalance_days, selection_days)):
        stop = rebalance_days[i + 1] if i + 1 < len(rebalance_days) else pd.Timestamp(END) + pd.Timedelta(days=1)
        days = spy_days[(spy_days >= rebalance) & (spy_days < stop)]
        names = universes[selection]["large"].index
        closes = pd.DataFrame({t: read_bars(t)["adj"] for t in names}).ffill()
        base = closes.loc[:rebalance - pd.Timedelta(days=1)].iloc[-1]
        path = (closes.reindex(days) / base).mean(axis=1)
        daily.append(path)
    level, pieces = 1.0, []
    for path in daily:
        pieces.append(path * level)
        level = pieces[-1].iloc[-1]
    return pd.concat(pieces) * 100_000


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_funnel(f, ranked):
    """One month's funnel: the 50 names, their momentum, the five kept."""
    large = f["large"].copy()
    large["momp"] = ranked.reindex(large.index)
    large = large.sort_values("momp")
    top = set(ranked.index[:PORTFOLIO])
    fig, ax = plt.subplots(figsize=(10, 11))
    colors = [BLUE if t in top else PALE for t in large.index]
    ax.barh(range(len(large)), large.momp, color=colors, height=0.72)
    ax.set_yticks(range(len(large)))
    ax.set_yticklabels(large.index, fontsize=8, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.grid(axis="y", visible=False)
    for i, (t, row) in enumerate(large.iterrows()):
        if t in top:
            ax.text(row.momp + 3, i, f"+{row.momp:.0f}%", va="center", color=INK, fontsize=9)
    ax.set_xlabel("過去 252 個交易日的漲跌幅（%）")
    ax.set_title(f"{EXAMPLE:%Y-%m-%d} 換股日：50 檔大型股的 12 個月動能，深藍 = 被選中的前 5 名",
                 loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2-ranking.png", dpi=130)
    plt.close(fig)


def fig_momentum(ticker="NVDA"):
    bars = read_bars(ticker)
    end = bars.loc[:EXAMPLE - pd.Timedelta(days=1)].index[-1]
    i = bars.index.get_loc(end)
    begin = bars.index[i - LOOKBACK]
    s = bars.loc[begin - pd.Timedelta(days=40):end + pd.Timedelta(days=5), "adj"]
    p0, p1 = bars.adj.loc[begin], bars.adj.loc[end]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(s.index, s, color=INK, lw=1.6)
    ax.axvspan(begin, end, color=BLUE, alpha=0.07, lw=0)
    for d, p, label, dy in [(begin, p0, f"252 個交易日前\n{begin:%Y-%m-%d}  {p0:.2f}", 30),
                            (end, p1, f"換股前一天\n{end:%Y-%m-%d}  {p1:.2f}", -46)]:
        ax.scatter(d, p, s=80, color=BLUE, zorder=5, edgecolor=PAPER, lw=2)
        ax.annotate(label, (d, p), xytext=(0, dy), textcoords="offset points",
                    ha="center", color=INK, fontsize=9)
    ax.text(begin + pd.Timedelta(days=12), s.max() * 0.95,
            f"動能 = {p1:.2f} ÷ {p0:.2f} - 1 = +{(p1 / p0 - 1) * 100:.0f}%",
            ha="left", va="top", color=INK, fontsize=12)
    ax.set_ylabel("還原權值後的收盤價（美元）")
    ax.set_title(f"{ticker} 的 12 個月動能怎麼算", loc="left", color=INK, fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(OUT / "fig1-momentum.png", dpi=130)
    plt.close(fig)
    return begin, end, p0, p1


GROUPS = {
    "能源": {"XOM", "CVX", "COP", "VLO", "SLB"},
    "金融": {"SCHW", "GS", "WFC", "MS", "AXP"},
    "醫療": {"LLY", "PFE", "ABBV", "UNH", "ELV", "CI", "BMY", "VRTX", "GILD", "SYK", "ISRG"},
    "半導體／硬體": {"NVDA", "AMAT", "AVGO", "AMD", "MU", "LRCX", "QCOM", "ADI", "INTC", "SNDK", "DELL"},
}


def group_of(t):
    return next((g for g, members in GROUPS.items() if t in members), "其他")


def fig_holdings(lean):
    days = sorted(lean)
    names = []
    for d in days:
        for t, _ in lean[d]:
            if t not in names:
                names.append(t)
    order = sorted(names, key=lambda t: (list(GROUPS).index(group_of(t)) if group_of(t) in GROUPS else 9,
                                         min(i for i, d in enumerate(days) if t in dict(lean[d])))
                   )
    color = {"能源": ORANGE, "金融": VIOLET, "醫療": AQUA, "半導體／硬體": BLUE, "其他": "#9a9993"}
    fig, ax = plt.subplots(figsize=(12, 0.24 * len(order) + 1.6))
    for row, t in enumerate(order):
        for i, d in enumerate(days):
            if t in dict(lean[d]):
                ax.add_patch(plt.Rectangle((i, row), 0.9, 0.8, color=color[group_of(t)], lw=0))
    ax.set_xlim(0, len(days))
    ax.set_ylim(len(order), 0)
    ax.set_yticks([r + 0.4 for r in range(len(order))])
    ax.set_yticklabels(order, fontsize=7.5, color=INK)
    ticks = [i for i, d in enumerate(days) if d.month == 1]
    ax.set_xticks([t + 0.45 for t in ticks])
    ax.set_xticklabels([f"{days[i]:%Y}" for i in ticks])
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color.values()]
    ax.legend(handles, list(color), loc="upper left", bbox_to_anchor=(0, -0.03 * 40 / len(order)),
              ncol=5, frameon=False)
    ax.set_title("每個月持有哪 5 檔（一格 = 一個月，顏色 = 產業）", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig3-holdings.png", dpi=130)
    plt.close(fig)


def fig_equity(strategy, spy, ew):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    series = [("本策略（動能前 5）", strategy, BLUE), ("50 檔等權持有", ew, AQUA), ("SPY 買進持有", spy, ORANGE)]
    for (label, s, c), dy in zip(series, [0, 5, -5]):
        top.plot(s.index, s / 1000, color=c, lw=2, label=label)
        top.annotate(f"{s.iloc[-1] / 10000:.1f} 萬",
                     (s.index[-1], s.iloc[-1] / 1000), xytext=(6, dy), textcoords="offset points",
                     va="center", color=INK, fontsize=9)
        dd = s / s.cummax() - 1
        bottom.plot(dd.index, dd * 100, color=c, lw=1.6)
    top.set_ylabel("資產（千美元）")
    top.legend(loc="upper left", frameon=False)
    top.set_title("10 萬美元的走勢，與從最高點回落的幅度", loc="left", color=INK, fontsize=12)
    bottom.set_ylabel("回撤（%）")
    bottom.xaxis.set_major_locator(mdates.YearLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig4-equity.png", dpi=130)
    plt.close(fig)


def fig_yearly(strategy, spy, ew):
    def yearly(s):
        y = s.groupby(s.index.year).last()
        first = pd.Series({s.index[0].year - 1: s.iloc[0]})
        return (pd.concat([first, y]).pct_change().dropna() * 100)
    table = pd.DataFrame({"本策略": yearly(strategy), "50 檔等權": yearly(ew), "SPY": yearly(spy)})
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.26
    x = np.arange(len(table))
    for k, (col, c) in enumerate(zip(table.columns, [BLUE, AQUA, ORANGE])):
        ax.bar(x + (k - 1) * width, table[col], width - 0.03, color=c, label=col)
        for xi, v in zip(x, table[col]):
            ax.text(xi + (k - 1) * width, v + (1.5 if v >= 0 else -4.5), f"{v:.0f}",
                    ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(min(-25, table.min().min() - 8), table.max().max() + 8)
    labels = [str(y) for y in table.index]
    labels[0] += "\n(9/16 起)"
    labels[-1] += "\n(至 9/15)"
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("當年報酬（%）")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("逐年報酬：動能策略在哪一年贏、哪一年輸", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig5-yearly.png", dpi=130)
    plt.close(fig)
    return table


def stats(s):
    years = (s.index[-1] - s.index[0]).days / 365.25
    return dict(end=s.iloc[-1], cagr=(s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1,
                dd=(1 - s / s.cummax()).max())


def main():
    use_cjk_font()
    files, spy_days, selection_days, rebalance_days, universes = build()
    lean = lean_rebalances()
    picks = verify(rebalance_days, selection_days, universes, lean)

    f = universes[EXAMPLE]
    print(f"\n{EXAMPLE.date()} funnel: file rows {f['members']}, with bars {f['priced']}, "
          f"> $5 {len(f['above5'])}, top-100 $vol {len(f['liquid'])}, top-50 cap {len(f['large'])}")
    print("  dollar volume cut-off (#100): %.0f M" % (f["liquid"].dollar_volume.min() / 1e6))
    print("  dropped by the cap step:", ", ".join(
        f"{t}({f['liquid'].loc[t].dollar_volume / 1e6:.0f}M)"
        for t in f["liquid"].index.difference(f["large"].index)))
    print("  top 10 momentum:", picks[EXAMPLE].head(10).round(1).to_dict())
    print("  bottom 5 momentum:", picks[EXAMPLE].tail(5).round(1).to_dict())
    print("  median momentum of the 50:", round(picks[EXAMPLE].median(), 1))
    cheap = [(d.date(), len(universes[d]["above5"]), universes[d]["priced"]) for d in selection_days]
    print("  names failing $5 per selection (min/max):",
          min(p - a for _, a, p in cheap), max(p - a for _, a, p in cheap))

    begin, end, p0, p1 = fig_momentum()
    print(f"\nNVDA momentum: {begin.date()} {p0:.2f} -> {end.date()} {p1:.2f} = {(p1/p0-1)*100:.1f}%")
    fig_funnel(f, picks[EXAMPLE])
    fig_holdings(lean)

    # turnover: how many of the five change each month
    days = sorted(lean)
    changes = [len(set(t for t, _ in lean[b]) - set(t for t, _ in lean[a])) for a, b in zip(days, days[1:])]
    print(f"\nnames replaced per rebalance: mean {np.mean(changes):.2f}, "
          f"distribution {pd.Series(changes).value_counts().sort_index().to_dict()}")
    counts = pd.Series([t for d in days for t, _ in lean[d]]).value_counts()
    print("months held, top 12:", counts.head(12).to_dict())
    print("distinct names:", len(counts))

    strategy = equity(load_result("momentum_effect_top5")).loc[START:END]
    spy = equity(load_result("spy_buy_and_hold_2021_2026")).loc[START:END]
    spy = spy * 100_000 / spy.iloc[0]  # that run starts with its own cash
    ew = equal_weight_universe(rebalance_days, selection_days, universes, spy_days)
    for name, s in [("strategy", strategy), ("equal-weight 50", ew), ("spy", spy)]:
        st = stats(s)
        print(f"{name:16s} end {st['end']:,.0f}  cagr {st['cagr']:.2%}  maxdd {st['dd']:.2%}")
    dd = strategy / strategy.cummax() - 1
    trough = dd.idxmin()
    print("strategy max drawdown trough", trough.date(), "peak", strategy.loc[:trough].idxmax().date(),
          "recovered", strategy.loc[trough:][strategy.loc[trough:] >= strategy.loc[:trough].max()].index[:1].tolist())
    fig_equity(strategy, spy, ew)
    table = fig_yearly(strategy, spy, ew)
    print(table.round(1))

    o = orders(load_result("momentum_effect_top5"))
    print(f"\norders {len(o)}, at 00:00 (removed from universe) {sum(o.time.dt.hour == 0)}, "
          f"at 08:00 (rebalance) {sum(o.time.dt.hour == 8)}")
    monthly = strategy.resample("ME").last().pct_change().dropna()
    print("best month", monthly.idxmax().date(), f"{monthly.max():.1%}",
          "worst month", monthly.idxmin().date(), f"{monthly.min():.1%}")


if __name__ == "__main__":
    main()
