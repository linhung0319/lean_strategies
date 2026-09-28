"""Regenerate the figures and numbers in strategy-explained.md.

Rebuilds the strategy's inputs -- the daily top-5 Technology universe, the
100-day EMA and the 252-day MAX -- from the same data folder LEAN read, checks
that they reproduce every order LEAN placed, then draws the figures.

    uv run run_local.py "Tech Top5 EMA Reversion" --data-folder data-sp500
    uv run run_local.py "SPY Buy & Hold 2021-2026" --data-folder data-sp500
    uv run python research/tech_ema_reversion/explain_figures.py

Needs `data-sp500/` (see "Constituents universes" in README.md) and both
backtests above: the trades and the equity curve come from their result
files, not from a re-simulation.
"""

import io
import json
import zipfile
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
SIZE, EMA_N, MAX_N = 5, 100, 252

FONT_CANDIDATES = ["Microsoft JhengHei", "Noto Sans TC", "Microsoft YaHei",
                   "PingFang TC", "Heiti TC", "Noto Sans CJK TC", "DejaVu Sans"]
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e4e1", "#fcfcfb"
# Reference categorical palette, fixed order (dataviz skill, palette.md).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def load_technology():
    # Import the set straight out of the algorithm so the two cannot drift.
    source = (REPO / "algorithms" / "tech_ema_reversion.py").read_text(encoding="utf-8")
    block = source[source.index("TECHNOLOGY = frozenset({"):source.index("})") + 2]
    scope = {}
    exec(block, scope)
    return scope["TECHNOLOGY"]


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

def read_bars(ticker):
    """Daily bars, split- and dividend-adjusted the way LEAN's default
    DataNormalizationMode.Adjusted does it: raw price x the factor of the
    first factor-file row dated on or after the bar."""
    with zipfile.ZipFile(DATA / "daily" / f"{ticker.lower()}.zip") as archive:
        raw = archive.read(archive.namelist()[0]).decode()
    bars = pd.read_csv(io.StringIO(raw), header=None,
                       names=["date", "open", "high", "low", "close", "volume"])
    bars.index = pd.to_datetime(bars.pop("date").astype(str).str[:8], format="%Y%m%d")
    bars = bars[["open", "high", "low", "close"]] / 10000.0

    factors = pd.read_csv(DATA / "factor_files" / f"{ticker.lower()}.csv", header=None,
                          names=["date", "price", "split", "ref"])
    factors.index = pd.to_datetime(factors["date"].astype(str), format="%Y%m%d")
    scale = (factors["price"] * factors["split"]).sort_index()
    position = scale.index.searchsorted(bars.index, side="left")
    return bars.mul(scale.iloc[position].to_numpy(), axis=0)


def lean_ema(close, n):
    """LEAN's ExponentialMovingAverage: seeded with the SMA of the first n."""
    k = 2 / (n + 1)
    out = np.full(len(close), np.nan)
    values = close.to_numpy()
    out[n - 1] = values[:n].mean()
    for i in range(n, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return pd.Series(out, index=close.index)


def universe_by_day(keep=None, exclude=()):
    """Top-5 names by constituent weight, keyed by the trading day they are in
    effect on: the file dated D is what selection sees on D+1. `keep` limits
    the ranking to one sector; None ranks the whole index."""
    files = sorted((DATA / "universes" / "etf" / "spy").glob("*.csv"))
    selected = {}
    for path in files:
        rows = [line.split(",") for line in path.read_text().splitlines() if line.strip()]
        ranked = sorted((float(r[3]), r[0]) for r in rows
                        if r[3] and r[0] not in exclude and (keep is None or r[0] in keep))
        selected[pd.Timestamp(path.stem)] = [t for _, t in ranked[-SIZE:]]
    return pd.Series(selected)


def orders(slug="tech_top5_ema_reversion"):
    result = json.loads((RESULTS / slug / "TechEmaReversion.json")
                        .read_text(encoding="utf-8"))
    rows = []
    for order in result["orders"].values():
        submitted = pd.Timestamp(order["time"]).tz_convert("America/New_York").tz_localize(None)
        rows.append(dict(ticker=order["symbol"]["value"], submitted=submitted,
                         filled=pd.Timestamp(order["lastFillTime"]).tz_convert(
                             "America/New_York").tz_localize(None).normalize(),
                         qty=order["quantity"], price=order["price"]))
    return pd.DataFrame(rows), result


def equity(result):
    values = result["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
    series = pd.Series({pd.Timestamp(v[0], unit="s"): v[4] for v in values})
    return series.groupby(series.index.normalize()).last()


def spy_equity():
    folder = RESULTS / "spy_buy_and_hold_2021_2026"
    path = next(p for p in folder.glob("*.json")
                if not p.name.endswith(("summary.json", "order-events.json"))
                and not p.name.startswith("data-monitor"))
    return equity(json.loads(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# reproduce the decisions
# --------------------------------------------------------------------------

def build():
    technology = load_technology()
    universe = universe_by_day(technology)
    tickers = sorted({t for names in universe for t in names})
    bars = {t: read_bars(t) for t in tickers}
    signals = {}
    for t, b in bars.items():
        signals[t] = pd.DataFrame({
            "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
            "ema": lean_ema(b["close"], EMA_N),
            # MAX's default selector on a TradeBar subscription is High, and
            # the window includes the bar that just closed.
            "max": b["high"].rolling(MAX_N).max(),
            "prior_max": b["high"].rolling(MAX_N).max().shift(1),
        })
    spy_days = read_bars("SPY").loc[START:END].index
    return technology, universe, signals, spy_days


def classify(trades, universe, signals):
    """Pair buys with sells and label why each position was closed."""
    trades = trades.sort_values("submitted").reset_index(drop=True)
    open_positions, closed = {}, []
    for _, t in trades.iterrows():
        if t.qty > 0:
            open_positions[t.ticker] = t
            continue
        entry = open_positions.pop(t.ticker)
        # Midnight submissions come from on_securities_changed.
        reason = "跌出前五" if t.submitted.hour == 0 else "創一年新高"
        closed.append(dict(ticker=t.ticker, entry=entry.filled, entry_px=entry.price,
                           exit=t.filled, exit_px=t.price, reason=reason))
    for ticker, entry in open_positions.items():
        last = signals[ticker]["close"].loc[:END].iloc[-1]
        closed.append(dict(ticker=ticker, entry=entry.filled, entry_px=entry.price,
                           exit=pd.Timestamp(END), exit_px=last, reason="仍持有"))
    frame = pd.DataFrame(closed).sort_values("entry").reset_index(drop=True)
    frame["ret"] = frame.exit_px / frame.entry_px - 1
    frame["days"] = (frame.exit - frame.entry).dt.days
    return frame


def verify(trades, universe, signals):
    """Every 8:00 order must be exactly what the rule says on the prior close."""
    problems = 0
    for _, t in trades[trades.submitted.dt.hour != 0].iterrows():
        s = signals[t.ticker].loc[:t.submitted.normalize() - pd.Timedelta(days=1)].iloc[-1]
        ok = (s.close < s.ema) if t.qty > 0 else (s.close >= s["max"])
        if not ok:
            problems += 1
            print("mismatch", t.ticker, t.submitted, dict(s))
    print(f"verified {len(trades)} orders, {problems} rule mismatches")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_universe(universe, book, spy_days):
    names = ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "ADBE", "CRM", "CSCO", "ACN", "MU",
             "AMD", "PLTR", "INTU", "IBM", "NOW", "TXN", "QCOM", "AMAT", "LRCX"]
    days = [d for d in spy_days]
    member = {n: [] for n in names}
    for d in days:
        today = universe.loc[:d - pd.Timedelta(days=1)].iloc[-1]
        for n in names:
            member[n].append(n in today)
    names = [n for n in names if any(member[n])]

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(names) + 1.4))
    for row, n in enumerate(reversed(names)):
        flags = pd.Series(member[n], index=days)
        runs = (flags != flags.shift()).cumsum()
        for _, run in flags[flags].groupby(runs[flags]):
            ax.barh(row, run.index[-1] - run.index[0] + pd.Timedelta(days=1),
                    left=run.index[0], height=0.62, color="#cfd9e6", linewidth=0)
        for _, trade in book[book.ticker == n].iterrows():
            ax.barh(row, trade.exit - trade.entry, left=trade.entry, height=0.32,
                    color=BLUE, linewidth=0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)), color=INK)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
    ax.set_title("誰在「科技股前五大」裡（淺色），策略實際持有的期間（深藍）",
                 loc="left", color=INK, fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#cfd9e6"), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["在股票池內", "持有中"], loc="upper left", bbox_to_anchor=(0, -0.06),
              ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-universe.png", dpi=130)
    plt.close(fig)


def fig_signal(ticker, s, book, start, end, filename, title, labels=True):
    s = s.loc[start:end]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(s.index, s.close, color=INK, lw=1.4, label="收盤價")
    ax.plot(s.index, s.ema, color=BLUE, lw=2, label="100 日 EMA（買進線）")
    ax.plot(s.index, s["max"], color=ORANGE, lw=2, label="252 日最高價（賣出線）")
    below = s.close < s.ema
    ax.fill_between(s.index, s.close, s.ema, where=below, color=BLUE, alpha=0.12,
                    linewidth=0, label="收盤 < EMA：可以買")
    for _, t in book[book.ticker == ticker].iterrows():
        if pd.Timestamp(start) <= t.entry <= pd.Timestamp(end):
            ax.scatter(t.entry, t.entry_px, s=80, color=AQUA, zorder=5, edgecolor=PAPER, lw=2)
            if labels:
                ax.annotate(f"買 {t.entry:%Y-%m-%d}\n{t.entry_px:.2f}", (t.entry, t.entry_px),
                            xytext=(0, -38), textcoords="offset points", ha="center",
                            color=INK, fontsize=9)
        if t.reason != "仍持有" and pd.Timestamp(start) <= t.exit <= pd.Timestamp(end):
            ax.scatter(t.exit, t.exit_px, s=80, color=RED, zorder=5, edgecolor=PAPER, lw=2)
            if labels:
                ax.annotate(f"賣 {t.exit:%Y-%m-%d}\n{t.exit_px:.2f}", (t.exit, t.exit_px),
                            xytext=(0, 18), textcoords="offset points", ha="center",
                            color=INK, fontsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_ylabel("還原權值後價格（USD）")
    ax.scatter([], [], s=80, color=AQUA, label="買進（隔天開盤成交）")
    ax.scatter([], [], s=80, color=RED, label="賣出")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=130)
    plt.close(fig)


def fig_exit_rule(s, day):
    """Zoom on the ten days around a real exit, bars drawn as high-low ranges."""
    window = s.loc[:day].iloc[-8:]
    window = pd.concat([window, s.loc[day:].iloc[1:3]])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(window))
    for i, (d, row) in enumerate(window.iterrows()):
        ax.vlines(i, row.low if "low" in row else row.close, row.high, color=MUTED, lw=2)
        ax.hlines(row.close, i - 0.25, i + 0.25, color=INK, lw=2.5)
    ax.step(x, window["max"], where="mid", color=ORANGE, lw=2, label="252 日最高價（含當天 High）")
    at = list(window.index).index(pd.Timestamp(day))
    row = window.loc[day]
    ax.annotate(f"{day:%m-%d} 收 {row.close:.2f} = 最高 {row.high:.2f}\n"
                f"前一天為止的一年最高只有 {row.prior_max:.2f}\n→ 觸發賣出，隔天開盤成交",
                (at, row.close), xytext=(-250, -30), textcoords="offset points",
                color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.plot([], [], color=MUTED, lw=2, label="當天 最低～最高")
    ax.plot([], [], color=INK, lw=2.5, label="收盤")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%m-%d") for d in window.index])
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("NVDA 2024 年 10 月：賣出條件其實是「收盤 = 當天最高 = 一年最高」",
                 loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig3-exit-rule.png", dpi=130)
    plt.close(fig)


def fig_equity(strategy, spy, ew, exposure):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    for series, color, label in [(strategy, BLUE, "本策略"),
                                 (ew, ORANGE, "一直抱著前五大科技股（等權）"),
                                 (spy, MUTED, "SPY 買進持有")]:
        norm = series / series.iloc[0] * 100_000
        top.plot(norm.index, norm, color=color, lw=2, label=label)
        top.annotate(f"{norm.iloc[-1] / 1000:,.0f}k", (norm.index[-1], norm.iloc[-1]),
                     xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    top.set_ylabel("資產（USD）")
    top.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    top.legend(loc="upper left", frameon=False)
    top.set_title("10 萬美元起跑，五年後的資產", loc="left", color=INK, fontsize=12)
    bottom.fill_between(exposure.index, exposure * 100, color=BLUE, alpha=0.25, linewidth=0)
    bottom.plot(exposure.index, exposure * 100, color=BLUE, lw=1.2)
    bottom.axhline(100, color=RED, lw=1, ls="--")
    bottom.annotate("100%：超過這條線就是借錢買", (exposure.index[0], 100), xytext=(0, 4),
                    textcoords="offset points", color=INK, fontsize=8)
    bottom.set_ylim(0, 125)
    bottom.set_ylabel("持股比例 %")
    bottom.xaxis.set_major_locator(mdates.YearLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig4-equity.png", dpi=130)
    plt.close(fig)


def fig_trades(book):
    colors = {"創一年新高": GREEN, "跌出前五": ORANGE, "仍持有": BLUE}
    book = book.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.27 * len(book) + 1.4))
    x = book.ret * 100
    ax.barh(range(len(book)), x, color=[colors[r] for r in book.reason], height=0.7)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(book)))
    ax.set_yticklabels([f"{t:<5} {a:%Y-%m-%d} → {b:%Y-%m-%d}" for t, a, b in
                        zip(book.ticker, book.entry, book.exit)], fontsize=8, color=INK,
                       family="monospace")
    for i, v in enumerate(x):
        ax.annotate(f"{v:+.0f}%", (max(v, 0), i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xlabel("單筆報酬 %")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, ["創一年新高而賣出", "跌出前五而賣出", "回測結束時仍持有"],
              frameon=False, loc="lower right")
    ax.set_title("全部 32 筆交易（由上而下依買進時間排列）", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig5-trades.png", dpi=130)
    plt.close(fig)


def equal_weight_hold(universe, signals, days):
    """Always hold the day's top five at 1/5 each, rebalanced daily, no fees."""
    closes = pd.DataFrame({t: s.close for t, s in signals.items()}).loc[:END]
    returns = closes.pct_change()
    daily = []
    for d in days[1:]:
        names = universe.loc[:d - pd.Timedelta(days=1)].iloc[-1]
        daily.append(returns.loc[d, names].mean())
    return pd.Series(np.cumprod([1] + [1 + r for r in daily]) * 100_000, index=days)


def main():
    use_cjk_font()
    technology, universe, signals, spy_days = build()
    trades, result = orders()
    verify(trades, universe, signals)
    book = classify(trades, universe, signals)

    strategy = equity(result).loc[START:END]
    spy = spy_equity().loc[START:END]
    ew = equal_weight_hold(universe, signals, spy_days)

    holdings = result["charts"]["Exposure"]["series"]
    long_key = next(k for k in holdings if "Long" in k)
    exposure = pd.Series({pd.Timestamp(v[0], unit="s").normalize(): v[1]
                          for v in holdings[long_key]["values"]}).loc[START:END]

    fig_universe(universe, book, spy_days)
    fig_signal("NVDA", signals["NVDA"], book, "2021-06-01", "2025-03-01",
               "fig2-nvda.png", "NVDA：跌破 EMA 買進，持有 2.75 年，直到某天收在一年最高點")
    exit_day = book[(book.ticker == "NVDA") & (book.reason == "創一年新高")].exit.iloc[0]
    signal_day = signals["NVDA"].loc[:exit_day - pd.Timedelta(days=1)].index[-1]
    fig_exit_rule(signals["NVDA"], signal_day)
    fig_signal("ORCL", signals["ORCL"], book, "2025-12-01", "2026-05-01",
               "fig6-orcl-churn.png", "ORCL 在 2026 年初：排名在第 5、6 名之間來回，被反覆買賣",
               labels=False)
    fig_trades(book)
    fig_equity(strategy, spy, ew, exposure)

    pd.set_option("display.width", 200)
    print(book.to_string())
    print(book.groupby("reason").agg(n=("ret", "size"), mean=("ret", "mean"),
                                     median=("ret", "median"), days=("days", "median")))
    for name, series in [("strategy", strategy), ("spy", spy), ("ew top5", ew)]:
        years = (series.index[-1] - series.index[0]).days / 365.25
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
        dd = (series / series.cummax() - 1).min()
        print(f"{name:9s} end {series.iloc[-1]:>12,.0f}  cagr {cagr:6.2%}  maxdd {dd:6.2%}")
    print("average exposure", round(exposure.mean(), 3))

    # Numbers quoted in the walkthrough.
    nv = signals["NVDA"]
    for d in ["2022-01-18", "2024-10-21"]:
        print("NVDA", d, nv.loc[d].round(3).to_dict())
    new_highs = nv.loc["2023-01-01":"2024-10-21"]
    print("NVDA closes above the prior 252-day high before the exit:",
          int((new_highs.close >= new_highs.prior_max).sum()),
          "of which close == high:", int((new_highs.close >= new_highs["max"]).sum()))
    for t in universe.iloc[[0, -1]]:
        print(t)


if __name__ == "__main__":
    main()
