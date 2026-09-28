"""Compare the same rules run over three different universes.

    uv run run_local.py "Tech Top5 EMA Reversion" --data-folder data-sp500
    uv run run_local.py "SP500 Top5 EMA Reversion" --data-folder data-sp500
    uv run run_local.py "SP500 Top5 EMA Reversion (one Alphabet)" --data-folder data-sp500
    uv run run_local.py "SPY Buy & Hold 2021-2026" --data-folder data-sp500
    uv run python research/tech_ema_reversion/compare_universes.py

Writes fig7-universe-comparison.png and fig8-sp500-universe.png, and prints
the tables quoted in strategy-explained.md's comparison section.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from explain_figures import (BLUE, END, GREEN, INK, MUTED, ORANGE, OUT, PAPER, SIZE, START,
                             VIOLET, classify, equity, load_technology, orders, read_bars,
                             spy_equity, use_cjk_font, universe_by_day, build)

RUNS = [
    ("科技股前五大", "tech_top5_ema_reversion", BLUE),
    ("S&P 500 前五大（合併 Alphabet）", "sp500_top5_ema_reversion_one_alphabet", ORANGE),
    ("S&P 500 前五大（原樣，GOOGL+GOOG 各佔一格）", "sp500_top5_ema_reversion", VIOLET),
]


def pnl_by_ticker(trades, signals):
    """Realised + unrealised profit per ticker, in dollars."""
    trades = trades.sort_values("submitted")
    open_positions, rows = {}, []
    for _, t in trades.iterrows():
        if t.qty > 0:
            open_positions[t.ticker] = t
        else:
            entry = open_positions.pop(t.ticker)
            rows.append((t.ticker, entry.qty * (t.price - entry.price)))
    for ticker, entry in open_positions.items():
        last = signals[ticker]["close"].loc[:END].iloc[-1]
        rows.append((ticker, entry.qty * (last - entry.price)))
    return pd.DataFrame(rows, columns=["ticker", "pnl"]).groupby("ticker").pnl.sum()


def summarise(label, slug, universe, signals):
    trades, result = orders(slug)
    book = classify(trades, universe, signals)
    series = equity(result).loc[START:END]
    years = (series.index[-1] - series.index[0]).days / 365.25
    stats = result["statistics"]
    return dict(
        label=label, equity=series, book=book,
        end=series.iloc[-1],
        cagr=(series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1,
        maxdd=(series / series.cummax() - 1).min(),
        sharpe=float(stats["Sharpe Ratio"]),
        orders=int(stats["Total Orders"]),
        pnl=pnl_by_ticker(trades, signals),
    )


def fig_comparison(runs, spy):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for run, colour in zip(runs, [c for _, _, c in RUNS]):
        norm = run["equity"] / run["equity"].iloc[0] * 100_000
        ax.plot(norm.index, norm, color=colour, lw=2, label=run["label"])
        ax.annotate(f"{norm.iloc[-1] / 1000:,.0f}k", (norm.index[-1], norm.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    norm = spy / spy.iloc[0] * 100_000
    ax.plot(norm.index, norm, color=MUTED, lw=2, label="SPY 買進持有")
    # This one all but lands on the raw index run, so it gets its own line.
    ax.annotate(f"{norm.iloc[-1] / 1000:,.0f}k", (norm.index[-1], norm.iloc[-1]),
                xytext=(6, -12), textcoords="offset points", va="center", color=INK)
    ax.set_ylabel("資產（USD）")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("同一套規則，換不同的選股池（10 萬美元起跑）", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig7-universe-comparison.png", dpi=130)
    plt.close(fig)


def fig_membership(universe, book, days, filename, title, names):
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(names) + 1.4))
    member = {n: [] for n in names}
    for d in days:
        today = universe.loc[:d - pd.Timedelta(days=1)].iloc[-1]
        for n in names:
            member[n].append(n in today)
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
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#cfd9e6"), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["在股票池內", "持有中"], loc="upper left", bbox_to_anchor=(0, -0.06),
              ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=130)
    plt.close(fig)


def main():
    use_cjk_font()
    technology, tech_universe, signals, days = build()
    index_universe = universe_by_day(exclude={"GOOG"})
    raw_universe = universe_by_day()

    # Bars for every name any of the three universes can hold.
    for names in list(index_universe) + list(raw_universe):
        for ticker in names:
            if ticker not in signals:
                bars = read_bars(ticker)
                signals[ticker] = pd.DataFrame({"close": bars["close"], "high": bars["high"]})

    universes = [tech_universe, index_universe, raw_universe]
    runs = [summarise(label, slug, universe, signals)
            for (label, slug, _), universe in zip(RUNS, universes)]
    spy = spy_equity().loc[START:END]

    fig_comparison(runs, spy)
    inside = index_universe.loc[START:END]
    counts = pd.Series([t for names in inside for t in names]).value_counts()
    names = list(counts.index)
    fig_membership(index_universe, runs[1]["book"], days, "fig8-sp500-universe.png",
                   f"S&P 500 前五大（合併 Alphabet）：五年來只有 {len(names)} 檔進過股票池",
                   names)

    print(f"{'universe':46s} {'end':>10s} {'cagr':>7s} {'maxdd':>7s} {'sharpe':>7s} {'orders':>7s}")
    for run in runs:
        print(f"{run['label']:46s} {run['end']:10,.0f} {run['cagr']:6.2%} "
              f"{run['maxdd']:6.2%} {run['sharpe']:7.3f} {run['orders']:7d}")
    print(f"{'SPY buy & hold':46s} {spy.iloc[-1] / spy.iloc[0] * 100_000:10,.0f}")

    for run in runs:
        print(f"\n== {run['label']}")
        print(run["book"].groupby("reason").agg(n=("ret", "size"), mean=("ret", "mean"),
                                                median=("ret", "median"), days=("days", "median")))
        print(run["pnl"].sort_values(ascending=False).round(0).to_string())

    # When each universe first lets the two big winners in, and at what price.
    for (name, universe), run in zip([("tech", tech_universe), ("index", index_universe)], runs):
        inside = universe.loc[START:END]
        for ticker in ["NVDA", "AVGO", "TSLA"]:
            days_in = [d for d, names in inside.items() if ticker in names]
            first = f"{min(days_in):%Y-%m-%d}" if days_in else "never"
            bought = run["book"][run["book"].ticker == ticker]
            price = f"{bought.entry_px.iloc[0]:.2f} on {bought.entry.iloc[0]:%Y-%m-%d}" \
                if len(bought) else "never bought"
            print(f"{name:6s} {ticker:5s} enters universe {first:>10s}  "
                  f"{len(days_in) / len(inside):4.0%} of days  first buy {price}")


if __name__ == "__main__":
    main()
