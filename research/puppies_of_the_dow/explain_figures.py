"""Regenerate the figures and numbers in strategy-explained.md.

Rebuilds every yearly selection -- price, trailing dividend yield, the ten
Dogs and the five Puppies -- from the same data folder LEAN read, checks it
against what LEAN logged, then draws the figures. It also replays the
selection with WBA put back in (see wba_*.csv), since the vendor no longer
serves it and LEAN therefore never saw it.

    uv run run_local.py "Puppies of the Dow" --data-folder data-dow
    uv run run_local.py "Dogs of the Dow" --data-folder data-dow
    uv run run_local.py "DIA Buy & Hold 2021-2026" --data-folder data-dow
    uv run python research/puppies_of_the_dow/explain_figures.py

Needs `data-dow/` (see the docstring of algorithms/puppies_of_the_dow.py),
the raw dividends snapshotted in `data-sp500/.cache/`, and the three
backtests above.
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
DATA = REPO / "data-dow" / "equity" / "usa"
CACHE = REPO / "data-sp500" / ".cache"
OUT = Path(__file__).resolve().parent
RESULTS = REPO / "results"

START, END = pd.Timestamp("2021-09-16"), pd.Timestamp("2026-09-15")
DOGS, PUPPIES = 10, 5
EXAMPLE = pd.Timestamp("2024-01-02")
SELECTIONS = [pd.Timestamp(d) for d in
              ["2021-01-04", "2022-01-03", "2023-01-03", "2024-01-02", "2025-01-02", "2026-01-02"]]
# The first selection is made during warm-up and traded when it ends.
REBALANCES = [START] + SELECTIONS[1:]

FONT_CANDIDATES = ["Microsoft JhengHei", "Noto Sans TC", "Microsoft YaHei",
                   "PingFang TC", "Heiti TC", "Noto Sans CJK TC", "DejaVu Sans"]
INK, MUTED, GRID, PAPER = "#0b0b0b", "#52514e", "#e4e4e1", "#fcfcfb"
# Reference categorical palette, fixed order (dataviz skill, palette.md).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
PALE, REST, NEUTRAL = "#cfd9e6", "#dddcd8", "#f0efec"


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
    """Raw bars plus LEAN's default Adjusted open/close (raw x the factor of
    the first factor-file row dated on or after the bar)."""
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
    factor = scale.iloc[position].to_numpy()
    bars["adj"] = bars["close"] * factor
    bars["adj_open"] = bars["open"] * factor
    return bars


def read_reference(name):
    return pd.read_csv(OUT / name, comment="#")


@lru_cache(maxsize=None)
def actions(ticker):
    """Raw (as-paid) dividends and split ratios by ex-date."""
    if ticker == "WBA":
        frame = read_reference("wba_dividends.csv")
        frame["ratio"] = 0.0
    else:
        frame = pd.read_csv(CACHE / f"{ticker}_actions.csv")
    frame["ex_date"] = pd.to_datetime(frame["ex_date"])
    return frame


@lru_cache(maxsize=None)
def wba_monthly():
    frame = read_reference("wba_monthly.csv")
    return pd.Series(frame["close"].to_numpy(), index=pd.PeriodIndex(frame["month"], freq="M"))


def roster(day):
    """The universe file the 00:00 selection on `day` reads: tickers, and
    whether each one has a weight (i.e. price data)."""
    folder = DATA / "universes" / "etf" / "dia"
    file_date = max(pd.Timestamp(p.stem) for p in folder.glob("*.csv") if pd.Timestamp(p.stem) < day)
    rows = [line.split(",") for line in (folder / f"{file_date:%Y%m%d}.csv").read_text().splitlines()
            if line.strip()]
    return {r[0]: bool(r[3]) for r in rows}


def price_before(ticker, day):
    """Fundamental.price: the previous session's raw close."""
    if ticker == "WBA":
        return float(wba_monthly()[day.to_period("M") - 1])
    bars = read_bars(ticker)
    return float(bars.loc[:day - pd.Timedelta(days=1), "close"].iloc[-1])


def trailing_dividends(ticker, day):
    """Dividends over the year before `day`, in the shares of `day`."""
    a = actions(ticker)
    window = a[(a.ex_date > day - pd.Timedelta(days=365)) & (a.ex_date < day) & (a.dividend > 0)]
    splits = a[(a.ratio > 0) & (a.ex_date < day)]
    total, paid = 0.0, []
    for row in window.itertuples():
        amount = row.dividend
        for split in splits.itertuples():
            if split.ex_date > row.ex_date:
                amount /= split.ratio
        total += amount
        paid.append((row.ex_date, row.dividend, amount))
    return total, paid


def select(day, with_wba=False, dogs_size=DOGS, portfolio=PUPPIES):
    rows = []
    for ticker, priced in roster(day).items():
        if not priced and not (with_wba and ticker == "WBA"):
            continue
        price = price_before(ticker, day)
        dividend, _ = trailing_dividends(ticker, day)
        if price > 5 and dividend > 0:
            rows.append(dict(ticker=ticker, price=price, dividend=dividend, yld=dividend / price))
    frame = pd.DataFrame(rows).set_index("ticker").sort_values("yld", ascending=False)
    dogs = frame.iloc[:dogs_size]
    puppies = dogs.sort_values("price").iloc[:portfolio]
    frame["group"] = ["puppy" if t in puppies.index else "dog" if t in dogs.index else "rest"
                      for t in frame.index]
    return frame, list(dogs.index), list(puppies.index)


def lean_selections(folder="puppies_of_the_dow"):
    log = (RESULTS / folder / "PuppiesOfTheDow-log.txt").read_text(encoding="utf-8")
    out = {}
    for date, body in re.findall(r"^(\d{4}-\d{2}-\d{2}) 00:00:00 SELECT (.+)$", log, re.M):
        rows = [p.split(":") for p in body.split(",")]
        out[pd.Timestamp(date)] = pd.DataFrame(
            [(t, float(p), float(y), g) for t, p, y, g in rows],
            columns=["ticker", "price", "yld", "group"]).set_index("ticker")
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


# --------------------------------------------------------------------------
# holding-period returns
# --------------------------------------------------------------------------

def period_return(ticker, begin, end):
    """Total return bought at `begin`'s open, sold at `end`'s open (or the
    last close for the final period). WBA: month-end closes plus the
    dividends in between -- an approximation, there are no daily bars."""
    if ticker == "WBA":
        p0 = wba_monthly()[begin.to_period("M") - 1]
        p1 = wba_monthly()[end.to_period("M") - 1]
        a = actions("WBA")
        paid = a[(a.ex_date >= begin) & (a.ex_date < end)].dividend.sum()
        return (p1 + paid) / p0 - 1
    bars = read_bars(ticker)
    p0 = bars.loc[begin, "adj_open"]
    p1 = bars.loc[end, "adj"] if end == END else bars.loc[end, "adj_open"]
    return p1 / p0 - 1


def basket_returns(picks):
    """picks: {rebalance day: [tickers]} -> per-period equal-weight return."""
    ends = REBALANCES[1:] + [END]
    out = {}
    for begin, end in zip(REBALANCES, ends):
        names = picks[begin]
        out[begin] = dict(names={t: period_return(t, begin, end) for t in names}, end=end)
        out[begin]["basket"] = np.mean(list(out[begin]["names"].values()))
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_ranking(frame, wba):
    """The example year: every dividend payer's yield, Dogs and Puppies marked."""
    frame = frame.copy()
    frame["ghost"] = False
    ghost = pd.DataFrame([dict(price=wba.price, dividend=wba.dividend, yld=wba.yld,
                               group="ghost", ghost=True)], index=["WBA"])
    full = pd.concat([frame, ghost]).sort_values("yld")
    fig, ax = plt.subplots(figsize=(10, 9))
    color = {"puppy": BLUE, "dog": PALE, "rest": REST}
    for i, (t, row) in enumerate(full.iterrows()):
        if row.ghost:
            ax.barh(i, row.yld * 100, height=0.72, color="none", edgecolor=MUTED,
                    hatch="///", lw=1)
            ax.text(row.yld * 100 + 0.08, i, f"${row.price:.2f}  ← 資料缺漏，LEAN 看不到",
                    va="center", fontsize=8.5, color=MUTED)
        else:
            ax.barh(i, row.yld * 100, height=0.72, color=color[row.group])
            ax.text(row.yld * 100 + 0.08, i, f"${row.price:.2f}", va="center", fontsize=8.5,
                    color=INK if row.group != "rest" else MUTED)
    ax.set_yticks(range(len(full)))
    ax.set_yticklabels(full.index, fontsize=8.5, color=INK)
    real = full[~full.ghost]
    cut = len(real) - DOGS  # bars below this index are not Dogs (WBA ghost sits above)
    boundary = [i for i, (t, row) in enumerate(full.iterrows()) if not row.ghost][cut] - 0.5
    ax.axhline(boundary, color=MUTED, lw=1, ls="--")
    ax.text(full.yld.max() * 100 * 0.98, boundary + 0.25, "↑ 殖利率前 10 名 = Dogs",
            ha="right", va="bottom", fontsize=9, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("過去 12 個月股息 ÷ 前一天收盤價（%）　　（右側數字 = 股價）")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, PALE, REST)]
    ax.legend(handles, ["Puppies：Dogs 裡股價最低的 5 檔（買進）", "其餘 Dogs", "其他成分股"],
              loc="lower right", frameon=False)
    ax.set_title(f"{EXAMPLE:%Y-%m-%d} 選股：道瓊成分股按股息殖利率排序", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-ranking.png", dpi=130)
    plt.close(fig)


def diverging(r, cap=0.4):
    """Blue for gains, red for losses, gray at zero (palette.md diverging pair)."""
    t = min(abs(r) / cap, 1.0)
    pole = np.array(matplotlib.colors.to_rgb(BLUE if r >= 0 else RED))
    mid = np.array(matplotlib.colors.to_rgb(NEUTRAL))
    return tuple(mid + (pole - mid) * (0.15 + 0.85 * t))


def fig_holdings(periods):
    days = list(periods)
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(days) + 1.2))
    for row, d in enumerate(days):
        names = sorted(periods[d]["names"].items(), key=lambda kv: -kv[1])
        for col, (t, r) in enumerate(names):
            ax.add_patch(plt.Rectangle((col, row), 0.94, 0.86, color=diverging(r), lw=0))
            ax.text(col + 0.47, row + 0.43, f"{t}\n{r * 100:+.0f}%", ha="center", va="center",
                    fontsize=9, color=INK)
        b = periods[d]["basket"]
        ax.text(5.15, row + 0.43, f"平均 {b * 100:+.1f}%", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 6.2)
    ax.set_ylim(len(days), 0)
    ax.set_yticks([r + 0.43 for r in range(len(days))])
    ax.set_yticklabels([f"{d:%Y-%m-%d}\n→ {periods[d]['end']:%Y-%m-%d}" for d in days],
                       fontsize=8.5, color=INK)
    ax.set_xticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("每一年買了哪 5 隻小狗、各自的持有期報酬（含股息；藍 = 賺、紅 = 賠）",
                 loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2-holdings.png", dpi=130)
    plt.close(fig)


def fig_equity(series):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
    for (label, s, c), dy in zip(series, [6, -6, 0]):
        top.plot(s.index, s / 1000, color=c, lw=2, label=label)
        top.annotate(f"{s.iloc[-1] / 10000:.1f} 萬", (s.index[-1], s.iloc[-1] / 1000),
                     xytext=(6, dy), textcoords="offset points", va="center", color=INK, fontsize=9)
        dd = s / s.cummax() - 1
        bottom.plot(dd.index, dd * 100, color=c, lw=1.6)
    for d in REBALANCES[1:]:
        top.axvline(d, color=GRID, lw=1, zorder=0)
    top.set_ylabel("資產（千美元）")
    top.legend(loc="upper left", frameon=False)
    top.set_title("10 萬美元的走勢（細直線 = 每年初換股），與從最高點回落的幅度",
                  loc="left", color=INK, fontsize=12)
    bottom.set_ylabel("回撤（%）")
    bottom.xaxis.set_major_locator(mdates.YearLocator())
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig3-equity.png", dpi=130)
    plt.close(fig)


def yearly(s):
    y = s.groupby(s.index.year).last()
    first = pd.Series({s.index[0].year - 1: s.iloc[0]})
    return pd.concat([first, y]).pct_change().dropna() * 100


def fig_yearly(table):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.26
    x = np.arange(len(table))
    for k, (col, c) in enumerate(zip(table.columns, [BLUE, AQUA, ORANGE])):
        ax.bar(x + (k - 1) * width, table[col], width - 0.03, color=c, label=col)
        for xi, v in zip(x, table[col]):
            ax.text(xi + (k - 1) * width, v + (0.8 if v >= 0 else -2.6), f"{round(v)}",
                    ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(min(-12, table.min().min() - 5), table.max().max() + 6)
    labels = [str(y) for y in table.index]
    labels[0] += "\n(9/16 起)"
    labels[-1] += "\n(至 9/15)"
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("當年報酬（%）")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("逐年報酬：小狗、狗、道瓊 ETF", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig4-yearly.png", dpi=130)
    plt.close(fig)


def fig_wba(without, with_wba):
    days = list(without)
    a = np.array([without[d]["basket"] for d in days]) * 100
    b = np.array([with_wba[d]["basket"] for d in days]) * 100
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(days))
    width = 0.36
    ax.bar(x - width / 2, a, width - 0.04, color=BLUE, label="LEAN 實際能選的（沒有 WBA）")
    ax.bar(x + width / 2, b, width - 0.04, color=ORANGE, label="把 WBA 放回去（估算）")
    for xi, v1, v2, d in zip(x, a, b, days):
        for off, v in ((-width / 2, v1), (width / 2, v2)):
            ax.text(xi + off, v + (0.8 if v >= 0 else -2.6), f"{round(v)}", ha="center",
                    fontsize=8, color=INK)
        if "WBA" in with_wba[d]["names"]:
            ax.text(xi + width / 2, min(v2, 0) - 6.5, f"WBA {with_wba[d]['names']['WBA'] * 100:+.0f}%",
                    ha="center", fontsize=8, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(min(a.min(), b.min()) - 10, max(a.max(), b.max()) + 6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:%Y-%m}" for d in days])
    ax.set_ylabel("持有期報酬（%）")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("少了一檔下市股，結果差多少：每個持有期的 5 檔平均報酬", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig5-wba.png", dpi=130)
    plt.close(fig)


def stats(s):
    years = (s.index[-1] - s.index[0]).days / 365.25
    return dict(end=s.iloc[-1], cagr=(s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1,
                dd=(1 - s / s.cummax()).max(), vol=s.pct_change().std() * np.sqrt(252))


def compound(periods, start=100_000):
    value = start
    for p in periods.values():
        value *= 1 + p["basket"]
    return value


def main():
    use_cjk_font()
    lean = lean_selections()

    # 1. reproduce every selection LEAN made
    picks, dog_picks, picks_wba, dog_picks_wba = {}, {}, {}, {}
    worst, mismatches = 0.0, 0
    for sel, reb in zip(SELECTIONS, REBALANCES):
        frame, dogs, puppies = select(sel)
        theirs = lean[sel]
        # LEAN rounds each dividend to the cent, so near-ties can swap order
        # outside the top ten; only the Dogs and the Puppies have to agree.
        if set(puppies) != set(theirs[theirs.group == "P"].index) or set(dogs) != set(theirs.index[:DOGS]):
            mismatches += 1
            print("mismatch", sel.date(), puppies, list(theirs[theirs.group == "P"].index))
        worst = max(worst, (frame.yld - theirs.yld.reindex(frame.index)).abs().max())
        picks[reb], dog_picks[reb] = puppies, dogs
        _, dogs_w, puppies_w = select(sel, with_wba=True)
        picks_wba[reb], dog_picks_wba[reb] = puppies_w, dogs_w
        print(f"{sel.date()}  dogs {dogs}\n            puppies {sorted(puppies)}"
              + (f"\n            with WBA: puppies {sorted(puppies_w)}" if puppies_w != puppies else ""))
    print(f"verified {len(SELECTIONS)} selections, {mismatches} mismatches, "
          f"max |yield - LEAN yield| {worst:.5f}")

    # 2. the worked example
    frame, dogs, puppies = select(EXAMPLE)
    print(f"\n{EXAMPLE.date()} ranking (price, trailing dividend, yield):")
    for t, row in frame.iterrows():
        print(f"  {t:5s} {row.price:8.2f} {row.dividend:6.3f} {row.yld:7.2%}  {row.group}")
    for t in ("VZ", "MMM", "IBM"):
        total, paid = trailing_dividends(t, EXAMPLE)
        print(f"  {t} dividends:", [(d.date().isoformat(), round(a, 4)) for d, _, a in paid],
              f"sum {total:.4f}, price {price_before(t, EXAMPLE):.2f}")
    members = roster(EXAMPLE)
    print(f"  roster {len(members)}, priced {sum(members.values())}, "
          f"dividend payers {len(frame)}, no dividend: "
          f"{sorted(t for t, p in members.items() if p and t not in frame.index)}")
    wba_frame, _, _ = select(EXAMPLE, with_wba=True)
    wba = wba_frame.loc["WBA"]
    print(f"  WBA would be {wba.price:.2f}, div {wba.dividend:.2f}, yield {wba.yld:.2%}, "
          f"rank {list(wba_frame.index).index('WBA') + 1}")
    fig_ranking(frame, wba)

    # 3. nominal price is not value: WMT's split
    for d in (pd.Timestamp("2024-01-02"), pd.Timestamp("2025-01-02")):
        print(f"WMT price before {d.date()}: {price_before('WMT', d):.2f}, "
              f"trailing div {trailing_dividends('WMT', d)[0]:.3f}")

    # 4. what was held and how each name did
    periods = basket_returns(picks)
    fig_holdings(periods)
    for d, p in periods.items():
        print(f"{d.date()} -> {p['end'].date()}  basket {p['basket']:+.2%}  "
              + ", ".join(f"{t} {r:+.1%}" for t, r in sorted(p["names"].items(), key=lambda kv: -kv[1])))
    held = pd.Series([t for p in picks.values() for t in p]).value_counts()
    print("periods held:", held.to_dict())
    changes = [len(set(picks[b]) - set(picks[a])) for a, b in zip(REBALANCES, REBALANCES[1:])]
    print("names replaced per rebalance:", changes)
    overlap = [len(set(picks[d]) & set(dog_picks[d])) for d in REBALANCES]
    print("dogs per year:", {d.date().isoformat(): v for d, v in dog_picks.items()})

    # 5. LEAN's equity curves
    puppy = equity(load_result("puppies_of_the_dow")).loc[START:END]
    dog = equity(load_result("dogs_of_the_dow")).loc[START:END]
    dia = equity(load_result("dia_buy_and_hold_2021_2026")).loc[START:END]
    dia = dia * 100_000 / dia.iloc[0]  # that run starts with its own cash
    for name, s in [("puppies", puppy), ("dogs", dog), ("dia", dia)]:
        st = stats(s)
        print(f"{name:8s} end {st['end']:,.0f}  cagr {st['cagr']:.2%}  maxdd {st['dd']:.2%}  vol {st['vol']:.1%}")
    dd = puppy / puppy.cummax() - 1
    trough = dd.idxmin()
    print("puppies max drawdown trough", trough.date(), "peak", puppy.loc[:trough].idxmax().date())
    fig_equity([("Puppies（5 檔）", puppy, BLUE), ("Dogs（10 檔）", dog, AQUA), ("DIA 買進持有", dia, ORANGE)])
    table = pd.DataFrame({"Puppies": yearly(puppy), "Dogs": yearly(dog), "DIA": yearly(dia)})
    fig_yearly(table)
    print(table.round(1))

    # 6. WBA: what the missing name costs
    dog_periods = basket_returns(dog_picks)
    periods_wba = basket_returns(picks_wba)
    dog_periods_wba = basket_returns(dog_picks_wba)
    print(f"\npandas replay, no WBA:   puppies {compound(periods):,.0f} (LEAN {puppy.iloc[-1]:,.0f}), "
          f"dogs {compound(dog_periods):,.0f} (LEAN {dog.iloc[-1]:,.0f})")
    print(f"pandas replay, with WBA: puppies {compound(periods_wba):,.0f}, "
          f"dogs {compound(dog_periods_wba):,.0f}")
    years = (END - START).days / 365.25
    for name, v in [("puppies no WBA", compound(periods)), ("puppies WBA", compound(periods_wba)),
                    ("dogs no WBA", compound(dog_periods)), ("dogs WBA", compound(dog_periods_wba))]:
        print(f"  {name:15s} cagr {(v / 100_000) ** (1 / years) - 1:.2%}")
    for d in REBALANCES:
        if "WBA" in periods_wba[d]["names"]:
            print(f"  {d.date()} WBA {periods_wba[d]['names']['WBA']:+.1%} replaces "
                  f"{sorted(set(picks[d]) - set(picks_wba[d]))}, basket "
                  f"{periods[d]['basket']:+.1%} -> {periods_wba[d]['basket']:+.1%}")
    fig_wba(periods, periods_wba)


if __name__ == "__main__":
    main()
