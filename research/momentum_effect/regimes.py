"""When does momentum work, and when should it be switched off?

Splits the 1927-2026 momentum record (ff_history.py's D10 vs MKT) into
alternating good and bad legs, reads the market state at every turn, and
turns what the turns have in common into a three-flag scorecard:

    xvol  volatility of big winners minus market, last 126 days
    ext   how far the winners ran ahead of the market, last 12 months
    disp  cross-sectional spread of 12-month returns across 49 industries

each flagged when above the 80th percentile of its own past. The scorecard is
then applied, unchanged, to the strategy's 1997-2026 re-simulation
(long_backtest.py), with 0-2 months of publication lag.

    uv run python research/momentum_effect/regimes.py [--sp500]

--sp500 also rebuilds the three signals from data-sp500-long/ (slow, ~2 min)
to see how much of the market-wide signal our own data reproduces.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import explain_figures as ef
import ff_history as ff
import long_figures as lf

HERE = ff.HERE
SIGNALS = ["xvol", "ext", "disp"]
LABELS = {"xvol": "贏家波動度", "ext": "贏家領先幅度", "disp": "產業分化程度", "mktvol": "大盤波動度", "mkt12": "大盤 12 月報酬"}


def state():
    """Market state at the END of each month (known before the next one)."""
    m = ff.load()
    ind = ff.section("49_Industry_Portfolios", "Average Value Weighted Returns -- Monthly")
    ind = ind.where(ind > -0.99)
    fd = ff.section("F-F_Research_Data_Factors_daily", daily=True)
    log_mkt = np.log1p(m.MKT)
    s = pd.DataFrame(index=m.index)
    daily = ff.section("25_Portfolios_ME_Prior_12_2_Daily", "Value Weighted Returns -- Daily", daily=True)
    s["xvol"] = ((daily["BIG HiPRIOR"] - fd["Mkt-RF"] - fd["RF"]).rolling(126).std()
                 * np.sqrt(252)).resample("ME").last().reindex(m.index)
    s["ext"] = np.log1p(m.D10).rolling(12).sum() - log_mkt.rolling(12).sum()
    s["disp"] = np.log1p(ind).rolling(12).sum().std(axis=1)
    s["mktvol"] = ((fd["Mkt-RF"] + fd["RF"]).rolling(126).std() * np.sqrt(252)).resample("ME").last().reindex(m.index)
    s["mkt12"] = log_mkt.rolling(12).sum()
    pct = s.apply(lambda c: c.expanding(60).rank(pct=True))
    flags = sum((pct[c] >= 0.8).astype(int) for c in SIGNALS)
    return m, s, pct, flags


def zigzag(series, threshold=0.15):
    """Turning points of `series` where it reverses by more than `threshold`."""
    points = [series.index[0]]
    direction, ext_t, ext_v = 0, series.index[0], series.iloc[0]
    for t, v in series.items():
        if direction >= 0 and v > ext_v:
            ext_t, ext_v = t, v
        elif direction >= 0 and v < ext_v * (1 - threshold):
            points.append(ext_t)
            direction, ext_t, ext_v = -1, t, v
        elif direction <= 0 and v < ext_v:
            ext_t, ext_v = t, v
        elif direction <= 0 and v > ext_v * (1 + threshold):
            points.append(ext_t)
            direction, ext_t, ext_v = 1, t, v
    points.append(ext_t)
    return points


def legs(m, pct):
    rel = ff.rel_curve(m.D10, m.MKT)
    pts = zigzag(rel)
    rows = []
    for a, b in zip(pts[:-1], pts[1:]):
        row = dict(start=f"{a:%Y-%m}", end=f"{b:%Y-%m}", months=(b.year - a.year) * 12 + b.month - a.month,
                   change=(rel[b] / rel[a] - 1) * 100)
        row.update({f"{c}@end": pct.loc[b, c] * 100 for c in SIGNALS + ["mktvol", "mkt12"]})
        rows.append(row)
    return pd.DataFrame(rows)


def forward_by_flags(m, flags, lag=0):
    x = np.log1p(m.D10) - np.log1p(m.MKT)
    d = pd.DataFrame({"n": flags.shift(lag), "next1": x.shift(-1), "next12": x.rolling(12).sum().shift(-12)})
    d = d.loc["1934":].dropna()
    return d.groupby("n").agg(月數=("next1", "size"),
                              下月超額年化=("next1", lambda s: s.mean() * 1200),
                              未來12月超額=("next12", lambda s: s.mean() * 100),
                              未來12月勝率=("next12", lambda s: (s > 0).mean() * 100))


def on_strategy(flags, lag=0):
    per, _, _ = lf.load()
    month = (per.index.to_period("M") - 1 - lag).to_timestamp("M")
    n = flags.reindex(month).set_axis(per.index)
    x = per.strategy_net - per.spy
    x12 = np.log1p(per.strategy_net).rolling(12).sum().shift(-11) - np.log1p(per.spy).rolling(12).sum().shift(-11)
    table = pd.DataFrame({"n": n, "x": x, "x12": x12}).groupby("n").agg(
        月數=("x", "size"), 下月超額年化=("x", lambda s: s.mean() * 1200),
        未來12月超額=("x12", lambda s: s.mean() * 100), 未來12月勝率=("x12", lambda s: (s > 0).mean() * 100))
    rules = {"原策略": per.strategy_net,
             "3 旗 → 改持 SPY": per.strategy_net.where(n < 3, per.spy),
             "依旗數縮部位（每旗 -1/3）": (1 - n / 3) * per.strategy_net + n / 3 * per.spy}
    perf = {(a, b): pd.DataFrame({k: lf.stats(v.loc[a:b], per.spy.loc[a:b]) for k, v in rules.items()}).T
            for a, b in (("1997", "2026"), ("2000", "2019"), ("2016", "2026"))}
    return table, perf, n


def sp500_signals():
    """The same three signals from our own S&P 500 data, monthly."""
    import long_backtest as lb
    data = lb.Data("data-sp500-long")
    days = data.bars("SPY").index
    spy = data.bars("SPY")["adj"]
    ends = pd.Series(days, index=days).groupby(days.to_period("M")).last()
    tickers = sorted({p.stem.upper() for p in (data.root / "daily").glob("*.zip")} - {"SPY"})
    adj = pd.DataFrame({t: data.bars(t)["adj"] for t in tickers}).reindex(days)
    ret = adj.pct_change(fill_method=None)
    rows = {}
    for i, t in enumerate(ends[:-1]):
        k = days.get_loc(t)
        if k < 260:
            continue
        weights, _ = data.constituents(data.files[data.files <= t][-1])
        members = [x for x in weights if x in adj.columns]
        mom = (adj.iloc[k - 21][members] / adj.iloc[k - 252][members] - 1).dropna()
        top = list(mom.sort_values().index[-max(10, len(mom) // 10):])
        window = ret.iloc[k - 125:k + 1]
        nxt = days.get_loc(ends.iloc[i + 1])
        rows[t] = dict(xvol=(window[top].mean(axis=1) - window[members].mean(axis=1)).std() * np.sqrt(252),
                       disp=np.log(adj.iloc[k][members] / adj.iloc[k - 252][members]).std(),
                       fwd=np.log((adj.iloc[nxt][top] / adj.iloc[k][top]).mean()) - np.log(spy.iloc[nxt] / spy.iloc[k]))
    s = pd.DataFrame(rows).T.astype(float)
    s.index = s.index.to_period("M").to_timestamp("M")
    s["ext"] = s["fwd"].shift(1).rolling(12).sum()
    return s[SIGNALS]


def fig_flags(m, flags):
    rel = ff.rel_curve(m.D10, m.MKT).loc["1934":]
    n = flags.loc["1934":]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot(rel.index, rel, color=ef.BLUE, lw=1.4)
    ax.set_yscale("log")
    for t in n[n == 3].index:
        ax.axvspan(t - pd.offsets.MonthBegin(1), t, color=ef.RED, alpha=0.35, lw=0)
    ax.set_ylabel("贏家 ÷ 大盤（對數）")
    ax.set_title("1934–2026：紅色 = 三個警訊同時亮起的月份（贏家波動度、領先幅度、產業分化都在歷史前 20%）",
                 loc="left", color=ef.INK, fontsize=11.5)
    colors = {0: ef.PALE, 1: ef.YELLOW, 2: ef.ORANGE, 3: ef.RED}
    ax2.bar(n.index, n, width=31, color=[colors[int(v)] for v in n])
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_ylabel("亮起的警訊數")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig12-flags.png", dpi=130)
    plt.close(fig)


def fig_turns(lg):
    up = lg[lg.change > 0]
    down = lg[lg.change < 0]
    cols = SIGNALS + ["mktvol", "mkt12"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    xs = np.arange(len(cols))
    for offset, frame, color, name in ((-0.18, up, ef.RED, "好時期結束時（= 壞時期開始）"),
                                       (0.18, down, ef.BLUE, "壞時期結束時（= 好時期開始）")):
        vals = frame[[f"{c}@end" for c in cols]]
        ax.bar(xs + offset, vals.median(), width=0.34, color=color, label=name)
        for i, c in enumerate(cols):
            ax.scatter(np.full(len(vals), xs[i] + offset), vals[f"{c}@end"], s=10, color=ef.INK, alpha=0.45, zorder=3)
    ax.axhline(50, color=ef.MUTED, lw=1, ls="--")
    ax.set_xticks(xs, [LABELS[c] for c in cols])
    ax.set_ylabel("在自身歷史中的百分位")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_title("轉折點當下的市場狀態：長條 = 中位數，點 = 每一次轉折", loc="left", color=ef.INK, fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig13-turns.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sp500", action="store_true")
    args = parser.parse_args()
    ef.use_cjk_font()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    m, s, pct, flags = state()

    lg = legs(m, pct)
    print("好壞時期（贏家相對大盤反轉超過 15%）與轉折點的狀態百分位\n", lg.round(0).to_string())
    up, down = lg[lg.change > 0], lg[lg.change < 0]
    print("\n好時期結束時，中位數百分位\n", up.filter(like="@end").median().round(0).to_string())
    print("\n壞時期結束時，中位數百分位\n", down.filter(like="@end").median().round(0).to_string())

    print("\n未來 12 個月超額報酬（%），依狀態五等分（1934–1975 / 1976–2026）")
    x12 = (np.log1p(m.D10) - np.log1p(m.MKT)).rolling(12).sum().shift(-12)
    for c in SIGNALS + ["mktvol", "mkt12"]:
        parts = []
        for a, b in (("1934", "1975"), ("1976", "2026")):
            d = pd.DataFrame({"s": s[c], "f": x12}).loc[a:b].dropna()
            parts.append((d.groupby(pd.qcut(d.s.rank(method="first"), 5, labels=False)).f.mean() * 100).round(1).tolist())
        print(f"  {LABELS[c]:<8} {parts[0]}  {parts[1]}")

    for lag in (0, 1, 2):
        print(f"\n依亮燈數（延遲 {lag} 個月）：French 贏家組合\n", forward_by_flags(m, flags, lag).round(1).to_string())
        table, perf, n = on_strategy(flags, lag)
        print(f"依亮燈數（延遲 {lag} 個月）：本策略相對 SPY\n", table.round(1).to_string())
        for (a, b), t in perf.items():
            print(f"  {a}–{b}\n", t.round(2).to_string())
    print("\n三燈月份：", ", ".join(f"{t:%Y-%m}" for t in flags[flags == 3].index))

    if args.sp500:
        mine = sp500_signals()
        both = mine.join(s[SIGNALS], rsuffix="_ff", how="inner")
        print("\n用 S&P 500 資料自己算的訊號，和 French 版本的相關係數：",
              {c: round(both[c].corr(both[c + "_ff"]), 2) for c in SIGNALS})
        mine_flags = sum((mine[c].expanding(24).rank(pct=True) >= 0.8).astype(int) for c in SIGNALS)
        table, perf, _ = on_strategy(mine_flags)
        print("自算訊號 → 本策略\n", table.round(1).to_string())
        print("自算三燈月份：", ", ".join(f"{t:%Y-%m}" for t in mine_flags[mine_flags == 3].index))

    fig_flags(m, flags)
    fig_turns(lg)


if __name__ == "__main__":
    main()
