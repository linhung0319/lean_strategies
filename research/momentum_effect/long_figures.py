"""Tables and figures for long-history.md, from long_backtest.py's output.

    uv run python research/momentum_effect/long_backtest.py
    uv run python research/momentum_effect/ff_history.py
    uv run python research/momentum_effect/long_figures.py

Every stop rule here is decided at the month-start rebalance from data that
existed before it; thresholds are expanding percentiles of the signal's own
past, never of the whole sample.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import explain_figures as ef
import ff_history as ff

HERE = ff.HERE
LONG = HERE / "long"


def load(tag="long"):
    per = pd.read_csv(LONG / f"periods-{tag}.csv", index_col=0, parse_dates=True)
    picks = pd.read_csv(LONG / f"picks-{tag}.csv", index_col=0, parse_dates=True)
    daily = pd.read_csv(LONG / f"daily-{tag}.csv", index_col=0, parse_dates=True)
    return per, picks, daily


def french_signal(index):
    """French's big-winner excess vol as of the month end before each rebalance."""
    m = ff.load()
    hot = (m.xvol > m.xvol_p80)
    # m is indexed by the month it applies to; a rebalance on day d uses month d.
    return hot.reindex(index.to_period("M").to_timestamp("M")).set_axis(index).fillna(False)


def rules(per, picks):
    p80 = picks.sel_xvol.expanding(24).quantile(0.8).shift(1)
    own_hot = picks.sel_xvol > p80
    bear = picks.spy24 < 0
    # Barroso & Santa-Clara style: shrink toward SPY as the picks get more volatile.
    target = picks.sel_xvol.expanding(24).median().shift(1)
    weight = (target / picks.sel_xvol).clip(upper=1).fillna(1)
    return {
        "原策略": per.strategy_net,
        "SPY": per.spy,
        "50 檔等權": per.ew50,
        "12-2 動能（跳過最近一個月）": per.skip_month - per.turnover * 0.0005,
        "A 過去 24 月 SPY 下跌 → 改持 SPY": per.strategy_net.where(~bear, per.spy),
        "B 入選 5 檔超額波動 > 歷史 80 百分位 → 改持 SPY": per.strategy_net.where(~own_hot, per.spy),
        "B' 用 French 贏家波動度訊號 → 改持 SPY": per.strategy_net.where(~french_signal(per.index), per.spy),
        "V 依入選 5 檔超額波動縮小部位，其餘持 SPY": weight * per.strategy_net + (1 - weight) * per.spy,
    }, own_hot


def stats(r, spy):
    years = (r.index[-1] - r.index[0]).days / 365.25 + 1 / 12
    eq = (1 + r).cumprod()
    rel = eq / (1 + spy).cumprod()
    x = r - spy
    return {"年化報酬": (eq.iloc[-1] ** (1 / years) - 1) * 100,
            "最大回撤(月)": (eq / eq.cummax() - 1).min() * 100,
            "Sharpe(月, rf=0)": r.mean() / r.std() * np.sqrt(12),
            "超額/年": x.mean() * 12 * 100,
            "相對 SPY 最大回撤": (rel / rel.cummax() - 1).min() * 100}


def yearly(per):
    out = per[["strategy_net", "ew50", "spy"]].groupby(per.index.year).apply(lambda g: (1 + g).prod() - 1) * 100
    out["超額"] = out.strategy_net - out.spy
    return out


def fig_long_equity(daily, start):
    d = daily.loc[start:] / daily.loc[start:].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for col, color, name in (("strategy", ef.BLUE, "原策略"), ("ew50", ef.AQUA, "50 檔等權"), ("spy", ef.MUTED, "SPY")):
        ax.plot(d.index, d[col], color=color, lw=1.6 if col == "strategy" else 1.2, label=name)
        ax.annotate(f"{d[col].iloc[-1]:.0f}x", (d.index[-1], d[col].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=ef.INK)
    ax.set_yscale("log")
    ax.axvspan(pd.Timestamp(start), pd.Timestamp("2015-12-31"), color=ef.PALE, alpha=0.4, lw=0)
    ax.text(pd.Timestamp(start) + pd.Timedelta(days=60), ax.get_ylim()[1] * 0.7,
            "陰影：倖存者偏差嚴重的期間（見內文）", fontsize=9, color=ef.MUTED)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
    ax.set_ylabel("資產倍數（對數）")
    ax.set_title(f"{start[:4]}–2026：同一套規則在 S&P 500 上的表現（pandas 重算，5 bp 交易成本）",
                 loc="left", color=ef.INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "fig9-long-equity.png", dpi=130)
    plt.close(fig)


def fig_long_yearly(y):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    colors = [ef.BLUE if v >= 0 else ef.RED for v in y["超額"]]
    ax.bar(y.index, y["超額"], color=colors, width=0.7)
    ax.axhline(0, color=ef.MUTED, lw=1)
    ax.set_ylabel("策略 − SPY（百分點）".replace("−", "-"))
    ax.set_title("逐年超額報酬（策略減 SPY）", loc="left", color=ef.INK, fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig10-long-yearly.png", dpi=130)
    plt.close(fig)


def fig_rules(per, r):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for name, color in (("原策略", ef.BLUE), ("B 入選 5 檔超額波動 > 歷史 80 百分位 → 改持 SPY", ef.ORANGE),
                        ("V 依入選 5 檔超額波動縮小部位，其餘持 SPY", ef.AQUA)):
        rel = (1 + r[name]).cumprod() / (1 + per.spy).cumprod()
        ax.plot(rel.index, rel, color=color, lw=1.6, label=name)
    ax.axhline(1, color=ef.MUTED, lw=1)
    ax.set_yscale("log")
    ax.set_ylabel("相對 SPY（對數，起點 = 1）")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("1997–2026：停用條件放在原策略上，相對 SPY 的累積表現", loc="left", color=ef.INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "fig11-long-rules.png", dpi=130)
    plt.close(fig)


def main():
    ef.use_cjk_font()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    per, picks, daily = load()
    print("每年平均：有價格資料的成分股 / 成分股")
    cov = picks.groupby(picks.index.year).apply(lambda g: (g.priced / g.members).mean() * 100)
    print(cov.round(1).to_string())
    y = yearly(per)
    print("\n逐年（%）\n", y.round(1).to_string())
    r, hot = rules(per, picks)
    for a, b in (("1997", "2026"), ("1997", "2007"), ("2008", "2015"), ("2016", "2026"), ("2021-10", "2026")):
        spy = per.spy.loc[a:b]
        table = pd.DataFrame({k: stats(v.loc[a:b], spy) for k, v in r.items()}).T
        print(f"\n{a}–{b}\n", table.round(2).to_string())
    print("\n規則 B 停用的月份比例：", round(hot.mean() * 100, 1), "%")
    print("規則 B 在 2026 年各月：", hot.loc["2026"].to_dict())
    fig_long_equity(daily, "1997-02-01")
    fig_long_yearly(y)
    fig_rules(per, r)
    eq = daily.strategy
    peak = eq.loc[:"2001"].idxmax()
    back = eq.loc[peak:][eq.loc[peak:] > eq.loc[peak]].index[:1]
    print("2000 高點", peak.date(), "→ 收復", back[0].date() if len(back) else None)
    rel = daily.strategy / daily.spy
    print("相對 SPY：2008 前高", rel.loc[:"2008-12"].idxmax().date(),
          "→ 收復", rel.loc["2009":][rel.loc["2009":] > rel.loc[:"2008-12"].max()].index[:1])


if __name__ == "__main__":
    main()
