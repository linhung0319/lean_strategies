"""Momentum over 1927-2026, from Kenneth French's data library.

The strategy itself can only be re-run as far back as free price data goes,
and that data loses every company that later died (see long-history.md). The
CRSP-based portfolios on French's site do not: they include delisted firms.
This script uses them as a proxy for "hold the large-cap winners":

    D10 = top decile by 12-2 month return, value weighted
    BW  = big (top NYSE size quintile) x top momentum quintile, value weighted
    MKT = the whole market, value weighted

and asks when D10 beat MKT, when it did not, and which signals, known in
advance, would have told you to stand aside.

    uv run python research/momentum_effect/ff_history.py

Downloads into research/momentum_effect/long/ff/ (gitignored) on first run.
"""

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import explain_figures as ef

HERE = Path(__file__).resolve().parent
CACHE = HERE / "long" / "ff"
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
FILES = ["25_Portfolios_ME_Prior_12_2", "10_Portfolios_Prior_12_2", "F-F_Research_Data_Factors",
         "F-F_Momentum_Factor", "F-F_Research_Data_Factors_daily", "F-F_Momentum_Factor_daily",
         "25_Portfolios_ME_Prior_12_2_Daily"]

def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = CACHE / f"{name}.csv"
        if target.exists():
            continue
        with urllib.request.urlopen(URL.format(name), timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            target.write_bytes(archive.read(archive.namelist()[0]))


def section(name, title=None, daily=False):
    """One table out of a French CSV: the first one, or the one after `title`."""
    lines = (CACHE / f"{name}.csv").read_text(encoding="latin-1").splitlines()
    i = next(k for k, l in enumerate(lines) if title in l) + 1 if title else 0
    while not lines[i].startswith(","):
        i += 1
    rows = [lines[i]]
    for line in lines[i + 1:]:
        if not re.match(r"^\s*\d{6,8}\s*,", line):
            break
        rows.append(line)
    frame = pd.read_csv(io.StringIO("\n".join(rows)), index_col=0)
    frame.columns = [c.strip() for c in frame.columns]
    frame.index = pd.to_datetime(frame.index.astype(str).str.strip(), format="%Y%m%d" if daily else "%Y%m")
    if not daily:
        frame.index = frame.index + pd.offsets.MonthEnd(0)
    return frame.replace([-99.99, -999], np.nan) / 100


def load():
    p25 = section("25_Portfolios_ME_Prior_12_2", "Value Weighted Returns -- Monthly")
    p10 = section("10_Portfolios_Prior_12_2", "Value Weight Returns -- Monthly")
    ff = section("F-F_Research_Data_Factors")
    m = pd.DataFrame({"BW": p25["BIG HiPRIOR"], "D10": p10.iloc[:, -1], "MKT": ff["Mkt-RF"] + ff["RF"],
                      "RF": ff["RF"], "UMD": section("F-F_Momentum_Factor").iloc[:, 0]}).dropna()
    fd = section("F-F_Research_Data_Factors_daily", daily=True)
    pd25 = section("25_Portfolios_ME_Prior_12_2_Daily", "Value Weighted Returns -- Daily", daily=True)
    d = pd.DataFrame({"MKT": fd["Mkt-RF"] + fd["RF"], "BW": pd25["BIG HiPRIOR"]})
    m["X"] = m.D10 - m.MKT
    # Signals as of the end of the previous month.
    log_mkt = np.log1p(m.MKT)
    m["bear24"] = (log_mkt.rolling(24).sum() < 0).shift(1)
    xvol = (d.BW - d.MKT).rolling(126).std() * np.sqrt(252)
    m["xvol"] = xvol.resample("ME").last().reindex(m.index).shift(1)
    m["xvol_p80"] = m["xvol"].expanding(60).quantile(0.8)
    return m.loc["1928-01":]


def rel_curve(r, b):
    return (1 + r).cumprod() / (1 + b).cumprod()


def stats(r, b, rf):
    n = len(r)
    eq = (1 + r).cumprod()
    rel = rel_curve(r, b)
    x = r - b
    return {"年化報酬": (eq.iloc[-1] ** (12 / n) - 1) * 100,
            "最大回撤": (eq / eq.cummax() - 1).min() * 100,
            "Sharpe": (r - rf).mean() / r.std() * np.sqrt(12),
            "超額/年": x.mean() * 12 * 100,
            "資訊比率": x.mean() / x.std() * np.sqrt(12),
            "相對最大回撤": (rel / rel.cummax() - 1).min() * 100}


def episodes(rel, depth=-0.15):
    dd = rel / rel.cummax() - 1
    out, start = [], None
    for t, v in dd.items():
        if start is None and v < 0:
            start = t
        if start is not None and v == 0:
            seg = dd.loc[start:t]
            out.append((start, seg.idxmin(), t, seg.min()))
            start = None
    if start is not None:
        seg = dd.loc[start:]
        out.append((start, seg.idxmin(), None, seg.min()))
    return [e for e in out if e[3] < depth]


def fig_history(m):
    rel = rel_curve(m.D10, m.MKT)
    roll = (m.X.rolling(60).mean() * 12 * 100)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    for k, (start, trough, end, depth) in enumerate(episodes(rel), start=1):
        ax.axvspan(start, end or m.index[-1], color=ef.PALE, alpha=0.55, lw=0)
        # Numbered to match the episode table in long-history.md.
        ax.annotate(str(k), (trough, rel.loc[trough]), xytext=(0, -10), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color=ef.INK, fontweight="bold")
    ax.plot(rel.index, rel, color=ef.BLUE, lw=1.6)
    ax.set_yscale("log")
    ax.set_ylabel("贏家組合 ÷ 大盤（對數）")
    ax.set_title("1927–2026：動能贏家（前 10%）相對大盤的累積表現；陰影 = 相對回撤超過 15%（編號見內文）",
                 loc="left", color=ef.INK, fontsize=12)
    ax2.fill_between(roll.index, roll, 0, where=roll >= 0, color=ef.BLUE, alpha=0.8, lw=0)
    ax2.fill_between(roll.index, roll, 0, where=roll < 0, color=ef.RED, alpha=0.8, lw=0)
    ax2.axhline(0, color=ef.MUTED, lw=1)
    ax2.set_ylabel("過去 5 年平均超額報酬（%/年）")
    fig.tight_layout()
    fig.savefig(HERE / "fig6-century.png", dpi=130)
    plt.close(fig)


def fig_states(m):
    normal = m[m.bear24 == False]  # noqa: E712
    bear = m[m.bear24 == True]  # noqa: E712
    q = pd.qcut(m.xvol, 5, labels=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    vals = [normal.X.mean() * 100, bear.X.mean() * 100]
    a.bar(["過去 24 個月\n大盤上漲", "過去 24 個月\n大盤下跌"], vals, color=[ef.BLUE, ef.RED], width=0.55)
    for i, v in enumerate(vals):
        a.text(i, v + 0.02, f"{v:+.2f}%\n（{[len(normal), len(bear)][i]} 個月）", ha="center", va="bottom", color=ef.INK)
    a.set_ylim(0, max(vals) * 1.45)
    a.set_title("依大盤狀態：贏家下個月的平均超額報酬", loc="left", color=ef.INK, fontsize=11)
    a.grid(axis="x", visible=False)
    means = m.groupby(q).X.mean() * 100
    worst = m.groupby(q).X.min() * 100
    colors = [ef.BLUE] * 4 + [ef.RED]
    b.bar(range(5), means, color=colors, width=0.6)
    for i in range(5):
        b.text(i, means.iloc[i] + 0.02, f"{means.iloc[i]:+.2f}%\n最差 {worst.iloc[i]:.0f}%", ha="center",
               va="bottom", fontsize=9, color=ef.INK)
    b.set_xticks(range(5), ["最低", "2", "3", "4", "最高"])
    b.set_xlabel("過去 126 天「贏家 - 大盤」的波動度（五等分）")
    b.set_ylim(0, means.max() * 1.45)
    b.set_title("依贏家自身的波動度：下個月平均超額報酬", loc="left", color=ef.INK, fontsize=11)
    b.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig7-states.png", dpi=130)
    plt.close(fig)


def rules(m):
    hot = m.xvol > m.xvol_p80
    return {
        "一直持有贏家": m.D10,
        "大盤": m.MKT,
        "A 過去 24 月大盤下跌 → 改持大盤": m.D10.where(m.bear24 == False, m.MKT),  # noqa: E712
        "B 贏家波動度 > 歷史 80 百分位 → 改持大盤": m.D10.where(~hot, m.MKT),
        "C 過去 24 月大盤下跌 → 改持現金": m.D10.where(m.bear24 == False, m.RF),  # noqa: E712
    }


def fig_rules(m, start="2000"):
    r = rules(m)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for (name, color) in (("一直持有贏家", ef.BLUE), ("B 贏家波動度 > 歷史 80 百分位 → 改持大盤", ef.ORANGE)):
        c = rel_curve(r[name].loc[start:], m.MKT.loc[start:])
        ax.plot(c.index, c, color=color, lw=1.8, label=name)
    ax.axhline(1, color=ef.MUTED, lw=1)
    ax.set_ylabel("相對大盤（起點 = 1）")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"{start}–2026：停用條件 B 對相對表現的影響", loc="left", color=ef.INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "fig8-rule.png", dpi=130)
    plt.close(fig)


def main():
    ef.use_cjk_font()
    fetch()
    m = load()
    pd.set_option("display.width", 200)
    y = m.groupby(m.index.year)[["D10", "MKT"]].apply(lambda g: (1 + g).prod() - 1)
    y["超額"] = y.D10 - y.MKT
    print("逐年（%）\n", (y * 100).round(1).to_string())
    dec = m.groupby(m.index.year // 10 * 10).apply(
        lambda g: pd.Series({"超額/年": g.X.mean() * 12 * 100, "勝率": (g.X > 0).mean() * 100}))
    print("\n逐十年\n", dec.round(1).to_string())
    print("\n相對回撤 > 15%")
    for s, t, e, d in episodes(rel_curve(m.D10, m.MKT)):
        print(f"  {s:%Y-%m} → 谷底 {t:%Y-%m} → 回復 {e:%Y-%m}" if e else f"  {s:%Y-%m} → 谷底 {t:%Y-%m} → 未回復",
              f"{d*100:.1f}%")
    for label, frame in (("bear24", m.groupby("bear24").X),):
        print("\n", label, (frame.agg(["mean", "count"]) * [100, 1]).round(2).to_string())
    for per in (("1928", "1975"), ("1976", "2026"), ("2000", "2026")):
        sl = m.loc[per[0]:per[1]]
        table = pd.DataFrame({k: stats(v.loc[per[0]:per[1]], sl.MKT, sl.RF) for k, v in rules(m).items()}).T
        print(f"\n{per[0]}–{per[1]}\n", table.round(2).to_string())
    fig_history(m)
    fig_states(m)
    fig_rules(m)


if __name__ == "__main__":
    main()
