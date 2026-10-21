"""Regenerate the figures and numbers in pca-in-2d.md.

The same PCA -> OLS -> z-score the strategy runs, shrunk to two stocks (MA and
V, the 60 sessions before the 2024-11-01 rebalance) and one component, so every
step can be drawn on a plane.

    uv run --group pca python research/pca_stat_arbitrage/pca_2d_figures.py

Needs `data-sp500/` only; no backtest result is read.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

from explain_figures import (EXAMPLE, ENTRY, OUT, INK, MUTED, GRID, PAPER, BLUE, ORANGE,
                             AQUA, RED, PALE, VIOLET, closes, use_cjk_font)

X, Y = "MA", "V"
HIGHLIGHT = pd.Timestamp("2024-10-31")


def model():
    """The strategy's _compute_weights_by_symbol for two stocks and one factor,
    in percent so the axes read as 'percent above its own 60-day average'."""
    sample = np.log(closes([X, Y], EXAMPLE))
    sample = (sample - sample.mean()) * 100
    pca = PCA().fit(sample)
    components = pca.components_.copy()
    # Signs from PCA are arbitrary: make PC1 point "both up" and PC2 "V up, MA down".
    if components[0].sum() < 0:
        components[0] *= -1
    if components[1][1] < 0:
        components[1] *= -1
    scores = sample.to_numpy() @ components.T
    design = sm.add_constant(scores[:, :1])
    fits = {t: sm.OLS(sample[t], design).fit() for t in sample.columns}
    resid = pd.DataFrame({t: f.resid for t, f in fits.items()})
    z = (resid - resid.mean()) / resid.std()
    cov = np.cov(sample.T)
    return dict(sample=sample, components=components, var=pca.explained_variance_,
                scores=pd.DataFrame(scores, index=sample.index, columns=["PC1", "PC2"]),
                fits=fits, resid=resid, z=z, cov=cov)


def directional_variance(sample, degrees):
    u = np.array([np.cos(np.radians(degrees)), np.sin(np.radians(degrees))])
    return (sample.to_numpy() @ u).var(ddof=1)


def square(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(f"{X}：比自己 60 天平均高幾 %")
    ax.set_ylabel(f"{Y}：比自己 60 天平均高幾 %")


def line(ax, degrees, lim, **kw):
    t = np.radians(degrees)
    r = lim * 1.5
    ax.plot([-r * np.cos(t), r * np.cos(t)], [-r * np.sin(t), r * np.sin(t)], **kw)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_points(m, lim):
    s = m["sample"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
    left.plot(s.index, s[X], color=BLUE, lw=2, label=X)
    left.plot(s.index, s[Y], color=ORANGE, lw=2, label=Y)
    left.axhline(0, color=MUTED, lw=0.8)
    # Labels sit in empty space and point at their day.
    for day, at in [(s.index[0], (pd.Timestamp("2024-08-19"), -6.4)),
                    (HIGHLIGHT, (pd.Timestamp("2024-10-01"), 5.3))]:
        left.axvline(day, color=MUTED, lw=0.8, ls=":")
        left.annotate(f"{day:%m-%d}\n{X} {s.loc[day, X]:+.2f}%\n{Y} {s.loc[day, Y]:+.2f}%",
                      (day, s.loc[day].mean()), xytext=at, textcoords="data", fontsize=9, color=INK,
                      va="center", arrowprops=dict(arrowstyle="->", color=MUTED))
    left.legend(frameon=False, loc="upper left")
    left.set_ylabel("比自己 60 天平均高幾 %")
    left.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    left.set_title("兩檔股票的 60 天走勢：大部分時間一起動", loc="left", color=INK, fontsize=11)

    right.scatter(s[X], s[Y], s=26, color=VIOLET, alpha=0.55, lw=0)
    for day in (s.index[0], HIGHLIGHT):
        p = s.loc[day]
        right.scatter(p[X], p[Y], s=90, color=RED, edgecolor=PAPER, lw=2, zorder=5)
        right.annotate(f"{day:%m-%d}\n({p[X]:+.2f}, {p[Y]:+.2f})", (p[X], p[Y]),
                       xytext=(10, -4), textcoords="offset points", fontsize=9, color=INK)
    square(right, lim)
    right.set_title("同樣的資料：每一天 = 平面上的一個點", loc="left", color=INK, fontsize=11)
    fig.suptitle("① 2 檔股票 = 2 個維度：x 軸是 MA，y 軸是 V，60 天就是 60 個點",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "pca2d-1-points.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_directions(m, lim, best):
    s = m["sample"]
    tries = [0, 30, round(best), 60, 90]
    colors = {0: BLUE, 30: PALE, round(best): RED, 60: PALE, 90: ORANGE}
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.35]})
    left.scatter(s[X], s[Y], s=22, color=VIOLET, alpha=0.45, lw=0)
    for d in tries:
        line(left, d, lim, color=colors[d], lw=3 if d == round(best) else 1.6,
             ls="-" if d in (0, 90, round(best)) else "--", zorder=4 if d == round(best) else 3)
        t = np.radians(d)
        r = lim * 0.82
        left.text(r * np.cos(t), r * np.sin(t), f"{d}°", color=INK, fontsize=10,
                  ha="left", va="bottom", fontweight="bold" if d == round(best) else "normal")
    square(left, lim)
    left.set_title("試很多方向（每條線都穿過原點）", loc="left", color=INK, fontsize=11)

    angles = np.arange(0, 181)
    v = np.array([directional_variance(s, a) for a in angles])
    right.plot(angles, v, color=INK, lw=2)
    for d in tries + [round(best) + 90]:
        val = directional_variance(s, d)
        c = colors.get(d, AQUA)
        right.scatter(d, val, s=70, color=c, edgecolor=PAPER, lw=2, zorder=5)
        label = {0: f"0°＝只看 {X}\n{val:.2f}", 90: f"90°＝只看 {Y}\n{val:.2f}",
                 round(best): f"{best:.0f}°：最大 {val:.2f}\n＝ PC1",
                 round(best) + 90: f"{best + 90:.0f}°：最小 {val:.2f}\n＝ PC2（和 PC1 垂直）"}.get(d, f"{val:.2f}")
        offset = {0: (10, -4), 30: (-34, 8), round(best): (10, 8), 60: (10, 4),
                  90: (10, 0), round(best) + 90: (12, -4)}[d]
        right.annotate(label, (d, val), xytext=offset, textcoords="offset points", fontsize=9,
                       color=INK, va="center" if d in (0, 90, round(best) + 90) else "bottom")
    right.set_xticks(range(0, 181, 30))
    right.set_xlabel("方向（角度）")
    right.set_ylabel("投影到這個方向後，60 個數字的變異數（%²）")
    right.set_ylim(0, v.max() * 1.18)
    right.set_title("每個方向上「散得多開」：42° 最開，132° 最擠", loc="left", color=INK, fontsize=11)
    fig.suptitle("② PCA 在找的東西：讓 60 個點散得最開的方向", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "pca2d-2-directions.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return v


def fig_shadows(m, lim, best):
    s = m["sample"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (k, deg, c, name) in zip(axes, [(0, best, RED, "PC1"), (1, best + 90, AQUA, "PC2")]):
        u = m["components"][k]
        proj = (s @ u)[:, None] * u
        ax.scatter(s[:, 0], s[:, 1], s=22, color=VIOLET, alpha=0.45, lw=0, zorder=3)
        for p, q in zip(s, proj):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=GRID, lw=0.9, zorder=2)
        line(ax, deg, lim, color=c, lw=2.4, zorder=4)
        ax.scatter(proj[:, 0], proj[:, 1], s=26, color=c, edgecolor=PAPER, lw=0.8, zorder=5)
        var = (s @ u).var(ddof=1)
        ax.text(-lim * 0.95 if k == 0 else lim * 0.05, lim * 0.9,
                f"{name} 方向 = ({u[0]:+.3f}, {u[1]:+.3f})\n影子的變異數 = {var:.2f}",
                color=INK, fontsize=10, va="top")
        square(ax, lim)
        ax.set_title(f"投影到 {name}：影子{'散很開' if k == 0 else '擠在一起'}", loc="left", color=INK, fontsize=11)
    fig.suptitle("③ 「投影」＝每個點往那條線畫垂直線，落下的位置（影子）", x=0.01, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "pca2d-3-shadows.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_ellipse(m, lim):
    s = m["sample"]
    cov = m["cov"]
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1.1]})
    left.scatter(s[X], s[Y], s=22, color=VIOLET, alpha=0.45, lw=0)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for k in (1, 2):
        left.add_patch(Ellipse((0, 0), 2 * k * np.sqrt(vals[0]), 2 * k * np.sqrt(vals[1]), angle=angle,
                               fill=False, color=MUTED, lw=1.2, ls="-" if k == 1 else ":"))
    for i, (c, name) in enumerate([(RED, "PC1"), (AQUA, "PC2")]):
        vec = m["components"][i] * 2 * np.sqrt(vals[i])
        left.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.6))
        left.text(vec[0] * 1.08, vec[1] * 1.08, f"{name}\n特徵值 {vals[i]:.2f}", color=INK, fontsize=9,
                  ha="left" if vec[0] > 0 else "right")
    square(left, lim)
    left.set_title("點雲像一顆橢圓：PCA 的方向就是橢圓的長軸、短軸", loc="left", color=INK, fontsize=11)

    total = np.trace(cov)
    rows = [("原本的座標軸", [(f"{X} 的變異數 {cov[0, 0]:.2f}", cov[0, 0], BLUE),
                            (f"{Y} 的變異數 {cov[1, 1]:.2f}", cov[1, 1], ORANGE)]),
            ("PCA 的座標軸", [(f"PC1 {vals[0]:.2f}（{vals[0] / total:.0%}）", vals[0], RED),
                             (f"PC2 {vals[1]:.2f}（{vals[1] / total:.0%}）", vals[1], AQUA)])]
    for y, (name, parts) in enumerate(rows):
        start = 0
        for label, width, c in parts:
            right.barh(y, width - 0.08, left=start, color=c, height=0.5)
            right.text(start + width / 2, y + 0.38, label, ha="center", va="bottom", fontsize=9, color=INK)
            start += width
        right.text(start + 0.3, y, f"合計 {start:.2f}", va="center", fontsize=10, color=INK)
    right.set_yticks([0, 1])
    right.set_yticklabels([r[0] for r in rows], color=INK)
    right.set_ylim(-0.6, 2.7)
    right.set_xlim(0, total * 1.22)
    right.invert_yaxis()
    right.grid(axis="y", visible=False)
    right.set_xlabel("變異數（%²）")
    right.text(0.3, 1.6, f"共變異數矩陣\n            {X}      {Y}\n{X}   {cov[0, 0]:6.2f}  {cov[0, 1]:6.2f}\n"
                        f"{Y}    {cov[1, 0]:6.2f}  {cov[1, 1]:6.2f}",
               fontsize=9.5, color=INK, va="top", linespacing=1.5)
    right.set_title("總變異數不變，只是重新分配：PC1 一個人就拿走 92%", loc="left", color=INK, fontsize=11)
    fig.suptitle("④ 電腦不用一個角度一個角度試：對共變異數矩陣求特徵向量，一次算出來",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "pca2d-4-ellipse.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return vals


def fig_fair(m):
    s = m["sample"]
    p = s.loc[HIGHLIGHT].to_numpy()
    u1 = m["components"][0]
    score = p @ u1
    foot = score * u1
    lo, hi = -1.0, 6.0
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(s[X], s[Y], s=18, color=VIOLET, alpha=0.3, lw=0)
    t = np.linspace(-2, 8, 2)
    ax.plot(t * u1[0], t * u1[1], color=RED, lw=2.4, label="PC1：「兩檔一起動」的那條線")
    # origin -> foot: the factor score
    ax.annotate("", xy=foot, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.4))
    ax.text(foot[0] / 2 + 0.15, foot[1] / 2 - 0.35, f"PC1 因子值 = {score:.2f}%\n"
            f"= {u1[0]:.3f}×{p[0]:.2f} + {u1[1]:.3f}×{p[1]:.2f}", color=INK, fontsize=9.5, ha="left", va="top")
    # foot -> actual: the residual
    ax.annotate("", xy=p, xytext=foot, arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=2.4))
    ax.plot([foot[0], p[0]], [foot[1], foot[1]], color=BLUE, lw=1.8, ls="--")
    ax.plot([p[0], p[0]], [foot[1], p[1]], color=ORANGE, lw=1.8, ls="--")
    ax.text(p[0], foot[1] - 0.12, f"{X} 殘差 {p[0] - foot[0]:+.2f}%", color=INK, fontsize=9.5,
            ha="center", va="top")
    ax.text(p[0] - 0.08, (foot[1] + p[1]) / 2, f"{Y} 殘差 {p[1] - foot[1]:+.2f}%", color=INK, fontsize=9.5,
            ha="right", va="center")
    ax.scatter(*foot, s=110, color=PAPER, edgecolor=RED, lw=2.4, zorder=6)
    ax.annotate(f"合理位置（垂足）\n{X} {foot[0]:+.2f}%  {Y} {foot[1]:+.2f}%", foot, xytext=(14, -6),
                textcoords="offset points", fontsize=9.5, color=INK, va="top")
    ax.scatter(*p, s=110, color=INK, edgecolor=PAPER, lw=2, zorder=6)
    ax.annotate(f"{HIGHLIGHT:%m-%d} 實際位置\n{X} {p[0]:+.2f}%  {Y} {p[1]:+.2f}%", p, xytext=(-12, 10),
                textcoords="offset points", fontsize=9.5, color=INK, ha="right")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(f"{X}：比自己 60 天平均高幾 %")
    ax.set_ylabel(f"{Y}：比自己 60 天平均高幾 %")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"⑤ {HIGHLIGHT:%Y-%m-%d}：合理價 = 實際點在 PC1 線上的垂足，殘差 = 點離線的距離",
                 loc="left", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(OUT / "pca2d-5-fair.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p, foot, score


def fig_z(m):
    z, resid = m["z"], m["resid"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for ax, t, c in [(top, X, BLUE), (bottom, Y, ORANGE)]:
        zz = z[t]
        ax.bar(zz.index, zz, color=[RED if v < ENTRY else c for v in zz], width=1.0, alpha=0.9)
        ax.axhline(ENTRY, color=RED, lw=1.2, ls="--")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylim(-3.2, 3.2)
        ax.set_ylabel(f"{t} 的 z")
        last = zz.iloc[-1]
        ax.annotate(f"{HIGHLIGHT:%m-%d}：殘差 {resid[t].iloc[-1]:+.2f}% ÷ 標準差 {resid[t].std():.2f}% = z {last:+.2f}",
                    (zz.index[-1], last), xytext=(-330, 22 if last > 0 else -26), textcoords="offset points",
                    fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED))
        buys = (zz < ENTRY).sum()
        ax.text(zz.index[0], -3.05, f"z < -1.5（紅色，策略會買 {t}）：{buys} 天", color=RED, fontsize=9)
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    top.set_title("⑥ 兩檔的 z 分數是彼此的鏡像：一檔便宜，另一檔就一定貴", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "pca2d-6-zscores.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    use_cjk_font()
    m = model()
    s = m["sample"]
    lim = float(np.ceil(s.abs().to_numpy().max() + 1))
    u1 = m["components"][0]
    best = np.degrees(np.arctan2(u1[1], u1[0]))
    print("components", m["components"].round(3).tolist(), "best angle", round(best, 2))
    print("cov", m["cov"].round(2).tolist())

    fig_points(m, lim)
    v = fig_directions(m, lim, best)
    print("variance at 0/30/42/60/90:", [round(directional_variance(s, a), 2) for a in (0, 30, 42, 60, 90)],
          "max", round(v.max(), 2), "min", round(v.min(), 2), "at", int(v.argmin()))
    fig_shadows(m, lim, best)
    vals = fig_ellipse(m, lim)
    print("eigenvalues", vals.round(2), "share", (vals / vals.sum()).round(3))
    p, foot, score = fig_fair(m)
    print(f"{HIGHLIGHT.date()} point {p.round(2)}, PC1 score {score:.2f}, foot {foot.round(2)}, "
          f"resid {(p - foot).round(2)}")
    for t in (X, Y):
        f = m["fits"][t]
        print(t, "OLS slope", round(f.params.iloc[1], 3), "const", round(f.params.iloc[0], 6),
              "R2", round(f.rsquared, 3), "resid std", round(m["resid"][t].std(), 2),
              "z last", round(m["z"][t].iloc[-1], 2), "days z<-1.5", int((m["z"][t] < ENTRY).sum()))
    print("z_MA + z_V max abs:", float((m["z"][X] + m["z"][Y]).abs().max()))
    fig_z(m)


if __name__ == "__main__":
    main()
