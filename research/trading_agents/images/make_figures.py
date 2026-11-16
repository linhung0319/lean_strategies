"""產生 analyst_team_explained.md 用的所有圖。

所有資料都是虛構的「ABC 科技」示意資料（固定亂數種子，可重現）。
執行：python make_figures.py   （圖會輸出到本檔案所在的資料夾）
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent

# ---------- 樣式（參考 dataviz 預設色盤，淺色模式） ----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CRIT = "#d03b3b"
BLUE_LIGHT = "#cde2fb"
ORANGE_LIGHT = "#fbe1d6"
GRAY_LIGHT = "#f0efec"

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2,
    "font.size": 10,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def note(fig, text="示意資料：虛構的 ABC 科技，非真實數據"):
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=8, color=MUTED)


# ---------- 虛構股價資料 ----------
def make_prices():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range(end="2026-11-13", periods=450)
    n = len(dates)
    # 用幾個錨點畫出「劇本」：盤整下跌 → 上升趨勢（含拉回） → 整理 → 急漲 → 漲勢鈍化
    anchors = [(0, 82), (70, 78), (130, 72), (190, 66), (250, 61), (280, 69), (305, 67),
               (340, 81), (360, 79), (380, 91), (420, 89), (435, 103), (442, 105.5), (449, 107.5)]
    xs, ys = zip(*anchors)
    trend = np.exp(np.interp(np.arange(n), xs, np.log(ys)))
    # 繞著劇本的小幅隨機波動（AR(1)，不會越飄越遠）
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.007)
    close = trend * np.exp(noise)
    opn = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.007, n)))
    lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.007, n)))
    volume = 12e6 * (1 + 0.25 * rng.standard_normal(n)).clip(0.5)
    volume[420:436] *= 1.8
    volume[436:] *= 1.2
    return pd.DataFrame({"Open": opn, "High": hi, "Low": lo, "Close": close, "Volume": volume},
                        index=dates)


def add_indicators(df):
    c = df["Close"]
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    df["ema10"] = c.ewm(span=10, adjust=False).mean()
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macds"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macdh"] = df["macd"] - df["macds"]
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn)
    df["boll"] = c.rolling(20).mean()
    sd = c.rolling(20).std()
    df["boll_ub"] = df["boll"] + 2 * sd
    df["boll_lb"] = df["boll"] - 2 * sd
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vwma"] = (c * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()
    return df


# ---------- 畫圖小工具（流程圖） ----------
def box(ax, x, y, w, h, text, fc=GRAY_LIGHT, ec=BASE, tc=INK, fs=10, bold=False, lw=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.4)
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.4, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ================= 圖 01：系統全貌 =================
def fig01_pipeline():
    fig, ax = canvas(12, 5.4, (0, 12), (0, 5.4))
    ax.text(0.1, 5.15, "TradingAgents 的完整流程：分析師團隊是第一站", fontsize=14,
            fontweight="bold", color=INK)
    box(ax, 0.1, 2.15, 1.5, 1.0, "輸入\n股票代號 + 日期\n例：ABC, 2026-11-13", fs=9)
    # analyst team frame
    frame = FancyBboxPatch((2.0, 0.45), 3.3, 4.35, boxstyle="round,pad=0.02,rounding_size=0.1",
                           fc="#eef4fc", ec=BLUE, lw=1.6)
    ax.add_patch(frame)
    ax.text(3.65, 4.55, "① 分析師團隊（本文主角）", ha="center", fontsize=11, fontweight="bold",
            color=BLUE)
    names = ["技術面分析師\n價格、成交量、指標", "基本面分析師\n財報、估值、內部人",
             "新聞分析師\n新聞、總經數據、預測市場", "社群情緒分析師\nStockTwits、Reddit"]
    ys = [3.55, 2.6, 1.65, 0.7]
    for nm, y in zip(names, ys):
        box(ax, 2.25, y, 2.8, 0.75, nm, fc="white", ec=BLUE, fs=9.5)
        arrow(ax, 1.6, 2.65, 2.25, y + 0.375, color=MUTED, lw=1)
    ax.text(3.65, 0.52, "四人同時開工、彼此不看對方的報告", ha="center", fontsize=8.5, color=INK2)
    # downstream
    box(ax, 5.8, 3.3, 1.7, 0.9, "② 多方 vs 空方\n研究員辯論", fs=9.5)
    box(ax, 5.8, 1.9, 1.7, 0.9, "③ 研究經理\n裁決、投資計畫", fs=9.5)
    box(ax, 7.95, 1.9, 1.5, 0.9, "④ 交易員\n具體交易提案", fs=9.5)
    box(ax, 7.95, 3.3, 1.5, 0.9, "⑤ 三位風控\n激進/中立/保守", fs=9.5)
    box(ax, 9.9, 2.6, 1.95, 1.0, "⑥ 投資組合經理\n最終評等\n（買進…賣出）", fs=9.5, fc="#fff4e5",
        ec=YELLOW)
    arrow(ax, 5.3, 2.65, 5.8, 3.75)
    ax.text(5.33, 3.05, "四份\n報告", fontsize=8.5, color=INK2)
    arrow(ax, 6.65, 3.3, 6.65, 2.8)
    arrow(ax, 7.5, 2.35, 7.95, 2.35)
    arrow(ax, 8.7, 2.8, 8.7, 3.3)
    arrow(ax, 9.45, 3.75, 9.9, 3.2)
    save(fig, "fig01_pipeline.png")


# ================= 圖 02：一位分析師如何工作 =================
def fig02_analyst_anatomy():
    fig, ax = canvas(12, 5.2, (0, 12), (0, 5.2))
    ax.text(0.1, 4.95, "一位分析師的工作方式：兩種模式", fontsize=14, fontweight="bold", color=INK)
    # lane A
    ax.text(0.1, 4.4, "A. 技術面 / 基本面 / 新聞：AI 自己決定要查什麼", fontsize=11,
            fontweight="bold", color=BLUE)
    box(ax, 0.2, 2.9, 2.2, 1.1, "任務說明（prompt）\n你的角色 + 今天日期\n+ 可用工具清單", fs=9)
    box(ax, 3.0, 2.9, 1.9, 1.1, "AI 思考\n「我還缺什麼資料？」", fs=9.5, fc="#eef4fc", ec=BLUE)
    box(ax, 5.6, 2.9, 2.2, 1.1, "呼叫工具\n例：get_stock_data\n(ABC, 過去一年)", fs=9)
    box(ax, 8.5, 2.9, 1.9, 1.1, "工具回傳資料\n（不會超過分析日）", fs=9)
    box(ax, 10.0, 1.55, 1.9, 0.95, "寫報告\n＋結尾摘要表", fs=9.5, fc="#fff4e5", ec=YELLOW)
    arrow(ax, 2.4, 3.45, 3.0, 3.45)
    arrow(ax, 4.9, 3.6, 5.6, 3.6)
    ax.text(5.25, 3.72, "缺", fontsize=8.5, color=INK2, ha="center")
    arrow(ax, 7.8, 3.45, 8.5, 3.45)
    arrow(ax, 9.45, 4.0, 3.95, 4.0, rad=0.25)
    ax.text(6.7, 4.42, "重複，最多 20 輪", fontsize=9, color=INK2, ha="center")
    arrow(ax, 3.95, 2.9, 10.0, 2.02, rad=0.12)
    ax.text(6.2, 2.15, "資料夠了（或 20 輪用完）", fontsize=9, color=INK2, ha="center")
    # lane B
    ax.plot([0.1, 11.9], [1.35, 1.35], color=GRID, lw=1, solid_capstyle="butt")
    ax.text(0.1, 1.05, "B. 社群情緒：程式先把資料抓好，AI 只負責讀和判斷", fontsize=11,
            fontweight="bold", color=ORANGE)
    box(ax, 0.2, 0.1, 2.6, 0.75, "程式先抓：Yahoo 新聞\n+ StockTwits + Reddit", fs=9)
    box(ax, 3.4, 0.1, 2.6, 0.75, "全部塞進 prompt\n（AI 不能呼叫工具）", fs=9)
    box(ax, 6.6, 0.1, 2.0, 0.75, "AI 分析", fs=9.5, fc=ORANGE_LIGHT, ec=ORANGE)
    box(ax, 9.2, 0.1, 2.7, 0.75, "固定格式：等級、分數\n信心度、敘述", fs=9, fc="#fff4e5",
        ec=YELLOW)
    arrow(ax, 2.8, 0.47, 3.4, 0.47)
    arrow(ax, 6.0, 0.47, 6.6, 0.47)
    arrow(ax, 8.6, 0.47, 9.2, 0.47)
    save(fig, "fig02_analyst_anatomy.png")


# ================= 圖 03：時間尺度 =================
def fig03_time_horizons():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    rows = [
        ("社群情緒", [(0, 7, ORANGE, "過去 7 天的貼文與新聞標題")]),
        ("新聞", [(0, 7, AQUA, "新聞：7 天"), (7, 365, "#a6e3cc", "總經數據：預設 1 年")]),
        ("技術面", [(0, 30, BLUE, "指標：近 30 天"), (30, 300, BLUE_LIGHT, "價格歷史：通常抓數月到一年")]),
        ("基本面", [(0, 730, "#d6d0f5", "財報：最近幾季到幾年")]),
    ]
    for i, (name, segs) in enumerate(rows):
        for s, e, col, lab in segs:
            ax.barh(i, e - s, left=max(s, 0.8), height=0.5, color=col, edgecolor=SURFACE, lw=2)
            xm = np.sqrt(max(s, 0.8) * e)
            dark = col in (ORANGE, AQUA, BLUE)
            ax.text(xm, i, lab, ha="center", va="center", fontsize=9,
                    color="white" if dark else INK)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1000)
    ax.set_xticks([1, 7, 30, 90, 365, 730])
    ax.set_xticklabels(["1 天", "1 週", "1 個月", "1 季", "1 年", "2 年"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10, color=INK)
    ax.invert_xaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("四位分析師「往回看多久」（最右邊是今天，越往左看得越遠；對數刻度）")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig03_time_horizons.png")


# ================= 圖 04：OHLCV / K 線 =================
def fig04_ohlcv(df):
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3.0], height_ratios=[3, 1.1], wspace=0.25,
                          hspace=0.08)
    # 左：示意 K 棒
    ax0 = fig.add_subplot(gs[:, 0])
    ax0.axis("off")
    ax0.set_xlim(0, 4)
    ax0.set_ylim(84, 111)
    ax0.set_title("一根 K 棒 = 一天", loc="left")
    o, h, l, c = 100, 109, 96, 106
    ax0.plot([1.5, 1.5], [l, h], color=INK2, lw=1.5)
    ax0.add_patch(Rectangle((1.1, o), 0.8, c - o, fc=BLUE, ec=BLUE))
    for v, lab in [(h, f"最高價 High {h}"), (c, f"收盤價 Close {c}"), (o, f"開盤價 Open {o}"),
                   (l, f"最低價 Low {l}")]:
        ax0.plot([1.95, 2.2], [v, v], color=MUTED, lw=1)
        ax0.text(2.25, v, lab, va="center", fontsize=9.5, color=INK)
    ax0.text(0.1, 84.5, "收盤 > 開盤 → 上漲日（藍）\n收盤 < 開盤 → 下跌日（橘）\n"
             "※台股習慣紅漲綠跌、美股相反，\n   本文用藍/橘避免混淆", fontsize=8.5, color=INK2,
             va="bottom")
    # 右：最近 25 天
    d = df.tail(25)
    x = np.arange(len(d))
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    for i, (_, r) in enumerate(d.iterrows()):
        col = BLUE if r.Close >= r.Open else ORANGE
        ax1.plot([i, i], [r.Low, r.High], color=col, lw=1.2)
        ax1.add_patch(Rectangle((i - 0.32, min(r.Open, r.Close)), 0.64,
                                max(abs(r.Close - r.Open), 0.08), fc=col, ec=col))
        ax2.bar(i, r.Volume / 1e6, color=col, width=0.64)
    ax1.set_title("ABC 最近 25 個交易日：價格（上）與成交量（下）")
    ax1.set_ylabel("股價（美元）")
    ax2.set_ylabel("成交量\n（百萬股）")
    ax1.tick_params(labelbottom=False)
    ticks = x[::5]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([d.index[i].strftime("%m/%d") for i in ticks])
    ax1.set_xlim(-1, len(d))
    ax2.grid(axis="x", visible=False)
    ax1.grid(axis="x", visible=False)
    note(fig)
    save(fig, "fig04_ohlcv.png")


# ================= 圖 05：趨勢（均線） =================
def fig05_trend(df):
    d = df.iloc[-250:]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d.index, d.Close, color=INK2, lw=1.3, label="收盤價")
    ax.plot(d.index, d.ema10, color=AQUA, lw=1.5, label="10 日 EMA（短期）")
    ax.plot(d.index, d.sma50, color=BLUE, lw=2, label="50 日均線（中期）")
    ax.plot(d.index, d.sma200, color=ORANGE, lw=2, label="200 日均線（長期）")
    # golden cross
    diff = np.sign(d.sma50 - d.sma200)
    cross = d.index[(diff.diff() > 0).fillna(False).values]
    if len(cross):
        cx = cross[0]
        cy = d.loc[cx, "sma50"]
        ax.scatter([cx], [cy], s=90, color=YELLOW, zorder=5, ec=SURFACE, lw=2)
        ax.annotate("黃金交叉\n50 日線由下往上穿過 200 日線\n→ 中期趨勢轉強", xy=(cx, cy),
                    xytext=(cx - pd.Timedelta(days=95), cy + 14), fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    last = d.iloc[-1]
    for col, val, c in [("收盤 ", last.Close, INK2), ("50 日 ", last.sma50, BLUE),
                        ("200 日 ", last.sma200, ORANGE)]:
        ax.text(d.index[-1] + pd.Timedelta(days=4), val, f"{col}{val:.1f}", va="center",
                fontsize=9, color=INK)
    ax.text(d.index[5], d.Close.max() - 1, "現在：價格 > 50 日線 > 200 日線\n= 典型上升趨勢",
            fontsize=10, color=INK, va="top",
            bbox=dict(fc="white", ec=BASE, boxstyle="round,pad=0.4"))
    ax.set_title("ABC 過去一年：用移動平均線看「趨勢方向」")
    ax.set_ylabel("股價（美元）")
    ax.legend(loc="lower right", ncol=4)
    ax.set_xlim(d.index[0], d.index[-1] + pd.Timedelta(days=40))
    note(fig)
    save(fig, "fig05_trend.png")
    return cross[0] if len(cross) else None


# ================= 圖 06：動能（RSI + MACD） =================
def fig06_momentum(df):
    d = df.iloc[-70:].copy()
    x = np.arange(len(d))
    pos = {ts: i for i, ts in enumerate(d.index)}
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.3, 1.3], "hspace": 0.18})
    a0, a1, a2 = axes
    a0.plot(x, d.Close, color=INK2, lw=1.6)
    a0.set_title("ABC 最近 70 個交易日：價格還在創新高，但「力道」在變弱")
    a0.set_ylabel("收盤價")
    tail = d.iloc[-25:]
    p1 = tail.iloc[:12].Close.idxmax()
    p2 = tail.Close.idxmax()
    i1, i2 = pos[p1], pos[p2]
    a0.scatter([i1, i2], [d.loc[p1, "Close"], d.loc[p2, "Close"]], color=INK, s=30, zorder=5)
    a0.plot([i1, i2], [d.loc[p1, "Close"], d.loc[p2, "Close"]], color=INK, lw=1)
    a0.text(i2, d.loc[p2, "Close"] + 1.2, "價格：後面的高點比前面高", ha="right", fontsize=9.5,
            color=INK)
    # RSI
    a1.plot(x, d.rsi, color=BLUE, lw=1.8)
    a1.axhspan(70, 100, color=ORANGE_LIGHT, lw=0)
    a1.axhspan(0, 30, color=BLUE_LIGHT, lw=0)
    a1.axhline(70, color=ORANGE, lw=1)
    a1.axhline(30, color=BLUE, lw=1)
    a1.text(18, 88, "> 70 超買區（短期漲太快）", fontsize=9, color=INK)
    a1.text(18, 10, "< 30 超賣區（短期跌太深）", fontsize=9, color=INK)
    a1.set_ylim(0, 100)
    a1.set_ylabel("RSI")
    a1.text(x[-1] + 0.8, d.rsi.iloc[-1], f"{d.rsi.iloc[-1]:.0f}", va="center", fontsize=9.5,
            color=INK)
    # MACD hist
    cols = [BLUE if v >= 0 else ORANGE for v in d.macdh]
    a2.bar(x, d.macdh, color=cols, width=0.7)
    a2.axhline(0, color=BASE, lw=1)
    a2.set_ylabel("MACD 柱狀圖")
    a2.scatter([i1, i2], [d.loc[p1, "macdh"], d.loc[p2, "macdh"]], color=INK, s=30, zorder=5)
    a2.plot([i1, i2], [d.loc[p1, "macdh"], d.loc[p2, "macdh"]], color=INK, lw=1)
    a2.text(i2, max(d.loc[p1, "macdh"], d.loc[p2, "macdh"]) + 0.15,
            "動能柱：後面的比前面短 → 「背離」警訊", ha="right", fontsize=9.5, color=INK)
    ticks = x[::10]
    a2.set_xticks(ticks)
    a2.set_xticklabels([d.index[i].strftime("%m/%d") for i in ticks])
    a2.set_xlim(-1, len(d) + 2)
    for a in axes:
        a.grid(axis="x", visible=False)
    note(fig)
    save(fig, "fig06_momentum.png")
    return p1, p2


# ================= 圖 07：布林通道 + ATR 停損 =================
def fig07_volatility(df):
    d = df.iloc[-60:]
    last = d.iloc[-1]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                 gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.22})
    xi = np.arange(len(d))
    a0.fill_between(xi, d.boll_lb, d.boll_ub, color=BLUE_LIGHT, lw=0, label="布林通道（上軌～下軌）")
    a0.plot(xi, d.boll, color=BLUE, lw=1.4, label="中軌（20 日均線）")
    a0.plot(xi, d.Close, color=INK2, lw=1.6, label="收盤價")
    for v, lab in [(last.boll_ub, "上軌"), (last.boll, "中軌"), (last.boll_lb, "下軌")]:
        a0.text(xi[-1] + 1, v, f"{lab} {v:.1f}", fontsize=9, va="center")
    a0.set_xlim(0, len(d) + 9)
    a0.set_xticks(xi[::10])
    a0.set_xticklabels([d.index[i].strftime("%m/%d") for i in xi[::10]])
    a0.set_title("布林通道：價格「正常活動範圍」")
    a0.legend(loc="upper left")
    a0.grid(axis="x", visible=False)
    # 停損示意
    entry = last.Close
    atr = last.atr
    stops = [("停損設 1 元", entry - 1, ORANGE), (f"停損設 2×ATR（{2*atr:.1f} 元）", entry - 2 * atr, BLUE)]
    a1.set_xlim(0, 10)
    a1.set_ylim(entry - 2 * atr - 2.5, entry + 3)
    rng = np.random.default_rng(3)
    path = entry + np.cumsum(rng.normal(0, atr * 0.45, 10))
    path[0] = entry
    path[3] = entry - 1.6  # 正常波動洗出場
    path[4:] = np.linspace(entry - 0.8, entry + 2.2, 6) + rng.normal(0, 0.3, 6)
    a1.plot(range(10), path, color=INK2, lw=1.6, marker="o", ms=4)
    a1.axhline(entry, color=MUTED, lw=1)
    a1.text(9.9, entry + 0.15, f"買進價 {entry:.1f}", ha="right", fontsize=9, color=INK)
    for lab, lvl, col in stops:
        a1.axhline(lvl, color=col, lw=1.6)
        a1.text(9.9, lvl + 0.15, f"{lab} → {lvl:.1f}", ha="right", fontsize=9, color=INK)
    a1.annotate("正常波動就碰到\n太近的停損，被洗出場", xy=(3, path[3]), xytext=(4.3, entry - 2.6),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED))
    a1.set_title(f"ATR = {atr:.1f}：ABC 平均一天晃動幅度")
    a1.set_xlabel("買進後第幾天")
    a1.set_ylabel("股價")
    a1.grid(axis="x", visible=False)
    note(fig)
    save(fig, "fig07_volatility.png")


# ================= 圖 08：三大報表的關係 =================
def fig08_statements():
    fig, ax = canvas(12, 4.6, (0, 12), (0, 4.6))
    ax.text(0.1, 4.35, "三大財務報表：同一家公司的三個視角", fontsize=14, fontweight="bold",
            color=INK)
    box(ax, 0.2, 0.5, 3.4, 3.3, "", fc="white", ec=BLUE, lw=1.5)
    ax.text(1.9, 3.45, "損益表", ha="center", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(1.9, 3.1, "一段期間的「成績單」", ha="center", fontsize=9.5, color=INK2)
    ax.text(0.45, 2.75, "營收（賣了多少）      68\n－ 成本                   －27\n"
            "＝ 毛利                     41\n－ 研發、行銷等費用  －24\n－ 稅                       －3\n"
            "＝ 淨利                     14", fontsize=9.5, color=INK, va="top", linespacing=1.55,
            family=["Microsoft JhengHei"])
    box(ax, 4.3, 0.5, 3.4, 3.3, "", fc="white", ec=AQUA, lw=1.5)
    ax.text(6.0, 3.45, "現金流量表", ha="center", fontsize=12, fontweight="bold", color="#0e7a54")
    ax.text(6.0, 3.1, "錢包「實際」進出多少現金", ha="center", fontsize=9.5, color=INK2)
    ax.text(4.55, 2.75, "淨利                        14\n＋ 折舊等非現金費用  ＋3\n"
            "－ 客戶還沒付的款        －1\n＝ 營業現金流            16\n－ 買設備（資本支出）－4\n"
            "＝ 自由現金流            12", fontsize=9.5, color=INK, va="top", linespacing=1.55)
    box(ax, 8.4, 0.5, 3.4, 3.3, "", fc="white", ec=VIOLET, lw=1.5)
    ax.text(10.1, 3.45, "資產負債表", ha="center", fontsize=12, fontweight="bold", color=VIOLET)
    ax.text(10.1, 3.1, "某一天的「照片」", ha="center", fontsize=9.5, color=INK2)
    ax.text(8.65, 2.75, "資產（擁有的）        400\n  其中現金               90\n"
            "負債（欠別人的）    150\n股東權益（淨值）    250\n\n資產 ＝ 負債 ＋ 股東權益", fontsize=9.5,
            color=INK, va="top", linespacing=1.55)
    arrow(ax, 3.6, 1.15, 4.3, 2.55, color=INK2)
    ax.text(3.95, 1.0, "淨利是\n起點", ha="center", fontsize=8.5, color=INK2)
    arrow(ax, 7.7, 1.15, 8.4, 2.1, color=INK2)
    ax.text(8.05, 1.0, "現金餘額\n寫入照片", ha="center", fontsize=8.5, color=INK2)
    ax.text(6.0, 0.15, "ABC 最近一季（單位：億美元，虛構）", ha="center", fontsize=9, color=MUTED)
    save(fig, "fig08_statements.png")


# ================= 圖 09：ABC 季度成績 =================
QUARTERS = ["25Q1", "25Q2", "25Q3", "25Q4", "26Q1", "26Q2", "26Q3"]
REV = np.array([43, 46, 50, 54, 58, 63, 68.0])
NI = np.array([7.8, 8.5, 9.6, 10.6, 11.6, 12.9, 14.0])
FCF = np.array([6.5, 7.4, 8.3, 9.1, 10.0, 11.2, 12.0])


def fig09_quarterly():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(QUARTERS))
    w = 0.26
    ax.bar(x - w, REV, w, color=BLUE, label="營收", edgecolor=SURFACE, lw=2)
    ax.bar(x, NI, w, color=AQUA, label="淨利", edgecolor=SURFACE, lw=2)
    ax.bar(x + w, FCF, w, color=VIOLET, label="自由現金流", edgecolor=SURFACE, lw=2)
    for xi, v in [(x[0] - w, REV[0]), (x[-1] - w, REV[-1])]:
        ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=9, color=INK)
    ax.text(x[-1], NI[-1] + 1, f"{NI[-1]:.0f}", ha="center", fontsize=9, color=INK)
    ax.text(x[-1] + w, FCF[-1] + 1, f"{FCF[-1]:.0f}", ha="center", fontsize=9, color=INK)
    yoy = (REV[-1] / REV[-5] - 1) * 100
    margin = NI[-1] / REV[-1] * 100
    ax.text(-0.4, 72, f"營收年增 {yoy:.0f}%（26Q3 vs 25Q3）\n淨利率 {margin:.1f}%（每賣 100 元賺 {margin:.0f} 元）\n"
            "自由現金流跟著淨利一起成長 → 賺的是「真錢」", fontsize=10, color=INK, va="top",
            bbox=dict(fc="white", ec=BASE, boxstyle="round,pad=0.4"))
    ax.set_xticks(x)
    ax.set_xticklabels(QUARTERS)
    ax.set_ylabel("億美元")
    ax.set_ylim(0, 80)
    ax.set_title("ABC 最近七季：營收、淨利、自由現金流")
    ax.legend(loc="upper center", ncol=3)
    ax.grid(axis="x", visible=False)
    note(fig)
    save(fig, "fig09_quarterly.png")


# ================= 圖 10：利率與「未來的錢」 =================
def fig10_rates():
    t = np.arange(0, 21)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for r, col, lab in [(0.01, BLUE, "利率 1%"), (0.05, ORANGE, "利率 5%")]:
        pv = 100 / (1 + r) ** t
        ax.plot(t, pv, color=col, lw=2.2, label=lab)
        ax.scatter([10], [pv[10]], color=col, s=50, zorder=5, ec=SURFACE, lw=2)
        ax.text(10.4, pv[10] + (2 if r == 0.01 else -5), f"{lab}：10 年後的 100 元\n現在只值 {pv[10]:.0f} 元",
                fontsize=9.5, color=INK)
        ax.text(20.3, pv[-1], f"{lab}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("幾年後才拿到這 100 元")
    ax.set_ylabel("換算成「今天」值多少元")
    ax.set_ylim(30, 105)
    ax.set_xlim(0, 22)
    ax.set_title("為什麼升息會打擊股價：利率越高，「未來的錢」在今天越不值錢")
    ax.legend(loc="upper right")
    ax.text(0.3, 33, "高成長公司的獲利多半在「很多年以後」，所以利率上升時跌得特別兇",
            fontsize=9.5, color=INK2)
    save(fig, "fig10_rates.png")


# ================= 圖 11：總經數據（小多圖） =================
def fig11_macro():
    months = pd.date_range(end="2026-10-31", periods=13, freq="M")
    fed = np.array([4.50, 4.50, 4.33, 4.33, 4.33, 4.25, 4.10, 4.10, 3.95, 3.95, 3.90, 3.88, 3.88])
    pce = np.array([2.95, 2.9, 2.92, 2.85, 2.8, 2.82, 2.78, 2.75, 2.76, 2.72, 2.7, 2.71, 2.69])
    spread = np.array([-0.08, -0.02, 0.05, 0.10, 0.15, 0.22, 0.25, 0.30, 0.36, 0.40, 0.45, 0.48, 0.52])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"wspace": 0.3})
    specs = [
        (fed, "聯邦基金利率（%）", "央行一路小幅降息", BLUE, None),
        (pce, "核心 PCE 通膨（年增 %）", "還在 2% 目標之上", ORANGE, 2.0),
        (spread, "殖利率曲線：10 年 - 2 年（%）", "從倒掛（<0）回到正常", VIOLET, 0.0),
    ]
    for ax, (y, title, msg, col, ref) in zip(axes, specs):
        ax.plot(months, y, color=col, lw=2)
        ax.scatter([months[-1]], [y[-1]], color=col, s=40, zorder=5, ec=SURFACE, lw=2)
        ax.text(months[-1], y[-1], f"  {y[-1]:.2f}", va="center", fontsize=9, color=INK)
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=1)
            ax.text(months[-1], ref, "2% 目標 " if ref == 2.0 else "0 線 ", va="bottom", ha="right",
                    fontsize=8.5, color=INK2)
        ax.set_title(title, fontsize=11)
        top = ref == 0.0
        ax.text(0.02, 0.94 if top else 0.04, msg, transform=ax.transAxes, fontsize=9.5, color=INK,
                va="top" if top else "bottom")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%y/%m"))
        ax.set_xticks(months[::4])
        ax.grid(axis="x", visible=False)
    axes[1].set_ylim(1.9, 3.1)
    fig.suptitle("新聞分析師從 FRED 拿到的數據長這樣（過去 1 年）", x=0.125, ha="left",
                 fontsize=13, fontweight="bold", y=1.03)
    note(fig, "示意資料：數字為虛構，僅說明圖表意義")
    save(fig, "fig11_macro.png")


# ================= 圖 12：預測市場 =================
def fig12_prediction():
    rng = np.random.default_rng(11)
    days = pd.date_range(end="2026-11-13", periods=31)
    base = np.r_[np.full(15, 0.55), np.linspace(0.55, 0.72, 16)]
    p = np.clip(base + rng.normal(0, 0.012, 31), 0, 1)
    p[0], p[-1] = 0.55, 0.72
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(days, p * 100, color=BLUE, lw=2.2)
    ax.scatter([days[0], days[-1]], [55, 72], color=BLUE, s=50, zorder=5, ec=SURFACE, lw=2)
    ax.text(days[0], 57.5, "一個月前 55%", fontsize=9.5, color=INK)
    ax.text(days[-1], 74.5, "現在 72%", fontsize=9.5, color=INK, ha="right")
    ax.annotate("就業數據轉弱的新聞出來後\n押注「會降息」的人變多", xy=(days[16], p[16] * 100),
                xytext=(days[3], 66), fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_ylim(40, 80)
    ax.set_ylabel("市場認為的機率（%）")
    ax.set_title("預測市場：「聯準會 12 月會降息嗎？」的機率變化")
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d"))
    ax.grid(axis="x", visible=False)
    note(fig, "示意資料：數字為虛構")
    save(fig, "fig12_prediction.png")


# ================= 圖 13：StockTwits 多空比例 =================
def fig13_stocktwits():
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11, 3.8), gridspec_kw={"height_ratios": [0.7, 1.25],
                                                                         "hspace": 0.55})
    bull, bear, none = 22, 3, 5
    tot = bull + bear + none
    left = 0
    for v, col, lab, tc in [(bull, BLUE, f"看多 {bull} 則", "white"),
                            (bear, ORANGE, f"看空 {bear} 則", "white"),
                            (none, "#d9d7d0", f"無標籤 {none} 則", INK)]:
        a0.barh(0, v, left=left, color=col, height=0.6, edgecolor=SURFACE, lw=2)
        a0.text(left + v / 2, 0, lab, ha="center", va="center", fontsize=9.5, color=tc)
        left += v
    a0.set_xlim(0, tot)
    a0.axis("off")
    a0.set_title(f"ABC 在 StockTwits 最近 {tot} 則貼文（發文者自己標的立場）")
    ratio = bull / (bull + bear) * 100
    # 量尺
    a1.set_xlim(0, 100)
    a1.set_ylim(0, 1)
    a1.axvspan(90, 100, color=ORANGE_LIGHT, lw=0)
    a1.plot([0, 100], [0.5, 0.5], color=BASE, lw=3, solid_capstyle="butt")
    for v, lab in [(50, "50%\n不確定"), (70, "70%\n溫和看多"), (90, "≧ 90%\n可能過熱\n（反向風險）")]:
        a1.plot([v, v], [0.38, 0.62], color=INK2, lw=1.4)
        a1.text(v, 0.3, lab, ha="center", va="top", fontsize=9, color=INK)
    a1.scatter([ratio], [0.5], s=140, color=BLUE, zorder=5, ec=SURFACE, lw=2)
    a1.text(ratio, 0.75, f"ABC：{ratio:.0f}%", ha="center", fontsize=10.5, fontweight="bold",
            color=INK)
    a1.axis("off")
    a1.set_title("有標立場的貼文中「看多」佔多少？（prompt 給的判讀標準）", fontsize=11)
    note(fig)
    save(fig, "fig13_stocktwits.png")


# ================= 圖 14：跨來源比較 =================
def fig14_cross_source():
    srcs = [("Yahoo 新聞（機構觀點）", 0.5, "3 篇講新合約、1 篇講反壟斷調查"),
            ("StockTwits（散戶即時）", 1.6, "88% 看多，接近過熱"),
            ("r/wallstreetbets（投機）", 1.8, "曬買權獲利截圖"),
            ("r/stocks（持平）", 0.6, "討論新合約的實際影響"),
            ("r/investing（長期）", -0.6, "擔心估值太高")]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    for i, (s, v, why) in enumerate(srcs[::-1]):
        col = BLUE if v > 0 else ORANGE
        ax.plot([0, v], [i, i], color=col, lw=2)
        ax.scatter([v], [i], color=col, s=90, zorder=5, ec=SURFACE, lw=2)
        ax.text(v + (0.12 if v > 0 else -0.12), i, why, va="center",
                ha="left" if v > 0 else "right", fontsize=9, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(srcs)))
    ax.set_yticklabels([s[0] for s in srcs[::-1]], fontsize=10, color=INK)
    ax.set_xlim(-2.6, 3.6)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_xticklabels(["明顯看空", "偏空", "中性", "偏多", "明顯看多"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_title("同一支股票，不同來源的「溫度」不一樣 → 落差本身就是訊號")
    note(fig, "示意：AI 讀完內容後的定性判斷，非程式計算的數字")
    save(fig, "fig14_cross_source.png")


# ================= 圖 15：證據地圖 =================
def fig15_evidence():
    fig, ax = canvas(12.5, 6.7, (0, 12.5), (0, 6.7))
    ax.text(0.1, 6.45, "四份報告交出去之後：同一份證據，多空兩方各取所需", fontsize=14,
            fontweight="bold", color=INK)
    tags = {"技": BLUE, "基": VIOLET, "新": AQUA, "情": ORANGE}
    reports = [("技", "技術面報告"), ("基", "基本面報告"), ("新", "新聞報告"), ("情", "情緒報告")]
    for i, (t, nm) in enumerate(reports):
        y = 4.6 - i * 1.15
        box(ax, 5.15, y, 2.2, 0.8, nm, fc="white", ec=tags[t], lw=1.6, fs=10.5, bold=True)
    bull = [("技", "上升趨勢：價格 > 50 日 > 200 日線"), ("基", "營收年增 36%、淨利率 21%"),
            ("基", "自由現金流跟著成長；CFO 買進"), ("新", "新合約；降息機率升到 72%"),
            ("情", "情緒偏多，主流話題是新合約")]
    bear = [("技", "RSI 超買 ＋ MACD 背離"), ("基", "本益比 43 倍、PEG 1.4：不便宜"),
            ("新", "反壟斷調查；通膨仍高"), ("情", "StockTwits 88% 看多：接近過熱"),
            ("情", "r/investing 擔心估值")]
    ax.text(2.3, 5.8, "多方研究員會引用", ha="center", fontsize=12, fontweight="bold", color=BLUE)
    ax.text(10.2, 5.8, "空方研究員會引用", ha="center", fontsize=12, fontweight="bold",
            color=ORANGE)
    for i, (t, txt) in enumerate(bull):
        y = 4.85 - i * 0.95
        box(ax, 0.1, y, 4.4, 0.7, "", fc="#eef4fc", ec=BLUE)
        ax.add_patch(Rectangle((0.2, y + 0.15), 0.4, 0.4, fc=tags[t], ec=tags[t]))
        ax.text(0.4, y + 0.35, t, ha="center", va="center", color="white", fontsize=9.5,
                fontweight="bold")
        ax.text(0.75, y + 0.35, txt, va="center", fontsize=9.5, color=INK)
    for i, (t, txt) in enumerate(bear):
        y = 4.85 - i * 0.95
        box(ax, 8.0, y, 4.4, 0.7, "", fc=ORANGE_LIGHT, ec=ORANGE)
        ax.add_patch(Rectangle((8.1, y + 0.15), 0.4, 0.4, fc=tags[t], ec=tags[t]))
        ax.text(8.3, y + 0.35, t, ha="center", va="center", color="white", fontsize=9.5,
                fontweight="bold")
        ax.text(8.65, y + 0.35, txt, va="center", fontsize=9.5, color=INK)
    arrow(ax, 5.15, 3.0, 4.5, 3.0, color=BLUE, lw=2)
    arrow(ax, 7.35, 3.0, 8.0, 3.0, color=ORANGE, lw=2)
    ax.text(6.25, 0.2, "→ 辯論 → 研究經理裁決 → 交易員 → 風控 → 投資組合經理", ha="center",
            fontsize=10, color=INK2)
    save(fig, "fig15_evidence.png")


def main():
    df = add_indicators(make_prices())
    last = df.iloc[-1]
    print("analysis date:", df.index[-1].date())
    for k in ["Open", "High", "Low", "Close", "Volume", "ema10", "sma50", "sma200", "rsi", "macd",
              "macds", "macdh", "boll", "boll_ub", "boll_lb", "atr", "vwma"]:
        print(f"  {k}: {last[k]:.2f}")
    print("  52w high/low:", round(df.Close.iloc[-252:].max(), 2), round(df.Close.iloc[-252:].min(), 2))
    print("  10 days ago close:", round(df.Close.iloc[-11], 2), " 25d ago:", round(df.Close.iloc[-26], 2))
    shares = 20.0  # 億股
    ttm_ni = NI[-4:].sum()
    eps = ttm_ni / shares
    print(f"  TTM NI {ttm_ni:.1f} 億, EPS {eps:.2f}, PE {last.Close/eps:.1f}, mcap {last.Close*shares:.0f} 億")
    print(f"  rev yoy {(REV[-1]/REV[-5]-1)*100:.1f}%  NI yoy {(NI[-1]/NI[-5]-1)*100:.1f}%  margin {NI[-1]/REV[-1]*100:.1f}%")
    fig01_pipeline()
    fig02_analyst_anatomy()
    fig03_time_horizons()
    fig04_ohlcv(df)
    gc = fig05_trend(df)
    print("  golden cross:", gc)
    p1, p2 = fig06_momentum(df)
    print("  peaks:", p1.date(), round(df.loc[p1, "Close"], 2), round(df.loc[p1, "macdh"], 3),
          p2.date(), round(df.loc[p2, "Close"], 2), round(df.loc[p2, "macdh"], 3))
    fig07_volatility(df)
    fig08_statements()
    fig09_quarterly()
    fig10_rates()
    fig11_macro()
    fig12_prediction()
    fig13_stocktwits()
    fig14_cross_source()
    fig15_evidence()


if __name__ == "__main__":
    main()
