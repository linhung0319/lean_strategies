"""Regenerate the figures and numbers in strategy-explained.md.

Re-runs the exact pipeline from algorithms/topological_portfolio.py against the
local data folder on one chosen date, so every figure and every number in the
document is a real output rather than a drawing of one.

    uv run --group topological python research/topological_portfolio/explain_figures.py

Needs `data-sp500/` (see "Constituents universes" in README.md). The equity
curve additionally needs the two backtests to have been run -- it reads their
result files out of `results/`, and skips that one figure if they are not
there. The figure labels are
Traditional Chinese, so it also wants a CJK font installed -- it picks the
first of FONT_CANDIDATES that matplotlib can find, and falls back to boxes
rather than failing if none is there.
"""

import io
import itertools
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from umap import UMAP

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data-sp500" / "equity" / "usa"
# Figures are written next to the document that embeds them; the backtest
# results they read stay where run_local.py puts them, which is regenerable
# output and therefore out of git.
OUT = Path(__file__).resolve().parent
RESULTS = REPO / "results"

# The date the document walks through, and a second one used only to show that
# the topology moves. Both are inside the backtest window.
MAIN_DATE = "2021-09-16"
CONTRAST_DATE = "2022-09-26"
LOOKBACK = 150
UNIVERSE_SIZE = 200

# Whichever of these is installed wins; the rest are fallbacks for other
# machines. DejaVu Sans is matplotlib's default and has no CJK coverage, so it
# is last and only keeps the Latin text readable.
FONT_CANDIDATES = ["Microsoft JhengHei", "Noto Sans TC", "Microsoft YaHei",
                   "PingFang TC", "Heiti TC", "Noto Sans CJK TC", "DejaVu Sans"]

INK = "#2c2c2c"
MUTED = "#8a8a8a"
GRID = "#e4e4e1"
PAPER = "#fcfcfb"
# Colourblind-safe categorical set, muted to sit quietly behind the labels.
PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756",
    "#72b7b2", "#eeca3b", "#9d755d", "#bab0ac", "#d67195",
    "#1b9e77", "#7570b3", "#a6761d",
]


# --------------------------------------------------------------------------
# the pipeline, reproduced
# --------------------------------------------------------------------------

def use_cjk_font():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in FONT_CANDIDATES if name in installed]
    if chosen[:1] == ["DejaVu Sans"] or not chosen:
        print("warning: no CJK font found; the Chinese labels will render as boxes")
    plt.rcParams["font.sans-serif"] = chosen or ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen[0] if chosen else "DejaVu Sans"


def read_closes(ticker, end, count):
    with zipfile.ZipFile(DATA / "daily" / f"{ticker.lower()}.zip") as archive:
        raw = archive.read(archive.namelist()[0]).decode()
    frame = pd.read_csv(io.StringIO(raw), header=None,
                        names=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["date"].astype(str).str[:8], format="%Y%m%d")
    # LEAN stores daily equity prices as deci-cents.
    return frame.set_index("date").loc[:end, "close"].iloc[-count:] / 10000.0


def selected_universe(day, size=UNIVERSE_SIZE):
    """The algorithm's filter: heaviest `size` constituents with a known weight."""
    files = sorted((DATA / "universes" / "etf" / "spy").glob("*.csv"))
    path = [p for p in files if p.stem <= day.replace("-", "")][-1]
    rows = [line.split(",") for line in path.read_text().splitlines() if line.strip()]
    ranked = sorted(((float(r[3]), r[0]) for r in rows if r[3]), reverse=True)[:size]
    return [t for _, t in ranked if (DATA / "daily" / f"{t.lower()}.zip").exists()], path.stem


def build(day):
    tickers, universe_file = selected_universe(day)
    prices = pd.DataFrame({t: read_closes(t, day, LOOKBACK) for t in tickers}).dropna(axis=1)
    log_returns = np.log(prices / prices.shift(1)).dropna().T

    mapper = km.KeplerMapper()
    lens = mapper.fit_transform(log_returns, projection=[
        PCA(n_components=0.8, random_state=1),
        UMAP(n_components=1, random_state=1, n_jobs=-1)])
    graph = mapper.map(lens, log_returns, DBSCAN(metric="correlation", n_jobs=-1))
    return dict(day=day, universe_file=universe_file, prices=prices,
                log_returns=log_returns, lens=lens, graph=graph,
                # The algorithm passes prices.columns, and clustering_symbols
                # relies on that being fancy-indexable by a list of positions.
                symbols=prices.columns, tickers=list(prices.columns))


# -- verbatim from algorithms/topological_portfolio.py ---------------------

def clustering_symbols(graph, symbol_list):
    linked_clusters = []
    for x, y in graph['links'].items():
        isin = False
        for i in range(len(linked_clusters)):
            if x in linked_clusters[i] or y in linked_clusters[i]:
                linked_clusters[i] = list(set(linked_clusters[i] + [x] + y))
                isin = True
        if isin:
            continue
        linked_clusters.append([x] + y)
    linked_clusters += [[x] for x in graph['nodes']
                        if x not in [z for y in linked_clusters for z in y]]
    return [[list([symbol_list[graph['nodes'][x]]][0]) for x in lc] for lc in linked_clusters]


def weight_distribution(clustered_symbols):
    weights = {}

    def assign_weights(nested_list, level=1):
        num_elements = len(nested_list)
        if num_elements == 0:
            return
        weight_per_element = 1 / num_elements
        for item in nested_list:
            if isinstance(item, list):
                assign_weights(item, level + 1)
            else:
                weights[item] = weights.get(item, 0) + weight_per_element / (2 ** (level - 1))

    assign_weights(clustered_symbols)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

# -- end verbatim ----------------------------------------------------------


def node_order(graph):
    """Nodes sorted the way the cover produced them: by cube, then cluster."""
    def key(name):
        cube, cluster = name.split("_")
        return int(cube.replace("cube", "")), int(cluster.replace("cluster", ""))
    return sorted(graph["nodes"], key=key)


def node_members(state, name):
    return [str(state["tickers"][i]) for i in state["graph"]["nodes"][name]]


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def save(fig, name):
    fig.patch.set_facecolor(PAPER)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {name}")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def figure_correlation(state, groups):
    """What DBSCAN actually measures: co-movement of daily returns, not level.

    ``groups`` is {label: [tickers]}, drawn in order so the block structure
    along the diagonal is the point of the picture.
    """
    returns = state["log_returns"].T          # columns = stocks
    selected = [t for members in groups.values() for t in members]
    corr = returns[selected].corr()

    fig = plt.figure(figsize=(12, 4.3))
    spec = fig.add_gridspec(1, 3, width_ratios=[1.55, 1, 1], wspace=0.45)

    ax = fig.add_subplot(spec[0, 0])
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(selected)))
    ax.set_yticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=90, fontsize=7.5, color=INK)
    ax.set_yticklabels(selected, fontsize=7.5, color=INK)
    edge = 0
    for label, members in groups.items():
        ax.add_patch(plt.Rectangle((edge - 0.5, edge - 0.5), len(members), len(members),
                                   fill=False, edgecolor=INK, linewidth=1.6))
        ax.annotate(label, (edge + len(members) / 2 - 0.5, -1.15), ha="center",
                    fontsize=8.5, color=INK, annotation_clip=False)
        edge += len(members)
    ax.tick_params(colors=MUTED, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("日報酬的相關係數矩陣", color=INK, fontsize=11, loc="left", pad=26)
    bar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=7.5, length=0)

    pairs = [(groups_pair, colour) for groups_pair, colour in
             zip(state["scatter_pairs"], (PALETTE[0], PALETTE[4]))]
    for index, ((left, right), colour) in enumerate(pairs):
        axis = fig.add_subplot(spec[0, index + 1])
        x = returns[left] * 100
        y = returns[right] * 100
        axis.scatter(x, y, s=16, color=colour, alpha=0.55, linewidths=0)
        axis.axhline(0, color=GRID, linewidth=1)
        axis.axvline(0, color=GRID, linewidth=1)
        rho = returns[left].corr(returns[right])
        axis.annotate(f"相關係數 {rho:.2f}", (0.03, 0.93), xycoords="axes fraction",
                      fontsize=9, color=INK)
        limit = max(abs(x).max(), abs(y).max()) * 1.08
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        style(axis, title=f"{left} vs {right}",
              xlabel=f"{left} 日報酬 %", ylabel=f"{right} 日報酬 %")

    fig.suptitle(f"演算法看的是「同一天一起漲跌」，不是誰漲得多  ({state['day']} 前 {LOOKBACK} 個交易日)",
                 color=INK, fontsize=11, x=0.005, ha="left", y=1.06)
    save(fig, "fig1-what-it-measures.png")


def figure_lens(state):
    """The 1-D lens value per stock, and the overlapping cover laid over it."""
    lens = state["lens"].ravel()
    graph = state["graph"]
    fig, ax = plt.subplots(figsize=(11, 3.4))

    # KeplerMapper's default cover: 10 intervals with 10% overlap. Drawn on two
    # staggered rows in one neutral colour, so the eye reads them as the cover
    # rather than as the clusters the dots below are coloured by.
    n_cubes, overlap = 10, 0.1
    low, high = lens.min(), lens.max()
    width = (high - low) / n_cubes
    pad = width * overlap / 2
    for index in range(n_cubes):
        left = low + index * width - pad
        right = left + width + 2 * pad
        row = index % 2
        bottom = 0.70 + row * 0.10
        ax.add_patch(plt.Rectangle((left, bottom), right - left, 0.075,
                                   facecolor="#b9c6d4", edgecolor="#8fa3b8",
                                   linewidth=0.8, alpha=0.75, clip_on=False))
        ax.annotate(f"cube{index}", ((left + right) / 2, bottom + 0.037),
                    ha="center", va="center", fontsize=7.5, color="#3d4c5c")

    rng = np.random.default_rng(3)
    jitter = rng.uniform(0.06, 0.52, size=len(lens))
    member_of = {}
    for position, name in enumerate(node_order(graph)):
        for index in graph["nodes"][name]:
            member_of.setdefault(index, position)
    colours = [PALETTE[member_of[i] % len(PALETTE)] if i in member_of else "#cfcfcc"
               for i in range(len(lens))]
    ax.scatter(lens, jitter, s=22, c=colours, alpha=0.85, linewidths=0)

    for ticker in ("XOM", "JPM", "KO", "AAPL", "AMT", "MRK", "V"):
        if ticker in state["tickers"]:
            i = state["tickers"].index(ticker)
            ax.annotate(ticker, (lens[i], jitter[i]), fontsize=7.5, color=INK,
                        xytext=(0, 7), textcoords="offset points", ha="center")

    ax.set_ylim(0, 0.90)
    ax.set_yticks([])
    style(ax, title=f"每檔股票被壓成一個數字，再切成 10 個重疊的區間  ({state['day']})",
          xlabel="PCA → UMAP 之後的一維座標（lens）")
    ax.annotate("灰點 = DBSCAN 判定的離群股，不進任何 node", (0.995, 0.02),
                xycoords="axes fraction", ha="right", fontsize=8, color=MUTED)
    save(fig, "fig2-lens-and-cover.png")


def figure_complex(state, labels):
    """The simplicial complex, laid out on the cover axis it came from."""
    graph = state["graph"]
    names = node_order(graph)
    position = {}
    for name in names:
        cube = int(name.split("_")[0].replace("cube", ""))
        cluster = int(name.split("_")[1].replace("cluster", ""))
        position[name] = (cube, -cluster)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    for source, targets in graph["links"].items():
        for target in targets:
            x = [position[source][0], position[target][0]]
            y = [position[source][1], position[target][1]]
            ax.plot(x, y, color=MUTED, linewidth=1.6, alpha=0.6, zorder=1)

    for order, name in enumerate(names):
        size = len(graph["nodes"][name])
        x, y = position[name]
        ax.scatter([x], [y], s=size * 34, color=PALETTE[order % len(PALETTE)],
                   alpha=0.85, linewidths=0, zorder=2)
        ax.annotate(str(size), (x, y), ha="center", va="center", fontsize=8,
                    color="white", zorder=3, weight="bold")
        ax.annotate(labels.get(name, name), (x, y), ha="center", fontsize=8.5,
                    color=INK, xytext=(0, -26 if y == 0 else 22),
                    textcoords="offset points", zorder=3)

    ax.set_xticks(range(10))
    ax.set_xticklabels([f"cube{i}" for i in range(10)])
    ax.set_yticks([])
    ax.set_ylim(-1.45, 0.62)
    ax.margins(x=0.06)
    style(ax, title=f"Mapper 複形：{len(names)} 個 node，{sum(len(v) for v in graph['links'].values())} 條邊  ({state['day']})",
          xlabel="node 來自哪個區間（圓圈大小 = 成員數，連線 = 兩個 node 共用股票）")
    save(fig, "fig3-simplicial-complex.png")


def figure_weights(state, labels):
    """Where the portfolio weight goes, node by node."""
    clustered = clustering_symbols(state["graph"], state["symbols"])
    weights = weight_distribution(clustered)
    names = node_order(state["graph"])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.0),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    sizes = [len(state["graph"]["nodes"][n]) for n in names]
    per_stock = [1 / s for s in sizes]
    scale = sum(per_stock[i] * sizes[i] for i in range(len(names)))
    bars = ax.barh(range(len(names)), [p / scale * 100 for p in per_stock],
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(names))], alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{labels.get(n, n)}  (n={len(state['graph']['nodes'][n])})"
                        for n in names], fontsize=8.5, color=INK)
    ax.invert_yaxis()
    for bar, size in zip(bars, sizes):
        ax.annotate(f"1/{size}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    style(ax, title="每檔股票分到的權重，只由它所在 node 的大小決定",
          xlabel="單一成員的權重 %")

    ordered = sorted(weights.items(), key=lambda kv: -kv[1])
    values = [v * 100 for _, v in ordered]
    ax2.plot(range(1, len(values) + 1), values, color=PALETTE[0], linewidth=1.8)
    ax2.fill_between(range(1, len(values) + 1), values, color=PALETTE[0], alpha=0.15)
    for rank in (0, len(values) - 1):
        ax2.annotate(f"{str(ordered[rank][0])} {values[rank]:.2f}%",
                     (rank + 1, values[rank]), fontsize=8, color=INK,
                     xytext=(6 if rank == 0 else -6, 6),
                     ha="left" if rank == 0 else "right", textcoords="offset points")
    ax2.grid(axis="y", color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    style(ax2, title=f"實際持倉 {len(weights)} 檔（股票池 {len(state['tickers'])} 檔）",
          xlabel="由重到輕排序", ylabel="權重 %")
    save(fig, "fig4-weights.png")
    return clustered, weights


def equity_curve(result_json):
    """Daily closing equity out of a LEAN result file, indexed by date.

    The run's own Benchmark series is not usable: LEAN's internal benchmark
    feed is hourly in backtesting and a converted data folder has daily bars
    only, so it comes back as a row of zeros. The comparison line is the
    separate "SPY Buy & Hold 2021-2026" variant instead.
    """
    if not result_json.exists():
        return None
    charts = json.loads(result_json.read_text(encoding="utf-8"))["charts"]
    points = pd.DataFrame(charts["Strategy Equity"]["series"]["Equity"]["values"])
    # Each row is [unix seconds, open, high, low, close].
    closes = points.iloc[:, 4] if points.shape[1] >= 5 else points.iloc[:, 1]
    series = pd.Series(closes.astype(float).to_numpy(),
                       index=pd.to_datetime(points.iloc[:, 0], unit="s"))
    return series[~series.index.duplicated(keep="last")]


def figure_equity():
    equity = equity_curve(
        RESULTS / "topological_portfolio" / "TopologicalPortfolio.json")
    baseline = equity_curve(
        RESULTS / "spy_buy_and_hold_2021_2026" / "StockBuyAndHold.json")
    if equity is None:
        print(f"skipped fig5: no backtest result under {RESULTS}; "
              f'run `run_local.py "Topological Portfolio" --data-folder data-sp500` first')
        return

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(equity.index, equity / equity.iloc[0] * 100, color=PALETTE[0],
            linewidth=1.8, label="Topological Portfolio")
    if baseline is not None:
        ax.plot(baseline.index, baseline / baseline.iloc[0] * 100,
                color=MUTED, linewidth=1.4, label="SPY Buy & Hold")
    ax.axhline(100, color=GRID, linewidth=1)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    style(ax, title="權益曲線（起點 = 100）", ylabel="指數")
    save(fig, "fig5-equity.png")


def report_stability(state, shifted_days=("2021-09-15", "2021-09-09")):
    """How much does the grouping move when the window shifts by a day or a week?

    Two different questions, with two very different answers: *which* stocks
    get held barely moves, *who is grouped with whom* moves a lot. Since the
    weight of a stock depends only on the size of its node, the second one is
    what the portfolio actually feels.
    """
    def fingerprint(built):
        members = [set(node_members(built, name)) for name in built["graph"]["nodes"]]
        held = set().union(*members)
        pairs = {pair for group in members
                 for pair in itertools.combinations(sorted(group), 2)}
        return held, pairs

    def jaccard(left, right):
        return len(left & right) / len(left | right)

    held, pairs = fingerprint(state)
    print(f"\nstability, against {state['day']} "
          f"({len(held)} held, {len(pairs)} same-node pairs)")
    for day in shifted_days:
        other_held, other_pairs = fingerprint(build(day))
        print(f"  vs {day}: held {len(other_held):>3} "
              f"(common {len(held & other_held):>3}, jaccard {jaccard(held, other_held):.2f})"
              f"   pairs {len(other_pairs):>5} "
              f"(common {len(pairs & other_pairs):>4}, jaccard {jaccard(pairs, other_pairs):.2f})")


# --------------------------------------------------------------------------

def main():
    print(f"font: {use_cjk_font()}")
    state = build(MAIN_DATE)
    contrast = build(CONTRAST_DATE)

    names = node_order(state["graph"])
    print(f"\n=== {MAIN_DATE} (universe file {state['universe_file']}) ===")
    print(f"universe {len(state['symbols'])}  lookback rows {state['log_returns'].shape[1]}")
    print(f"nodes {len(names)}  links {sum(len(v) for v in state['graph']['links'].values())}")
    for name in names:
        members = node_members(state, name)
        print(f"  {name:<16} n={len(members):>3}  {', '.join(members)}")
    print("  links:", dict(state["graph"]["links"]))

    # Short labels are a human reading of the member lists printed above.
    labels = {
        "cube0_cluster0": "銀行 + 能源",
        "cube1_cluster0": "旅遊 / 循環",
        "cube2_cluster0": "半導體",
        "cube3_cluster0": "成長科技",
        "cube4_cluster0": "大型軟體",
        "cube4_cluster1": "支付",
        "cube5_cluster0": "生技工具",
        "cube5_cluster1": "醫療器材",
        "cube6_cluster0": "商業服務",
        "cube7_cluster0": "防禦 / 公用",
        "cube8_cluster0": "REITs",
        "cube8_cluster1": "製藥",
        "cube9_cluster0": "工業 + 保險",
    }

    state["scatter_pairs"] = [("AMAT", "LRCX"), ("INTC", "PSA")]
    figure_correlation(state, {
        "半導體": node_members(state, "cube2_cluster0")[:4],
        "REITs": node_members(state, "cube8_cluster0")[:4],
        "防禦 / 公用": node_members(state, "cube7_cluster0")[:4],
    })
    figure_lens(state)
    figure_complex(state, labels)
    clustered, weights = figure_weights(state, labels)
    figure_equity()

    print(f"\nconnected components: {len(clustered)}  sizes {[len(c) for c in clustered]}")
    print(f"held {len(weights)} of {len(state['tickers'])}; "
          f"weights {min(weights.values()):.4f}..{max(weights.values()):.4f}")
    appearances = {}
    for cluster in clustered:
        for node in cluster:
            for symbol in node:
                appearances[symbol] = appearances.get(symbol, 0) + 1
    repeats = sorted(str(s) for s, n in appearances.items() if n > 1)
    print(f"counted more than once ({len(repeats)}): {', '.join(repeats)}")

    report_stability(state)

    print(f"\n=== {CONTRAST_DATE} (universe file {contrast['universe_file']}) ===")
    for name in node_order(contrast["graph"]):
        members = node_members(contrast, name)
        print(f"  {name:<16} n={len(members):>3}  {', '.join(members)}")


if __name__ == "__main__":
    main()
