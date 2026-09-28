"""Named strategy variants for run_local.py.

The first 16 are the ports of sp500/main.py:14-37 and run against LEAN's
bundled SPY. The rest run against symbols produced by convert_data.py, so
they need `--data-folder data`.

Local tooling only. Never uploaded to QuantConnect — the files in algorithms/
are self-contained and carry their own defaults.
"""

import re

ALGORITHM_CLASSES = {
    "spy_buy_and_hold": "SpyBuyAndHold",
    "spy_threshold_rebalance": "SpyThresholdRebalance",
    "spy_periodic_rebalance": "SpyPeriodicRebalance",
    "spy_ma_trend": "SpyMaTrend",
    "spy_ma_entry_exit": "SpyMaEntryExit",
    "spy_vol_adjusted": "SpyVolAdjusted",
    "spy_momentum": "SpyMomentum",
    "stock_buy_and_hold": "StockBuyAndHold",
    "stock_ma_entry_exit": "StockMaEntryExit",
    "stock_monthly_rebalance": "StockMonthlyRebalance",
    "topological_portfolio": "TopologicalPortfolio",
    "tech_ema_reversion": "TechEmaReversion",
    "momentum_effect": "MomentumEffect",
}

VARIANTS = {
    # Threshold variants
    "50/50 Rebalance ±1%": ("spy_threshold_rebalance", {"target": 0.50, "threshold": 0.01}),
    "50/50 Rebalance ±5%": ("spy_threshold_rebalance", {"target": 0.50, "threshold": 0.05}),
    "50/50 Rebalance ±10%": ("spy_threshold_rebalance", {"target": 0.50, "threshold": 0.10}),
    "50/50 Rebalance ±20%": ("spy_threshold_rebalance", {"target": 0.50, "threshold": 0.20}),
    # Different target allocations
    "60/40 Rebalance ±5%": ("spy_threshold_rebalance", {"target": 0.60, "threshold": 0.05}),
    "70/30 Rebalance ±5%": ("spy_threshold_rebalance", {"target": 0.70, "threshold": 0.05}),
    "80/20 Rebalance ±5%": ("spy_threshold_rebalance", {"target": 0.80, "threshold": 0.05}),
    # Periodic rebalancing
    "50/50 Monthly Rebalance": ("spy_periodic_rebalance", {"target": 0.50, "frequency": "M"}),
    "50/50 Quarterly Rebalance": ("spy_periodic_rebalance", {"target": 0.50, "frequency": "Q"}),
    "50/50 Annual Rebalance": ("spy_periodic_rebalance", {"target": 0.50, "frequency": "Y"}),
    # Buy and hold benchmarks
    "100% Buy & Hold SPY": ("spy_buy_and_hold", {"spy_weight": 1.0}),
    "50/50 No Rebalance": ("spy_buy_and_hold", {"spy_weight": 0.5}),
    # Dynamic allocation
    "MA Trend Following (200d)": (
        "spy_ma_trend",
        {"ma_period": 200, "above_weight": 0.60, "below_weight": 0.40},
    ),
    "Volatility Adjusted (20d)": (
        "spy_vol_adjusted",
        {"lookback": 20, "target_vol": 0.15, "min_weight": 0.10, "max_weight": 0.90},
    ),
    # Market timing
    "200-day MA Entry/Exit": ("spy_ma_entry_exit", {"ma_period": 200}),
    "12-Month Momentum": ("spy_momentum", {"lookback_months": 12}),
}

# Converted data. These need a data folder built by convert_data.py:
#   uv run --group data convert_data.py convert --source yfinance --ticker NVDA \
#       --exchange-code Q --start 2015-01-02 --end 2026-08-07
#   uv run run_local.py "NVDA 200-day MA Entry/Exit" --data-folder data
NVDA_WINDOW = {"ticker": "NVDA", "start_date": "2015-01-02", "end_date": "2026-08-07"}

VARIANTS.update(
    {
        "NVDA Buy & Hold": ("stock_buy_and_hold", {**NVDA_WINDOW, "weight": 1.0}),
        "NVDA 200-day MA Entry/Exit": ("stock_ma_entry_exit", {**NVDA_WINDOW, "ma_period": 200}),
        "NVDA 50-day MA Entry/Exit": ("stock_ma_entry_exit", {**NVDA_WINDOW, "ma_period": 50}),
    }
)

# Constituents universe. Needs the folder built by the `universe` command and
# the ML stack from the `topological` dependency group:
#   uv run --group data convert_data.py universe --etf SPY \
#       --start 2020-01-02 --end 2026-09-15 --out data-sp500
#   uv run --group topological run_local.py "Topological Portfolio" \
#       --data-folder data-sp500
VARIANTS["Topological Portfolio"] = (
    "topological_portfolio",
    {
        "start_date": "2021-09-16",
        "end_date": "2026-09-15",
        "history_lookback": 150,
        "recalibrate_period": 125,
        "universe_size": 200,
    },
)

# Its like-for-like baseline. LEAN's internal benchmark feed is hourly in
# backtesting (Lean/Engine/DataFeeds/UniverseSelection.cs:337), so a daily-only
# data folder leaves Alpha, Beta and Treynor at zero -- running buy-and-hold
# over the same window and the same data folder is how the comparison gets
# made here instead.
VARIANTS["SPY Buy & Hold 2021-2026"] = (
    "stock_buy_and_hold",
    {"ticker": "SPY", "start_date": "2021-09-16", "end_date": "2026-09-15", "weight": 1.0},
)

# Top-5 Technology mean reversion. Same data folder as the topological run:
#   uv run run_local.py "Tech Top5 EMA Reversion" --data-folder data-sp500
VARIANTS["Tech Top5 EMA Reversion"] = (
    "tech_ema_reversion",
    {
        "start_date": "2021-09-16",
        "end_date": "2026-09-15",
        "universe_size": 5,
        "ema_period": 100,
        "max_period": 252,
    },
)

# The same rules over the whole index instead of Technology only. The second
# one drops GOOG: the weight proxy credits Alphabet's full share count to both
# share classes, so without it the pair holds two of the five slots.
_TOP5 = {"start_date": "2021-09-16", "end_date": "2026-09-15", "universe_size": 5,
         "ema_period": 100, "max_period": 252, "sector": "ANY"}

VARIANTS["SP500 Top5 EMA Reversion"] = ("tech_ema_reversion", dict(_TOP5))
VARIANTS["SP500 Top5 EMA Reversion (one Alphabet)"] = (
    "tech_ema_reversion", {**_TOP5, "exclude": "GOOG"},
)

# 12-month momentum rotation over the 50 largest liquid names. Same data folder:
#   uv run run_local.py "Momentum Effect Top5" --data-folder data-sp500
VARIANTS["Momentum Effect Top5"] = (
    "momentum_effect",
    {
        "start_date": "2021-09-16",
        "end_date": "2026-09-15",
        "lookback": 252,
        "liquidity_filter_size": 100,
        "universe_size": 50,
        "portfolio_size": 5,
    },
)

# The local-vs-QuantConnect comparison run. Its window is deliberately not
# NVDA_WINDOW: 2020-2025 straddles 2021-03-31, where LEAN's bundled data stops
# and the converted yfinance data takes over.
VARIANTS["NVDA Monthly Rebalance 60/40"] = (
    "stock_monthly_rebalance",
    {"ticker": "NVDA", "start_date": "2020-01-01", "end_date": "2025-12-31", "weight": 0.60},
)


def slug(name):
    """Filesystem-safe identifier for a variant name."""
    cleaned = (
        name.replace("&", "and")
        .replace("%", "pct")
        .replace("±", "pm")
        .replace("/", "_")
    )
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", cleaned)
    return cleaned.strip("_").lower()
