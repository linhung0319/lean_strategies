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
