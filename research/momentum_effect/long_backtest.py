"""Re-simulate momentum_effect.py over a long window, and test stop rules on it.

LEAN runs the strategy once per data folder; this script rebuilds the same
monthly decisions in pandas (the funnel and MOMP(252) exactly as
explain_figures.py does, which matched LEAN on every rebalance) so that rule
variants can be compared quickly. Fills are at the rebalance day's open, as
LEAN's market-on-open orders are; fees are a flat 5 bp per unit of turnover.

    uv run --group data convert_data.py universe --etf SPY \\
        --start 1996-01-02 --end 2026-09-15 --out data-sp500-long
    uv run python research/momentum_effect/long_backtest.py [--data data-sp500-long]

Writes picks / monthly returns to research/momentum_effect/long/ for
long_figures.py. See long-history.md for what the numbers mean, and in
particular for how much survivorship bias the pre-2015 part carries.
"""

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "long"
LOOKBACK, LIQUID, UNIVERSE, PORTFOLIO = 252, 100, 50, 5
FEE = 0.0005


class Data:
    def __init__(self, folder):
        self.root = REPO / folder / "equity" / "usa"
        files = sorted((self.root / "universes" / "etf" / "spy").glob("*.csv"))
        self.files = pd.DatetimeIndex([pd.Timestamp(p.stem) for p in files])
        self._bars = {}

    def bars(self, ticker):
        """Raw OHLCV plus LEAN's default Adjusted factor, or None."""
        if ticker in self._bars:
            return self._bars[ticker]
        path = self.root / "daily" / f"{ticker.lower()}.zip"
        bars = None
        if path.exists():
            with zipfile.ZipFile(path) as archive:
                raw = archive.read(archive.namelist()[0]).decode()
            bars = pd.read_csv(io.StringIO(raw), header=None,
                               names=["date", "open", "high", "low", "close", "volume"])
            bars.index = pd.to_datetime(bars.pop("date").astype(str).str[:8], format="%Y%m%d")
            bars[["open", "high", "low", "close"]] /= 10000.0
            factors = pd.read_csv(self.root / "factor_files" / f"{ticker.lower()}.csv", header=None,
                                  names=["date", "price", "split", "ref"])
            factors.index = pd.to_datetime(factors["date"].astype(str), format="%Y%m%d")
            scale = (factors["price"] * factors["split"]).sort_index()
            position = scale.index.searchsorted(bars.index, side="left")
            bars["factor"] = scale.iloc[np.minimum(position, len(scale) - 1)].to_numpy()
            bars["adj"] = bars["close"] * bars["factor"]
            bars["adj_open"] = bars["open"] * bars["factor"]
            bars["momp"] = bars["adj"].pct_change(LOOKBACK, fill_method=None) * 100
            # The academic 12-2 definition: skip the most recent month.
            bars["mom122"] = bars["adj"].shift(21) / bars["adj"].shift(LOOKBACK) - 1
        self._bars[ticker] = bars
        return bars

    def constituents(self, file_date):
        path = self.root / "universes" / "etf" / "spy" / f"{file_date:%Y%m%d}.csv"
        rows = [line.split(",") for line in path.read_text().splitlines() if line.strip()]
        return {r[0]: float(r[3]) for r in rows if r[3]}, len(rows)


def last_before(bars, day):
    i = bars.index.searchsorted(day, side="left") - 1
    return bars.iloc[i] if i >= 0 else None


def select(data, day):
    """The selection at 00:00 on `day` and the 08:00 ranking that follows."""
    file_date = data.files[data.files < day][-1]
    weights, members = data.constituents(file_date)
    rows = []
    for ticker, weight in weights.items():
        bars = data.bars(ticker)
        if bars is None:
            continue
        bar = last_before(bars, day)
        if bar is None:
            continue
        dollar_volume = bar.close * bar.volume
        if bar.close > 5 and dollar_volume > 0:
            rows.append((ticker, weight, dollar_volume, bar.momp, bar.mom122))
    frame = pd.DataFrame(rows, columns=["ticker", "weight", "dv", "momp", "mom122"]).set_index("ticker")
    liquid = frame.sort_values("dv", kind="stable").iloc[-LIQUID:]
    large = liquid.sort_values("weight", ascending=False, kind="stable").iloc[:UNIVERSE]
    ranked = large["momp"].dropna().sort_values(ascending=False)
    skip = large["mom122"].dropna().sort_values(ascending=False)
    return dict(members=members, priced=len(weights), large=list(large.index),
                ranked=ranked, picks=list(ranked.index[:PORTFOLIO]),
                picks_122=list(skip.index[:PORTFOLIO]))


def trailing_returns(data, tickers, day, n=126):
    """Equal-weight daily returns of `tickers` over the n sessions before `day`."""
    closes = {}
    for t in tickers:
        b = data.bars(t)
        closes[t] = b["adj"].iloc[max(0, b.index.searchsorted(day) - n - 1):b.index.searchsorted(day)]
    return pd.DataFrame(closes).pct_change(fill_method=None).iloc[1:].mean(axis=1)


def signals(data, s, day):
    """What is known at 00:00 on `day` that a stop rule could use."""
    excess = trailing_returns(data, s["picks"], day) - trailing_returns(data, s["large"], day)
    spy = data.bars("SPY")["adj"].loc[:day - pd.Timedelta(days=1)]
    return dict(sel_xvol=excess.std() * np.sqrt(252),
                spy24=spy.iloc[-1] / spy.iloc[-505] - 1 if len(spy) > 505 else np.nan,
                spy12=spy.iloc[-1] / spy.iloc[-253] - 1 if len(spy) > 253 else np.nan,
                pick_momp=float(s["ranked"].iloc[:PORTFOLIO].mean()),
                median_momp=float(s["ranked"].median()))


def path_between(data, tickers, start, stop):
    """Daily value of 1 put equally into `tickers` at `start`'s open, held to `stop`'s open.

    A name whose data ends inside the period is held at its last price.
    """
    paths = []
    for t in tickers:
        b = data.bars(t)
        seg = b.loc[start:stop]
        if seg.empty or seg.index[0] != start:
            continue
        base = seg["adj_open"].iloc[0]
        p = seg["adj"] / base
        if seg.index[-1] == stop:  # sell at stop's open, not its close
            p.iloc[-1] = seg["adj_open"].iloc[-1] / base
        paths.append(p.rename(t))
    if not paths:
        return None
    return pd.concat(paths, axis=1)


def run(folder, start, end):
    data = Data(folder)
    spy = data.bars("SPY").loc[:end]
    days = spy.index
    month_starts = pd.Series(days, index=days).groupby(days.to_period("M")).first()
    month_starts = [d for d in month_starts if d >= pd.Timestamp(start)]
    records, daily = [], {"strategy": [], "ew50": [], "spy": [], "skip_month": []}
    for i, day in enumerate(month_starts):
        stop = month_starts[i + 1] if i + 1 < len(month_starts) else days[-1]
        s = select(data, day)
        records.append(dict(date=day, members=s["members"], priced=s["priced"],
                            picks=",".join(s["picks"]), large=",".join(s["large"]),
                            momp=",".join(f"{s['ranked'][t]:.1f}" for t in s["picks"]),
                            picks_122=",".join(s["picks_122"]), **signals(data, s, day)))
        for key, names in (("strategy", s["picks"]), ("ew50", s["large"]), ("spy", ["SPY"]),
                           ("skip_month", s["picks_122"])):
            p = path_between(data, names, day, stop)
            v = p.ffill().mean(axis=1)
            daily[key].append(v)
        print(day.date(), s["priced"], "/", s["members"], s["picks"], flush=True)
    picks = pd.DataFrame(records).set_index("date")
    return picks, daily


def chain(pieces):
    """Turn per-period paths (each starting near 1) into one daily equity curve."""
    level, out = 1.0, []
    for p in pieces:
        seg = p * level
        out.append(seg.iloc[:-1])
        level = seg.iloc[-1]
    return pd.concat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data-sp500-long")
    parser.add_argument("--start", default="1997-02-01")
    parser.add_argument("--end", default="2026-09-15")
    parser.add_argument("--tag", default="long")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    picks, daily = run(args.data, args.start, args.end)
    picks.to_csv(OUT / f"picks-{args.tag}.csv")
    # Monthly period returns (open to next month-start open) and daily paths.
    periods = pd.DataFrame({k: [p.iloc[-1] - 1 for p in v] for k, v in daily.items()},
                           index=picks.index)
    turnover = [1.0]
    rows = picks["picks"].str.split(",")
    for a, b in zip(rows.iloc[:-1], rows.iloc[1:]):
        turnover.append(len(set(b) - set(a)) / PORTFOLIO * 2)
    periods["turnover"] = turnover
    periods["strategy_net"] = periods["strategy"] - periods["turnover"] * FEE
    periods.to_csv(OUT / f"periods-{args.tag}.csv")
    curves = pd.DataFrame({k: chain(v) for k, v in daily.items()})
    curves.to_csv(OUT / f"daily-{args.tag}.csv")


if __name__ == "__main__":
    main()
