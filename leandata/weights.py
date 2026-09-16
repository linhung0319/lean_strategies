"""Reconstructing ETF constituent weights from prices and share counts.

LEAN's ``ETFConstituentUniverse`` carries a ``Weight`` per holding, and a
strategy that ranks or sizes by it needs that column populated. The real
number is the fund's own -- published daily by the issuer, not by any free
price API -- so this module rebuilds a *proxy* instead:

    weight(t, i) = cap(t, i) / sum over that day's members of cap(t, .)
    cap(t, i)    = price(t, i) * shares outstanding(t, i)

For a capitalisation-weighted fund such as SPY that is the right shape. Two
differences from the published sheet are worth stating plainly, because they
are properties of the method rather than bugs:

* **Full cap, not float-adjusted.** S&P weights exclude shares held by
  insiders, governments and other strategic holders. Names with a large
  closely-held block therefore come out heavier here than in the fund.
* **Filings-dated share counts.** Yahoo reports shares outstanding a few
  times a year, so a buyback is applied on the date it was *reported*, not
  the date it happened.

Both move a constituent by a fraction of its weight, which is immaterial to
which names land in the top of a ranking and material if you quote the weight
itself as the fund's.

The one subtlety that is *not* cosmetic is splits. A share count filed before
a split, carried forward past it, is stated in pre-split shares while the
price beside it is post-split -- multiply the two and the company loses the
split factor's worth of market cap until the next filing. So both series are
restated into today's share terms before they meet, where the split factor
cancels:

    cap = (raw price / cum) * (raw shares * cum)

with ``cum(t)`` the product of every split ratio whose ex-date is after ``t``
-- the same quantity ``sources.yfinance_source`` uses to un-adjust Yahoo's
feed, and the reason this module needs the splits at all.
"""

from datetime import date
from typing import Mapping

import numpy as np
import pandas as pd

from .errors import ValidationError
from .model import (
    CONSTITUENT_COLUMNS,
    ConstituentPanel,
    IndexMembership,
    Provenance,
    SecurityHistory,
    SymbolSpec,
)


def cumulative_split_factor(index: pd.DatetimeIndex, splits: pd.DataFrame) -> np.ndarray:
    """How many of today's shares one share held at each timestamp became.

    1.0 everywhere when there are no splits. Ratios ex-dated *at* a timestamp
    are excluded: a price or share count dated on the ex-date is already
    quoted in post-split terms.
    """
    values = np.ones(len(index), dtype="float64")
    if splits is None or len(splits) == 0:
        return values
    ratios = splits["ratio"].sort_index()
    # suffix[k] is the product of ratios[k:], so indexing it by "how many
    # split days fall at or before t" gives cum(t) in one pass.
    suffix = np.append(np.cumprod(ratios.to_numpy()[::-1])[::-1], 1.0)
    positions = np.searchsorted(ratios.index.to_numpy(), index.to_numpy(), side="right")
    return suffix[positions]


def restated_market_caps(
    histories: Mapping[str, SecurityHistory],
    shares: Mapping[str, pd.Series],
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """A ``calendar`` x ticker frame of market capitalisation, NaN where unknown.

    ``histories`` and ``shares`` are keyed alike; a ticker missing from either,
    or with no share count at all, comes back as an all-NaN column rather than
    being dropped, so the caller can report on it.
    """
    columns = {}
    for ticker, history in histories.items():
        close = history.bars["close"].reindex(calendar)
        split_factor = cumulative_split_factor(calendar, history.splits)
        adjusted_close = close.to_numpy(dtype="float64") / split_factor

        reported = shares.get(ticker)
        if reported is None or len(reported) == 0:
            columns[ticker] = pd.Series(np.nan, index=calendar, dtype="float64")
            continue
        reported_factor = cumulative_split_factor(pd.DatetimeIndex(reported.index), history.splits)
        restated = pd.Series(
            reported.to_numpy(dtype="float64") * reported_factor, index=reported.index
        )
        # Back-fill as well as forward-fill: the window almost always opens
        # before the first filing inside it, and the alternative is to throw
        # away the leading months of every ticker.
        aligned = restated.reindex(restated.index.union(calendar)).ffill().bfill().reindex(calendar)

        columns[ticker] = pd.Series(
            adjusted_close * aligned.to_numpy(dtype="float64"), index=calendar, dtype="float64"
        )

    frame = pd.DataFrame(columns, index=calendar)
    frame.index.name = "date"
    return frame


def cap_weighted_panel(
    etf: SymbolSpec,
    membership: IndexMembership,
    caps: pd.DataFrame,
    *,
    rename: Mapping[str, str] | None = None,
    provenance: Provenance | None = None,
) -> ConstituentPanel:
    """Normalise ``caps`` within each day's roster into a ConstituentPanel.

    ``membership`` decides who is in the file on a given day; ``caps`` decides
    what their weight is. A member with no market cap is still written -- with
    an empty weight, which is how LEAN spells "not known" -- so that the file
    keeps saying who was in the index even where the proxy could not price
    them. ``rename`` maps index tickers onto the spelling the data folder uses
    (see :mod:`leandata.tickers`).
    """
    if not isinstance(caps.index, pd.DatetimeIndex):
        raise ValidationError("caps must be indexed by a DatetimeIndex of trading days")
    rename = rename or {}

    index_pairs: list[tuple[pd.Timestamp, str]] = []
    weights: list[float] = []

    for timestamp in caps.index:
        members = membership.members_on(timestamp.date())
        if not members:
            continue
        priced = [name for name in members if name in caps.columns]
        row = caps.loc[timestamp, priced] if priced else pd.Series(dtype="float64")
        values = row.to_numpy(dtype="float64")
        known = np.isfinite(values) & (values > 0)
        total = values[known].sum()

        for name in members:
            index_pairs.append((timestamp, rename.get(name, name)))
            if name in row.index and total > 0:
                cap = float(row[name])
                weights.append(cap / total if np.isfinite(cap) and cap > 0 else np.nan)
            else:
                weights.append(np.nan)

    if not index_pairs:
        raise ValidationError(
            "no constituent rows were produced; the calendar and the membership "
            "window probably do not overlap"
        )

    rows = pd.DataFrame(
        {
            "weight": weights,
            "shares_held": np.nan,
            "market_value": np.nan,
        },
        index=pd.MultiIndex.from_tuples(index_pairs, names=["date", "ticker"]),
    )[list(CONSTITUENT_COLUMNS)]
    # A ticker can be renamed onto one that is already in the roster; keep the
    # heavier row rather than letting the panel fail its uniqueness check.
    rows = rows.groupby(level=["date", "ticker"]).max()
    return ConstituentPanel(etf=etf, rows=rows, provenance=provenance)


def trading_calendar(
    history: SecurityHistory, start: date | None = None, end: date | None = None
) -> pd.DatetimeIndex:
    """The days a universe file should exist for, taken from the ETF's own bars."""
    index = history.bars.index
    if start is not None:
        index = index[index >= pd.Timestamp(start)]
    if end is not None:
        index = index[index <= pd.Timestamp(end)]
    return pd.DatetimeIndex(index, name="date")
