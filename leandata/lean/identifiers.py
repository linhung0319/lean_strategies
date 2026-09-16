"""LEAN ``SecurityIdentifier`` strings, the second column of a universe file.

A universe file names each constituent by its SID rather than its ticker,
because a ticker alone is ambiguous across renames. LEAN builds one with
``SecurityIdentifier.GenerateEquity(date, symbol, market)``
(``Lean/Common/SecurityIdentifier.cs:490``), which packs the fields into a
single ``ulong`` and renders it as ``"<TICKER> <base36 of the ulong>"``.

Only the equity layout is implemented, and for equities most of that ulong is
zero::

    {put/call:1}{oa-date:5}{style:1}{strike:6}{strike-scale:2}{market:3}{security-type:2}

leaving ``oa_date * 10^14 + market * 100 + 1``. ``oa_date`` is the OLE
automation day number of the security's *first* date -- in LEAN, the first row
of its map file (``SecurityIdentifier.cs:451-461``), which is what
``leandata.lean.mapfiles`` writes as the first bar's date.

Nothing here reads or writes files; it is pure arithmetic, and
``tests/test_identifiers.py`` pins it against SIDs taken from LEAN's own
bundled data.
"""

from datetime import date
from typing import Final

from ..errors import ValidationError

# Lean/Common/SecurityIdentifier.cs:80-100. Only the two that are non-zero for
# an equity are named; the rest multiply zero.
_SECURITY_TYPE_OFFSET: Final = 1
_MARKET_OFFSET: Final = 100
_DAYS_OFFSET: Final = 10**14

# SecurityType.Equity, Lean/Common/Global.cs.
_EQUITY: Final = 1
# Market.Encode, Lean/Common/Market.cs:35. Only "usa" is wired up here --
# every other market would also need its own map and market-hours entries.
_MARKET_IDENTIFIERS: Final = {"usa": 1}

# DateTime.ToOADate counts days from this epoch.
_OA_EPOCH: Final = date(1899, 12, 30)

# Lean/Common/SecurityIdentifier.cs:36 -- the ticker is capped at 12 characters.
_MAX_TICKER_LENGTH: Final = 12
# Lean/Common/SecurityIdentifier.cs:44.
_INVALID_TICKER_CHARACTERS: Final = frozenset("| ")


def encode_base36(value: int) -> str:
    """``Extensions.EncodeBase36``: digits then upper case letters, no padding."""
    if value < 0:
        raise ValidationError(f"base36 encoding is unsigned, got {value}")
    if value == 0:
        return "0"
    digits = []
    while value:
        value, remainder = divmod(value, 36)
        digits.append("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[remainder])
    return "".join(reversed(digits))


def equity_security_id(ticker: str, first_date: date, market: str = "usa") -> str:
    """The SID string for an equity that first traded under ``ticker`` on ``first_date``.

    ``first_date`` is the *first* date of the map file, not the date being
    written -- a SID is constant for the life of the security.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValidationError("equity_security_id needs a non-empty ticker")
    if len(ticker) > _MAX_TICKER_LENGTH:
        raise ValidationError(
            f"ticker {ticker!r} is longer than LEAN's {_MAX_TICKER_LENGTH}-character limit"
        )
    invalid = sorted(_INVALID_TICKER_CHARACTERS.intersection(ticker))
    if invalid:
        raise ValidationError(f"ticker {ticker!r} contains invalid character(s) {invalid}")
    try:
        market_identifier = _MARKET_IDENTIFIERS[market]
    except KeyError:
        known = ", ".join(sorted(_MARKET_IDENTIFIERS))
        raise ValidationError(f"unknown market {market!r}; known: {known}") from None

    oa_date = (first_date - _OA_EPOCH).days
    if oa_date < 0:
        raise ValidationError(f"first_date {first_date} predates the OLE automation epoch")

    properties = (
        oa_date * _DAYS_OFFSET
        + market_identifier * _MARKET_OFFSET
        + _EQUITY * _SECURITY_TYPE_OFFSET
    )
    return f"{ticker} {encode_base36(properties)}"
