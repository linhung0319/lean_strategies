"""Number formatting that reproduces LEAN's on-disk text byte for byte.

LEAN writes its data files from C# ``decimal`` values, and two pieces of
.NET semantics leak into the bytes:

``Math.Round(value, n)``
    Banker's rounding, and -- crucially -- it only ever *reduces* the number
    of decimal places. Rounding a decimal that already has fewer places than
    ``n`` leaves it untouched rather than padding with zeros. That is why
    ``Data/equity/usa/factor_files/aapl.csv`` prints a split factor of
    ``0.25`` as ``0.25`` but ``0.0357143`` as ``0.03571430``: the first has
    scale 2 and stays there, the second came out of a division with scale
    well past 8 and got cut down to exactly 8.

``Extensions.Normalize()``
    ``input / 1.000000000000000000000000000000000m``, which strips trailing
    zeros. Used for reference prices and for scaled bar prices.

Python's ``Decimal`` carries the same significant-digits-and-exponent model,
so both behaviours port exactly. Floats do not -- always come in through
``to_decimal``, never ``Decimal(some_float)``.
"""

from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final

# LEAN stores equity prices in deci-cents: see LeanData.Scale in
# Lean/Common/Util/LeanData.cs and TradeBar._scaleFactor.
PRICE_SCALE_FACTOR: Final = Decimal(10_000)
# Sub-cent price resolution kept before scaling. Four places is what LEAN's
# own writers round to and what keeps every bundled row an integer.
DEFAULT_PRICE_PLACES: Final = 4


def to_decimal(value) -> Decimal:
    """Convert to Decimal via its shortest string form.

    ``Decimal(0.1)`` is ``0.1000000000000000055511151231257827``; going
    through ``str`` gives ``0.1``. Seventeen digits of binary noise would
    otherwise propagate into every price factor.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def net_round(value: Decimal, places: int) -> Decimal:
    """Emulate C# ``Math.Round(decimal, int)``.

    Rounds half to even, and never increases the number of decimal places.
    """
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int) or -exponent <= places:
        return value
    return value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)


def to_str(value: Decimal) -> str:
    """Render a Decimal in plain notation, preserving its scale."""
    return format(value, "f")


def strip_zeros(value: Decimal) -> str:
    """Emulate ``Extensions.NormalizeToStr``: drop trailing zeros, no exponent.

    ``Decimal("973100.00").normalize()`` is ``9.731E+5``; only the ``f``
    presentation type expands it back. Getting this wrong writes scientific
    notation into the data files, which LEAN parses as garbage.
    """
    if value == 0:
        # normalize() turns Decimal("0.00") into Decimal("0"), but be explicit.
        return "0"
    return format(value.normalize(), "f")


def scale_price(value, *, places: int = DEFAULT_PRICE_PLACES) -> str:
    """Format a raw price as LEAN's deci-cent integer field.

    Quantising to ``places`` first is what turns a float like
    ``97.31000137329102`` back into the ``973100`` the bundled files carry,
    and keeps genuine sub-penny data (pre-decimalisation sixteenths, for
    instance) intact.
    """
    quantized = to_decimal(value).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
    return strip_zeros(quantized * PRICE_SCALE_FACTOR)


def format_volume(value) -> str:
    """Format a volume field. Volume is not scaled, and LEAN writes integers."""
    return strip_zeros(to_decimal(value).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
