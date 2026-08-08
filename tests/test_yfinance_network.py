"""The one test that actually calls Yahoo. Deselected by default.

    uv run --group data pytest -m network

Everything else runs against fixtures, so this exists only to catch yfinance
changing its response shape under us -- a renamed column or a dropped
actions field would break the adapter silently otherwise.
"""

from datetime import date, timedelta

import pytest

from leandata.model import Resolution
from leandata.sources import get_source
from leandata.sources.base import FetchRequest

pytestmark = pytest.mark.network


def test_a_live_fetch_satisfies_the_canonical_contract():
    yfinance = pytest.importorskip("yfinance")
    assert yfinance  # the import is the point

    source = get_source("yfinance")
    end = date.today() - timedelta(days=1)
    history = source.fetch(
        FetchRequest(ticker="SPY", resolution=Resolution.DAILY, start=end - timedelta(days=14), end=end)
    )

    # SecurityHistory validates on construction, so getting here already
    # proves the schema, index and price invariants hold.
    assert len(history.bars) >= 5
    assert history.bars.index.tz is None
    assert history.symbol.ticker == "SPY"
    assert history.provenance.source == "yfinance"
    assert (history.bars["close"] > 0).all()
