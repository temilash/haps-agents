from __future__ import annotations

import pytest

from sim.endpoints import endpoint_capacities, endpoint_weights
from sim.state import Cell


def _cells(*pops: float) -> list[Cell]:
    return [Cell(f"c{i}", 0.0, 0.0, population=p) for i, p in enumerate(pops)]


def test_weights_normalised_and_superlinear():
    w = endpoint_weights(_cells(100, 400, 0), exponent=1.5)
    assert sum(w.values()) == pytest.approx(1.0)
    # 4x the population -> 8x the weight with exponent 1.5.
    assert w["c1"] / w["c0"] == pytest.approx(8.0)
    assert w["c2"] == 0.0


def test_exponent_one_is_population_share():
    w = endpoint_weights(_cells(1, 3), exponent=1.0)
    assert w == pytest.approx({"c0": 0.25, "c1": 0.75})


def test_no_population_spreads_evenly():
    assert endpoint_weights(_cells(0, 0), 1.5) == {"c0": 0.5, "c1": 0.5}


def test_capacities_follow_tick_demand():
    w = endpoint_weights(_cells(100, 400), 1.5)
    caps = endpoint_capacities(w, total_demand_mbps=1000.0, headroom=1.2)
    assert sum(caps.values()) == pytest.approx(1200.0)
    assert caps["c1"] == pytest.approx(1200.0 * w["c1"])
