from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from sim.config import AppConfig
from sim.engine import Simulation
from sim.region import load_region, region_config

GOLDEN = Path(__file__).parent / "data" / "simple_golden.json.gz"
RURAL = (54.45, -2.10)
STEVENAGE = (51.90, -0.20)


def _config(model: str | None) -> AppConfig:
    cfg = region_config(load_region("england"))
    # The flat mesh and the golden snapshot use the original single demand curve.
    cfg.demand_model = "single"
    if model is not None:
        cfg.ground_model = model
    return cfg


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    """Shared root, so backhaul calibration is cached across this module."""
    return tmp_path_factory.mktemp("mesh")


def _sim(project: Path, model: str | None = "mesh", outage=None) -> Simulation:
    cfg = _config(model)
    sim = Simulation(config=cfg, region="england", project_root=project)
    if outage is not None:
        sim.set_tower_status(*outage, 15.0, 0.1)
    return sim


@pytest.fixture(scope="module")
def normal(project):
    return _sim(project).step("20:00")


@pytest.fixture(scope="module")
def rural(project):
    return _sim(project, outage=RURAL).step("20:00")


@pytest.fixture(scope="module")
def stevenage(project):
    return _sim(project, outage=STEVENAGE).step("20:00")


def test_normal_all_served_no_backhaul_over_capacity(normal):
    t = normal.totals
    assert t.unserved_critical_mbps == pytest.approx(0.0, abs=1e-3)
    assert t.unserved_general_mbps == pytest.approx(0.0, abs=1e-3)
    assert t.pct_served == pytest.approx(100.0, abs=1e-6)
    backhaul = [link for link in normal.links if link.link_type == "backhaul"]
    assert backhaul
    assert all(link.utilisation <= 1.0 + 1e-9 for link in backhaul)
    assert t.backhaul_links_over_80pct == 0
    assert all(c.haps_served_critical_mbps + c.haps_served_general_mbps == 0 for c in normal.cells)


def test_transit_is_happening(normal):
    assert normal.totals.pct_crossing_backhaul > 30.0
    assert sum(c.transit_mbps for c in normal.cells) > 0


def test_corridor_outage_congests_more_than_rural(rural, stevenage):
    assert (
        stevenage.totals.backhaul_links_over_80pct > rural.totals.backhaul_links_over_80pct
    )


def test_flow_never_exceeds_capacity(normal, rural, stevenage, project):
    carlisle = _sim(project)
    carlisle.set_tower_status(54.89, -2.93, 30.0, 0.1)
    haps_tick = carlisle.step("20:00")
    assert sum(c.haps_served_general_mbps for c in haps_tick.cells) > 0
    for result in (normal, rural, stevenage, haps_tick):
        for link in result.links:
            assert link.flow_mbps <= link.capacity_mbps + 1e-6, link
        for c in result.cells:
            served = (
                c.ground_served_mbps + c.haps_served_critical_mbps + c.haps_served_general_mbps
            )
            assert served <= c.demand_critical_mbps + c.demand_general_mbps + 1e-6


def test_backhaul_calibration_is_cached(project, normal):
    path = project / "data/processed/england/backhaul.parquet"
    assert path.exists()
    mtime = path.stat().st_mtime_ns
    _sim(project)
    assert path.stat().st_mtime_ns == mtime
    cfg = _config("mesh")
    cfg.mesh.backhaul_min_mbps = 700
    Simulation(config=cfg, region="england", project_root=project)
    assert path.stat().st_mtime_ns != mtime
    # Leave the default calibration in place for the rest of the module.
    _sim(project)


def _assert_matches(expected, actual, where: str = "") -> None:
    if isinstance(expected, dict):
        for key, value in expected.items():
            _assert_matches(value, actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), where
        for i, (e, a) in enumerate(zip(expected, actual)):
            _assert_matches(e, a, f"{where}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), (where, expected, actual)
    else:
        assert actual == expected, (where, expected, actual)


# How max-flow splits one HAPS among the cells it covers isn't unique and varies
# with PYTHONHASHSEED (networkx iterates sets), so these per-cell fields are
# compared as totals over cells.
SPLIT_FIELDS = (
    "haps_served_critical_mbps",
    "haps_served_general_mbps",
    "unserved_critical_mbps",
    "unserved_general_mbps",
    "latency_ms",
    "hops",
)


def _split_invariant(tick: dict) -> dict:
    cells = [{k: v for k, v in c.items() if k not in SPLIT_FIELDS} for c in tick["cells"]]
    sums = {k: sum(c[k] for c in tick["cells"]) for k in SPLIT_FIELDS[:4]}
    return {**tick, "cells": cells, "cell_sums": sums}


@pytest.mark.parametrize("model", [None, "simple"])
def test_simple_model_unchanged(tmp_path, model):
    """Simple-model output recorded before the mesh model was added."""
    with gzip.open(GOLDEN, "rt") as f:
        golden = json.load(f)
    sim = _sim(tmp_path, model)
    actual = {"12:00": sim.step("12:00").to_dict(), "20:00": sim.step("20:00").to_dict()}
    sim.set_tower_status(54.89, -2.93, 30.0, 0.1)
    actual["20:00 carlisle"] = sim.step("20:00").to_dict()
    for key in golden:
        _assert_matches(_split_invariant(golden[key]), _split_invariant(actual[key]), key)
    assert not (tmp_path / "data/processed/england/backhaul.parquet").exists()
