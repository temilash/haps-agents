from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sim.engine import Simulation
from sim.grid import haversine_km
from sim.state import Cell


def _tiny_config(**overrides) -> dict:
    data = {
        "grid": {"h3_resolution": 5},
        "demand": {
            "trough_hour": 4.0,
            "peak_hour": 20.0,
            "active_fraction_trough": 0.02,
            "active_fraction_peak": 0.15,
            "critical_user_share": 0.05,
            "critical_mbps_per_user": 0.1,
            "general_user_share": 0.95,
            "general_mbps_per_user": 2.0,
            "ground_overprovision": 1.3,
        },
        "network": {
            "haps_coverage_km": 50,
            "haps_access_mbps": 5000,
            "isl_range_km": 250,
            "isl_mbps": 10000,
            "feeder_range_km": 150,
            "feeder_mbps": 10000,
        },
        "latency": {
            "ground_latency_ms": 20,
            "haps_base_ms": 20,
            "per_haps_ms": 3,
            "speed_km_per_ms": 300,
        },
        "haps": [{"id": "H1", "lat": 53.48, "lon": -2.24}],
        "gateways": [{"id": "GW-1", "lat": 53.48, "lon": -2.24}],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    return data


def _sim(tmp_path: Path, cells: list[Cell], cfg: dict | None = None) -> Simulation:
    payload = cfg if cfg is not None else _tiny_config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return Simulation(config_path=path, cells=cells, project_root=tmp_path)


def test_haversine_sanity():
    deg_lat = haversine_km(0.0, 0.0, 1.0, 0.0)
    deg_lon_eq = haversine_km(0.0, 0.0, 0.0, 1.0)
    assert abs(deg_lat - 111.195) < 0.5
    assert abs(deg_lon_eq - 111.195) < 0.5
    london_manchester = haversine_km(51.5074, -0.1278, 53.4808, -2.2426)
    assert 250 < london_manchester < 280


def test_no_overflow_when_towers_are_up(tmp_path: Path):
    cells = [Cell("c1", 53.48, -2.24, population=80_000)]
    sim = _sim(tmp_path, cells)
    result = sim.step("20:00")
    assert result.totals.unserved_critical_mbps == pytest.approx(0.0, abs=1e-6)
    assert result.totals.unserved_general_mbps == pytest.approx(0.0, abs=1e-6)
    assert result.totals.pct_served == pytest.approx(100.0, abs=1e-6)
    assert result.cells[0].haps_served_critical_mbps == pytest.approx(0.0, abs=1e-6)
    assert result.cells[0].haps_served_general_mbps == pytest.approx(0.0, abs=1e-6)


def test_critical_served_before_general(tmp_path: Path):
    cells = [Cell("c1", 53.48, -2.24, population=100_000, tower_status=0.0)]
    cfg = _tiny_config(network={"haps_access_mbps": 50.0})
    sim = _sim(tmp_path, cells, cfg)
    result = sim.step("20:00")
    cell = result.cells[0]
    haps_total = cell.haps_served_critical_mbps + cell.haps_served_general_mbps
    assert haps_total == pytest.approx(50.0, abs=1e-4)
    assert cell.haps_served_critical_mbps == pytest.approx(50.0, abs=1e-4)
    assert cell.haps_served_general_mbps == pytest.approx(0.0, abs=1e-4)
    assert cell.unserved_critical_mbps > 0
    assert cell.unserved_general_mbps > cell.unserved_critical_mbps


def test_no_flow_exceeds_capacity(tmp_path: Path):
    cells = [
        Cell("c1", 53.48, -2.24, population=500_000, tower_status=0.0),
        Cell("c2", 54.89, -2.93, population=80_000, tower_status=0.0),
    ]
    cfg = _tiny_config(
        haps=[
            {"id": "H1", "lat": 53.48, "lon": -2.24},
            {"id": "H4", "lat": 54.89, "lon": -2.93},
        ],
        gateways=[
            {"id": "GW-1", "lat": 53.48, "lon": -2.24},
            {"id": "GW-2", "lat": 53.76, "lon": -2.70},
        ],
        network={"haps_access_mbps": 200.0, "isl_mbps": 150.0, "feeder_mbps": 180.0},
    )
    sim = _sim(tmp_path, cells, cfg)
    result = sim.step("20:00")
    for link in result.links:
        assert link.flow_mbps <= link.capacity_mbps + 1e-6


def test_no_gateways_means_no_haps_service(tmp_path: Path):
    cells = [Cell("c1", 53.48, -2.24, population=80_000, tower_status=0.0)]
    cfg = _tiny_config(gateways=[])
    sim = _sim(tmp_path, cells, cfg)
    result = sim.step("20:00")
    cell = result.cells[0]
    assert cell.haps_served_critical_mbps == pytest.approx(0.0, abs=1e-9)
    assert cell.haps_served_general_mbps == pytest.approx(0.0, abs=1e-9)
    assert cell.unserved_critical_mbps == pytest.approx(cell.demand_critical_mbps)
    assert cell.unserved_general_mbps == pytest.approx(cell.demand_general_mbps)


def test_relay_haps_has_higher_latency(tmp_path: Path):
    cells = [
        Cell("near-direct", 53.0, 0.0, population=50_000, tower_status=0.0),
        Cell("near-relay", 54.5, 0.0, population=50_000, tower_status=0.0),
    ]
    cfg = _tiny_config(
        haps=[
            {"id": "H-direct", "lat": 53.0, "lon": 0.0},
            {"id": "H-relay", "lat": 54.5, "lon": 0.0},
        ],
        gateways=[{"id": "GW-1", "lat": 53.0, "lon": 0.0}],
    )
    sim = _sim(tmp_path, cells, cfg)
    dist_relay = haversine_km(54.5, 0.0, 53.0, 0.0)
    assert dist_relay > cfg["network"]["feeder_range_km"]
    assert dist_relay < cfg["network"]["isl_range_km"]

    direct = sim.routes["H-direct"]
    relay = sim.routes["H-relay"]
    assert direct is not None and relay is not None
    assert relay.latency_ms > direct.latency_ms
    assert relay.hops > direct.hops

    result = sim.step("20:00")
    by_id = {c.cell_id: c for c in result.cells}
    assert by_id["near-relay"].latency_ms > by_id["near-direct"].latency_ms
