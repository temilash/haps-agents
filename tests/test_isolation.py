from __future__ import annotations

from pathlib import Path

import pytest
from shapely.geometry import Point

from sim.config import AppConfig
from sim.engine import Simulation
from sim.grid import haversine_km, load_boundary
from sim.region import load_region, region_config

BELFAST = (54.597, -5.930)
STRANRAER = (55.0, -5.0)  # GB side of the North Channel


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("iso")


def _sim(root: Path, proxy: bool = True) -> Simulation:
    base = region_config(load_region("uk")).model_dump()
    cfg = AppConfig.model_validate(
        {
            **base,
            "isolation_proxy": proxy,
            "fleet": {"count": None},
            "haps": [
                {"id": "H1", "lat": BELFAST[0], "lon": BELFAST[1]},
                {"id": "H2", "lat": STRANRAER[0], "lon": STRANRAER[1]},
            ],
        }
    )
    return Simulation(config=cfg, region="uk", project_root=root)


def _ni_cells(sim: Simulation) -> set[str]:
    ni = load_boundary(sim.region.boundary_path, "Northern Ireland", "CTRY23NM")
    return {c.cell_id for c in sim.cells if ni.contains(Point(c.lon, c.lat))}


def _feeder_flow(result, gateway_id: str) -> float:
    return sum(l.flow_mbps for l in result.links if l.link_type == "feeder" and l.to_id == gateway_id)


def test_geometry_of_the_scenario():
    gw = {g["id"]: g for g in load_region("uk").assets["gateways"]}
    glasgow = (gw["GW-Glasgow"]["lat"], gw["GW-Glasgow"]["lon"])
    assert haversine_km(*BELFAST, *glasgow) > 150  # H1 can't feed Glasgow directly
    assert haversine_km(*STRANRAER, *glasgow) <= 150  # H2 can
    assert haversine_km(*BELFAST, *STRANRAER) <= 250  # ISL across the North Channel


def test_subsea_cut_isolates_northern_ireland(root):
    sim = _sim(root)
    ni = _ni_cells(sim)
    before = sim.step("20:00")
    assert before.totals.isolated_cells == 0
    ni_ground_before = sum(c.ground_served_mbps for c in before.cells if c.cell_id in ni)

    sim.set_subsea_cut("all")
    after = sim.step("20:00")
    by_id = {c.cell_id: c for c in after.cells}
    assert all(by_id[cid].isolated for cid in ni)
    assert not by_id[sim.topology.hubs[0].cell_id].isolated  # London

    # Ground keeps at most local_traffic_share of demand in NI.
    share = sim.config.local_traffic_share
    for cid in ni:
        c = by_id[cid]
        assert c.ground_served_mbps <= share * (c.demand_critical_mbps + c.demand_general_mbps) + 1e-6
    ni_ground_after = sum(by_id[cid].ground_served_mbps for cid in ni)
    assert ni_ground_after < 0.5 * ni_ground_before
    ni_served = lambda r: sum(  # noqa: E731
        c.ground_served_mbps + c.haps_served_critical_mbps + c.haps_served_general_mbps
        for c in r.cells if c.cell_id in ni
    )
    assert ni_served(after) < ni_served(before)

    # Belfast's gateway is unusable; NI traffic reaches GB over the ISL.
    assert _feeder_flow(after, "GW-Belfast") == 0.0
    assert sum(by_id[cid].haps_served_critical_mbps + by_id[cid].haps_served_general_mbps for cid in ni) > 0
    isl = {(l.from_id, l.to_id): l.flow_mbps for l in after.links if l.link_type == "isl"}
    assert isl.get(("H1", "H2"), 0.0) > 0
    assert _feeder_flow(after, "GW-Glasgow") > 0

    # Repairing the cables restores everything.
    sim.set_subsea_cut(None)
    again = sim.step("20:00")
    assert again.totals.isolated_cells == 0
    assert sum(c.ground_served_mbps for c in again.cells if c.cell_id in ni) == pytest.approx(ni_ground_before)


def test_proxy_can_be_switched_off(root):
    sim = _sim(root, proxy=False)
    sim.set_subsea_cut("all")
    result = sim.step("20:00")
    assert result.totals.isolated_cells == 0
    assert not any(c.isolated for c in result.cells)


def test_gateway_in_isolated_area_is_disabled_until_repaired(root):
    sim = _sim(root)
    sim.set_subsea_cut("all")
    sim.step("20:00")
    assert "GW-Belfast" not in {g.id for g in sim._active_gateways()}
    sim.set_subsea_cut(None)
    sim.step("20:00")
    assert "GW-Belfast" in {g.id for g in sim._active_gateways()}
