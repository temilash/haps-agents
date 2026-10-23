from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx
import pytest
from shapely.geometry import Point

from sim.controllers import make_controller
from sim.engine import Simulation
from sim.grid import load_boundary
from sim.region import DEFAULT_REGION, load_region, region_config
from sim.runner import run

COUNTRIES = ("England", "Scotland", "Wales", "Northern Ireland")


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("uk")


@pytest.fixture(scope="module")
def uk(root) -> Simulation:
    return Simulation(region="uk", project_root=root)


def _sim(root: Path, haps: int | None = None) -> Simulation:
    cfg = region_config(load_region("uk"))
    if haps is not None:
        cfg.fleet.count = haps
    return Simulation(config=cfg, region="uk", project_root=root)


def _country_cells(sim: Simulation) -> dict[str, list[str]]:
    region = sim.region
    out: dict[str, list[str]] = {}
    for name in COUNTRIES:
        shape = load_boundary(region.boundary_path, name, "CTRY23NM")
        out[name] = [c.cell_id for c in sim.cells if shape.contains(Point(c.lon, c.lat))]
    return out


def test_uk_is_the_default_region():
    assert DEFAULT_REGION == "uk"


def test_cells_in_all_four_countries(uk):
    by_country = _country_cells(uk)
    for name in COUNTRIES:
        assert by_country[name], name
    assert sum(len(v) for v in by_country.values()) == len(uk.cells)


def test_small_landmasses_dropped(uk):
    pop = {c.cell_id: c.population for c in uk.cells}
    totals = defaultdict(float)
    for cid, mass in uk.topology.landmass.items():
        totals[mass] += pop[cid]
    assert min(totals.values()) >= uk.region.spec.min_component_population


def test_every_landmass_reaches_a_hub(uk):
    G = uk.ground_graph()
    reachable = nx.node_connected_component(G, "pop:London")
    assert all(f"net:{c.cell_id}" in reachable for c in uk.cells)
    assert len(set(uk.topology.landmass.values())) > 2  # islands as well as GB and NI


def test_northern_ireland_reaches_gb_hubs_via_subsea(uk):
    assert {"Belfast|Glasgow", "Belfast|Liverpool"} <= set(uk.subsea_links())
    ni = _country_cells(uk)["Northern Ireland"]
    G = uk.ground_graph()
    path = nx.shortest_path(G, f"net:{ni[0]}", "pop:London")
    types = {G.edges[a, b]["type"] for a, b in zip(path, path[1:])}
    assert "subsea" in types


def test_cutting_subsea_isolates_northern_ireland(root):
    sim = _sim(root)
    ni = _country_cells(sim)["Northern Ireland"]
    sim.set_subsea_cut("all")
    G = sim.ground_graph()
    for cid in ni:
        assert nx.has_path(G, f"net:{cid}", "pop:Belfast")  # still reaches its own hub
        assert not nx.has_path(G, f"net:{cid}", "pop:London")
    sim.set_subsea_cut(["Belfast|Glasgow"])  # one cable left
    assert nx.has_path(sim.ground_graph(), f"net:{ni[0]}", "pop:London")
    sim.set_subsea_cut(None)
    assert nx.has_path(sim.ground_graph(), f"net:{ni[0]}", "pop:London")
    with pytest.raises(ValueError, match="unknown subsea"):
        sim.set_subsea_cut(["Belfast|Paris"])


@pytest.mark.parametrize("n", [0, 1, 10])
def test_fleet_size_runs(root, n):
    sim = _sim(root, haps=n)
    assert [h.id for h in sim.haps] == [f"H{i}" for i in range(1, n + 1)]
    assert all(route is not None for route in sim.routes.values())
    results = run(sim, make_controller("greedy", sim.config.greedy), hours=1)
    assert len(results) == 6 and all(len(r.haps) == n for r in results)


def test_auto_placed_haps_have_a_gateway_route(uk):
    assert len(uk.haps) == uk.config.fleet.count == 6
    assert all(route is not None for route in uk.routes.values())


def test_more_haps_never_reduce_daily_service(root):
    served = []
    for n in (0, 2, 6, 10):
        sim = _sim(root, haps=n)
        results = run(sim, make_controller("greedy", sim.config.greedy))
        demand = sum(r.totals.demand_mbps for r in results)
        unserved = sum(r.totals.unserved_critical_mbps + r.totals.unserved_general_mbps for r in results)
        served.append(100.0 * (demand - unserved) / demand)
    for fewer, more in zip(served, served[1:]):
        assert more >= fewer - 0.01, served


def test_england_unchanged(tmp_path):
    sim = Simulation(region="england", project_root=tmp_path)
    assert len(sim.cells) == 571
    assert [(h.id, h.lat, h.lon) for h in sim.haps] == [
        ("H1", 53.4808, -2.2426),
        ("H2", 53.8008, -1.5491),
        ("H3", 54.9783, -1.6178),
        ("H4", 54.895, -2.932),
    ]
    assert sim.subsea_links() == []
    assert len(sim.topology.hubs) == 14
