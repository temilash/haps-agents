from __future__ import annotations

from collections import Counter
from pathlib import Path

import networkx as nx
import pytest

from sim.config import MeshConfig
from sim.demand import demand_mbps
from sim.engine import Simulation
from sim.grid import haversine_km
from sim.region import load_region
from sim.state import Cell
from sim.topology import RURAL, URBAN, auto_backbone, auto_hubs, build_topology, neighbour_pairs


def _topology(region_name: str, tmp_path: Path, **mesh_overrides):
    region = load_region(region_name)
    sim = Simulation(region=region, project_root=tmp_path)
    cfg = sim.config.mesh.model_copy(update=mesh_overrides)
    frac = sim.config.demand.active_fraction_peak
    peak = {c.cell_id: sum(demand_mbps(c.population, frac, sim.config.demand)) for c in sim.cells}
    return sim.cells, peak, cfg, build_topology(sim.cells, peak, cfg, region.backbone)


@pytest.fixture(scope="module")
def england(tmp_path_factory):
    return _topology("england", tmp_path_factory.mktemp("england"))


@pytest.fixture(scope="module")
def island(tmp_path_factory):
    return _topology("test_island", tmp_path_factory.mktemp("island"))


def _degrees(pairs) -> Counter:
    deg: Counter = Counter()
    for a, b in pairs:
        deg[a] += 1
        deg[b] += 1
    return deg


def test_england_backbone_from_yaml(england):
    _cells, _peak, _cfg, topo = england
    assert len(topo.hubs) == 14
    assert topo.hubs[0].name == "London"  # biggest first
    for other in ("Manchester", "Birmingham", "Leeds"):
        assert tuple(sorted(("London", other))) in topo.backbone
    assert nx.is_connected(topo.hub_graph())


@pytest.mark.parametrize("which", ["england", "island"])
def test_every_cell_reaches_a_hub(which, request):
    cells, _peak, _cfg, topo = request.getfixturevalue(which)
    G = topo.ground_graph()
    reachable = nx.node_connected_component(G, f"pop:{topo.hubs[0].name}")
    assert all(f"net:{c.cell_id}" in reachable for c in cells)


def test_city_cells_are_within_a_hub_radius(england):
    cells, _peak, _cfg, topo = england
    by_id = {c.cell_id: c for c in cells}
    hubs = {h.name: h for h in topo.hubs}
    for cid, (hub, km) in topo.metro.items():
        h = hubs[hub]
        assert km <= h.radius_km or cid == h.cell_id
        # Nearest hub among those whose radius covers the cell.
        c = by_id[cid]
        covering = [o for o in topo.hubs if haversine_km(c.lat, c.lon, o.lat, o.lon) <= o.radius_km]
        if covering:
            assert hub == min(covering, key=lambda o: haversine_km(c.lat, c.lon, o.lat, o.lon)).name


def test_pruning_keeps_city_links_and_more_urban_than_rural(england):
    cells, _peak, _cfg, topo = england
    kept = set(topo.neighbours)
    assert len(kept) < len(topo.all_neighbours)
    city = topo.city_cells
    for a, b in topo.all_neighbours:
        if a in city or b in city:
            assert (a, b) in kept
    deg = _degrees(kept)
    non_city = [c.cell_id for c in cells if c.cell_id not in city]
    urban = [deg[c] for c in non_city if topo.density[c] == URBAN]
    rural = [deg[c] for c in non_city if topo.density[c] == RURAL]
    assert sum(urban) / len(urban) > sum(rural) / len(rural)


def test_full_mode_keeps_every_neighbour(tmp_path):
    *_rest, topo = _topology("test_island", tmp_path, mesh_mode="full")
    assert topo.neighbours == topo.all_neighbours


def test_test_island_auto_hubs_and_backbone(island):
    _cells, _peak, cfg, topo = island
    assert len(topo.hubs) == 4  # ~35,000 km2 -> clamp(round(3.5), 4, 20)
    assert [h.name for h in topo.hubs if h.radius_km == cfg.big_hub_radius_km] == [topo.hubs[0].name]
    for i, a in enumerate(topo.hubs):
        for b in topo.hubs[i + 1 :]:
            assert haversine_km(a.lat, a.lon, b.lat, b.lon) >= cfg.hub_min_separation_km
    assert nx.is_connected(topo.hub_graph())


def test_auto_backbone_is_connected_even_with_sparse_settings(england):
    cells, peak, cfg, _topo = england
    sparse = cfg.model_copy(update={"hub_count": 12, "hub_degree": 1, "big_hub_count": 0})
    hubs = auto_hubs(cells, peak, sparse)
    links = auto_backbone(hubs, sparse)
    G = nx.Graph()
    G.add_nodes_from(h.name for h in hubs)
    G.add_edges_from(links)
    assert len(hubs) == 12 and nx.is_connected(G)


def test_islands_without_a_hub_get_bridged():
    # Two separate patches of cells; only the first has a hub.
    import h3

    a = h3.latlng_to_cell(50.0, 0.0, 5)
    b = h3.latlng_to_cell(50.0, 3.0, 5)
    ids = list(h3.grid_disk(a, 1)) + list(h3.grid_disk(b, 1))
    cells = [Cell(cid, *h3.cell_to_latlng(cid), population=1000.0) for cid in ids]
    peak = {cid: (100.0 if cid == a else 1.0) for cid in ids}
    cfg = MeshConfig(hub_count=1)
    topo = build_topology(cells, peak, cfg)
    assert len(topo.bridges) == 1
    G = topo.ground_graph()
    assert nx.is_connected(G)
    assert len(neighbour_pairs(cells)) == len(topo.all_neighbours)
