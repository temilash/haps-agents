"""Ground network shape: city hubs, metro and backbone links, pruned neighbour mesh."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import h3
import networkx as nx
import numpy as np

from sim.config import MeshConfig
from sim.grid import haversine_km
from sim.state import Cell

Pair = tuple[str, str]
URBAN, SUBURBAN, RURAL = "urban", "suburban", "rural"


@dataclass
class Hub:
    name: str
    lat: float
    lon: float
    radius_km: float
    cell_id: str  # the cell the hub sits in (nearest cell centre)


@dataclass
class Topology:
    hubs: list[Hub]
    backbone: dict[Pair, float]  # (hub, hub) sorted -> km
    metro: dict[str, tuple[str, float]]  # city cell -> (hub, km)
    all_neighbours: dict[Pair, float]  # every H3 neighbour pair (a < b) -> km
    neighbours: dict[Pair, float]  # the kept ones
    density: dict[str, str]  # cell -> urban / suburban / rural
    hub_distance_km: dict[str, float]  # neighbour-path km to the nearest city cell
    # Subsea links between landmasses, keyed by node pair ("pop:{hub}" or
    # "net:{cell}"): hub-to-hub ones from backbone.yaml, and one per landmass
    # without a hub, from its most populous cell (its tree root) to the
    # nearest cell of a landmass with one.
    subsea: dict[Pair, float] = field(default_factory=dict)
    roots: set[str] = field(default_factory=set)
    landmass: dict[str, int] = field(default_factory=dict)  # cell -> index, biggest first

    @property
    def city_cells(self) -> set[str]:
        return set(self.metro)

    def subsea_ids(self) -> dict[str, Pair]:
        """Readable ids ("Belfast|Glasgow", or cell ids) -> node pair."""
        return {"|".join(n.split(":", 1)[1] for n in pair): pair for pair in self.subsea}

    def hub_graph(self, cut: set[str] = frozenset()) -> nx.Graph:
        """Hubs joined by backbone and hub-to-hub subsea links."""
        G = nx.Graph()
        G.add_nodes_from(h.name for h in self.hubs)
        G.add_edges_from(self.backbone)
        for sid, (a, b) in self.subsea_ids().items():
            if sid not in cut and a.startswith("pop:") and b.startswith("pop:"):
                G.add_edge(a[4:], b[4:])
        return G

    def ground_graph(self, cut: set[str] = frozenset()) -> nx.Graph:
        """net:{cell} and pop:{hub} nodes joined by kept links, with `km` and
        `type`. Subsea links whose id is in `cut` are left out."""
        G = nx.Graph()
        for (a, b), km in self.neighbours.items():
            G.add_edge(f"net:{a}", f"net:{b}", km=km, type="neighbour")
        for cid, (hub, km) in self.metro.items():
            G.add_edge(f"net:{cid}", f"pop:{hub}", km=km, type="metro")
        for (a, b), km in self.backbone.items():
            G.add_edge(f"pop:{a}", f"pop:{b}", km=km, type="backbone")
        for sid, (a, b) in self.subsea_ids().items():
            if sid not in cut:
                G.add_edge(a, b, km=self.subsea[(a, b)], type="subsea")
        return G


def neighbour_pairs(cells: list[Cell]) -> dict[Pair, float]:
    """Undirected H3-neighbour pairs (a < b) within the grid, with distance in km."""
    by_id = {c.cell_id: c for c in cells}
    pairs: dict[Pair, float] = {}
    for c in cells:
        for nb in h3.grid_ring(c.cell_id, 1):
            if nb in by_id and c.cell_id < nb:
                o = by_id[nb]
                pairs[(c.cell_id, nb)] = haversine_km(c.lat, c.lon, o.lat, o.lon)
    return pairs


def _nearest_cell(lat: float, lon: float, cells: list[Cell]) -> Cell:
    return min(cells, key=lambda c: haversine_km(c.lat, c.lon, lat, lon))


# --- hubs -----------------------------------------------------------------------


def default_hub_count(cells: list[Cell]) -> int:
    area = sum(h3.cell_area(c.cell_id, unit="km^2") for c in cells)
    return int(np.clip(round(area / 10_000), 4, 20))


def auto_hubs(cells: list[Cell], peak_demand: dict[str, float], cfg: MeshConfig) -> list[Hub]:
    """Highest-demand cells, at least hub_min_separation_km apart."""
    count = cfg.hub_count if cfg.hub_count is not None else default_hub_count(cells)
    hubs: list[Hub] = []
    for c in sorted(cells, key=lambda c: (-peak_demand[c.cell_id], c.cell_id)):
        if len(hubs) >= count:
            break
        if any(
            haversine_km(c.lat, c.lon, h.lat, h.lon) < cfg.hub_min_separation_km for h in hubs
        ):
            continue
        radius = cfg.big_hub_radius_km if not hubs else cfg.hub_radius_km
        hubs.append(Hub(f"hub{len(hubs) + 1}", c.lat, c.lon, radius, c.cell_id))
    return hubs


def hubs_from_spec(spec: dict, cells: list[Cell], cfg: MeshConfig) -> list[Hub]:
    hubs = []
    for name, h in spec.items():
        cell = _nearest_cell(h["lat"], h["lon"], cells)
        radius = float(h.get("radius_km", cfg.hub_radius_km))
        hubs.append(Hub(name, float(h["lat"]), float(h["lon"]), radius, cell.cell_id))
    return hubs


def metro_cells(hubs: list[Hub], cells: list[Cell]) -> dict[str, tuple[str, float]]:
    """Cells within a hub's radius join it; a cell in two radii joins the nearest.

    A hub's own cell always joins it, so a small radius still gets one cell.
    """
    out: dict[str, tuple[str, float]] = {}
    for c in cells:
        best: tuple[float, str] | None = None
        for h in hubs:
            d = haversine_km(c.lat, c.lon, h.lat, h.lon)
            if d <= h.radius_km and (best is None or d < best[0]):
                best = (d, h.name)
        if best is not None:
            out[c.cell_id] = (best[1], best[0])
    by_id = {c.cell_id: c for c in cells}
    for h in hubs:
        c = by_id[h.cell_id]
        out[h.cell_id] = (h.name, haversine_km(c.lat, c.lon, h.lat, h.lon))
    return out


def rank_hubs(
    hubs: list[Hub], metro: dict[str, tuple[str, float]], peak_demand: dict[str, float]
) -> list[Hub]:
    """Biggest first, by the peak demand of the hub's city cells."""
    size = {h.name: 0.0 for h in hubs}
    for cid, (hub, _km) in metro.items():
        size[hub] += peak_demand.get(cid, 0.0)
    return sorted(hubs, key=lambda h: (-size[h.name], h.name))


# --- backbone -------------------------------------------------------------------


def _hub_km(a: Hub, b: Hub) -> float:
    return haversine_km(a.lat, a.lon, b.lat, b.lon)


def _key(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)


def auto_backbone(ranked: list[Hub], cfg: MeshConfig) -> dict[Pair, float]:
    """Each hub to its nearest hubs, the biggest hubs fully meshed, then the
    shortest missing links until the hub graph is connected."""
    links: dict[Pair, float] = {}
    for h in ranked:
        others = sorted((o for o in ranked if o.name != h.name), key=lambda o: _hub_km(h, o))
        for o in others[: cfg.hub_degree]:
            links[_key(h.name, o.name)] = _hub_km(h, o)
    big = ranked[: cfg.big_hub_count]
    for i, a in enumerate(big):
        for b in big[i + 1 :]:
            links[_key(a.name, b.name)] = _hub_km(a, b)
    return _connect(ranked, links)


def _connect(hubs: list[Hub], links: dict[Pair, float]) -> dict[Pair, float]:
    G = nx.Graph()
    G.add_nodes_from(h.name for h in hubs)
    G.add_edges_from(links)
    candidates = sorted(
        (_hub_km(a, b), a.name, b.name)
        for i, a in enumerate(hubs)
        for b in hubs[i + 1 :]
        if _key(a.name, b.name) not in links
    )
    for km, a, b in candidates:
        if nx.is_connected(G):
            break
        if not nx.has_path(G, a, b):
            links[_key(a, b)] = km
            G.add_edge(a, b)
    return links


def backbone_from_spec(spec: list, hubs: list[Hub]) -> dict[Pair, float]:
    by_name = {h.name: h for h in hubs}
    links: dict[Pair, float] = {}
    for a, b in spec:
        if a not in by_name or b not in by_name:
            raise ValueError(f"backbone.yaml link [{a}, {b}] names an unknown hub")
        links[_key(a, b)] = _hub_km(by_name[a], by_name[b])
    return links


# --- neighbour mesh -----------------------------------------------------------


def density_classes(cells: list[Cell], cfg: MeshConfig) -> dict[str, str]:
    """By population rank within the region."""
    ranked = sorted(cells, key=lambda c: (-c.population, c.cell_id))
    n = len(ranked)
    n_urban = round(n * cfg.urban_share)
    n_suburban = round(n * (cfg.urban_share + cfg.suburban_share))
    return {
        c.cell_id: URBAN if i < n_urban else SUBURBAN if i < n_suburban else RURAL
        for i, c in enumerate(ranked)
    }


def landmass_links(
    cells: list[Cell],
    pairs: dict[Pair, float],
    city: set[str],
    landing_radius_km: float = 30.0,
    landing_min_population: float = 500.0,
) -> tuple[dict[str, int], dict[Pair, float], set[str]]:
    """Landmass of each cell, plus a subsea link for each landmass without a
    city cell, from its most populous cell (returned as a tree root) to a
    landmass that has one. The cable lands on the most populous cell within
    landing_radius_km of that landmass's nearest point, with at least
    landing_min_population people (else the nearest such cell anywhere on it)."""
    by_id = {c.cell_id: c for c in cells}
    G = nx.Graph()
    G.add_nodes_from(by_id)
    G.add_edges_from(pairs)
    comps = sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))
    landmass = {cid: i for i, comp in enumerate(comps) for cid in comp}
    served = [cid for comp in comps if comp & city for cid in comp]
    links: dict[Pair, float] = {}
    roots: set[str] = set()
    for comp in comps:
        if comp & city or not served:
            continue
        root = max(comp, key=lambda c: (by_id[c].population, c))
        r = by_id[root]
        _km, nearest = min(
            (haversine_km(r.lat, r.lon, by_id[o].lat, by_id[o].lon), o) for o in served
        )
        target = [o for o in served if landmass[o] == landmass[nearest]]
        n = by_id[nearest]
        populated = [o for o in target if by_id[o].population >= landing_min_population]
        close = [
            o for o in populated
            if haversine_km(n.lat, n.lon, by_id[o].lat, by_id[o].lon) <= landing_radius_km
        ]
        if close:
            landing = max(close, key=lambda o: (by_id[o].population, o))
        elif populated:
            landing = min(populated, key=lambda o: (haversine_km(r.lat, r.lon, by_id[o].lat, by_id[o].lon), o))
        else:
            landing = nearest
        km = haversine_km(r.lat, r.lon, by_id[landing].lat, by_id[landing].lon)
        links[_key(f"net:{root}", f"net:{landing}")] = km
        roots.add(root)
    return landmass, links, roots


def prune_neighbours(
    cells: list[Cell],
    pairs: dict[Pair, float],
    city: set[str],
    density: dict[str, str],
    cfg: MeshConfig,
    roots: set[str] = frozenset(),
) -> tuple[dict[Pair, float], dict[str, float]]:
    """Tree to the nearest city cell (or landmass root) plus density-based
    extra links.

    Returns the kept links and each cell's tree distance to a city cell or root.
    """
    G = nx.Graph()
    G.add_nodes_from(c.cell_id for c in cells)
    for (a, b), km in pairs.items():
        G.add_edge(a, b, km=km)
    dist, paths = nx.multi_source_dijkstra(G, city | roots, weight="km")

    kept: set[Pair] = set()
    for cid, path in paths.items():
        if len(path) >= 2:
            kept.add(_key(path[-1], path[-2]))
    tree_degree = {c.cell_id: 0 for c in cells}
    for a, b in kept:
        tree_degree[a] += 1
        tree_degree[b] += 1

    # City cells keep all their links.
    for a, b in pairs:
        if a in city or b in city:
            kept.add((a, b))

    def budget(cid: str) -> float:
        if cid in city or density[cid] == URBAN:
            return float("inf")  # all six neighbours
        extra = cfg.suburban_extra_links if density[cid] == SUBURBAN else cfg.rural_extra_links
        return tree_degree[cid] + extra

    degree = {c.cell_id: 0 for c in cells}
    for a, b in kept:
        degree[a] += 1
        degree[b] += 1

    # Extras: nearest-to-hub cells choose first, and prefer neighbours nearer a hub.
    for cid in sorted(degree, key=lambda c: (dist.get(c, float("inf")), c)):
        options = sorted(
            (nb for nb in G.neighbors(cid) if _key(cid, nb) not in kept),
            key=lambda nb: (dist.get(nb, float("inf")), nb),
        )
        for nb in options:
            if degree[cid] >= budget(cid):
                break
            if degree[nb] >= budget(nb):
                continue
            kept.add(_key(cid, nb))
            degree[cid] += 1
            degree[nb] += 1

    return {p: pairs[p] for p in sorted(kept)}, dict(dist)


def build_hubs(
    cells: list[Cell],
    peak_demand: dict[str, float],
    cfg: MeshConfig,
    backbone_spec: dict | None = None,
) -> tuple[list[Hub], dict[str, tuple[str, float]]]:
    """Hubs (biggest first) and their city cells."""
    if backbone_spec is not None:
        hubs = hubs_from_spec(backbone_spec.get("hubs", {}), cells, cfg)
    else:
        hubs = auto_hubs(cells, peak_demand, cfg)
    metro = metro_cells(hubs, cells)
    return rank_hubs(hubs, metro, peak_demand), metro


def build_topology(
    cells: list[Cell],
    peak_demand: dict[str, float],
    cfg: MeshConfig,
    backbone_spec: dict | None = None,
) -> Topology:
    hubs, metro = build_hubs(cells, peak_demand, cfg, backbone_spec)
    city = set(metro)
    pairs = neighbour_pairs(cells)
    landmass, subsea, roots = landmass_links(
        cells, pairs, city, cfg.subsea.landing_radius_km, cfg.subsea.landing_min_population
    )
    hub_mass = {h.name: landmass[h.cell_id] for h in hubs}

    if backbone_spec is not None:
        backbone = backbone_from_spec(backbone_spec.get("links", []), hubs)
        by_name = {h.name: h for h in hubs}
        for a, b in backbone_spec.get("subsea_links", []):
            if a not in by_name or b not in by_name:
                raise ValueError(f"backbone.yaml subsea link [{a}, {b}] names an unknown hub")
            subsea[_key(f"pop:{a}", f"pop:{b}")] = _hub_km(by_name[a], by_name[b])
    else:
        backbone = {}
        for (a, b), km in auto_backbone(hubs, cfg).items():
            if hub_mass[a] == hub_mass[b]:
                backbone[(a, b)] = km
            else:  # hubs on different landmasses are joined under the sea
                subsea[_key(f"pop:{a}", f"pop:{b}")] = km

    density = density_classes(cells, cfg)
    if cfg.mesh_mode == "pruned":
        kept, hub_dist = prune_neighbours(cells, pairs, city, density, cfg, roots)
    else:
        G = nx.Graph()
        G.add_nodes_from(c.cell_id for c in cells)
        for (a, b), km in pairs.items():
            G.add_edge(a, b, km=km)
        kept = pairs
        hub_dist = dict(nx.multi_source_dijkstra_path_length(G, city | roots, weight="km"))
    topo = Topology(
        hubs=hubs,
        backbone=backbone,
        metro=metro,
        all_neighbours=pairs,
        neighbours=kept,
        density=density,
        hub_distance_km=hub_dist,
        subsea=subsea,
        roots=roots,
        landmass=landmass,
    )
    if hubs and not nx.is_connected(topo.hub_graph()):
        parts = [sorted(p) for p in nx.connected_components(topo.hub_graph())]
        print(f"WARNING: hub graph (backbone + subsea) is not connected: {parts}", file=sys.stderr)
    return topo
