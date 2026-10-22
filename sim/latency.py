"""Shortest-path HAPS latency and per-cell averages."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from sim.config import LatencyConfig
from sim.grid import haversine_km
from sim.state import Gateway, Haps


@dataclass
class HapsRoute:
    latency_ms: float
    hops: int
    distance_km: float


def haps_routes(
    haps: list[Haps],
    gateways: list[Gateway],
    isl_range_km: float,
    feeder_range_km: float,
    cfg: LatencyConfig,
) -> dict[str, HapsRoute | None]:
    """Shortest distance path from each HAPS through ISLs to any gateway."""
    G = nx.Graph()
    for h in haps:
        G.add_node(h.id)
    gw_nodes = []
    for g in gateways:
        node = f"gw:{g.id}"
        gw_nodes.append(node)
        G.add_node(node)

    for i, a in enumerate(haps):
        for b in haps[i + 1 :]:
            d = haversine_km(a.lat, a.lon, b.lat, b.lon)
            if d <= isl_range_km:
                G.add_edge(a.id, b.id, distance=d)
        for g in gateways:
            d = haversine_km(a.lat, a.lon, g.lat, g.lon)
            if d <= feeder_range_km:
                G.add_edge(a.id, f"gw:{g.id}", distance=d)

    routes: dict[str, HapsRoute | None] = {}
    for h in haps:
        best: tuple[float, list[str]] | None = None
        for gw in gw_nodes:
            if h.id not in G or gw not in G:
                continue
            if not nx.has_path(G, h.id, gw):
                continue
            path = nx.shortest_path(G, h.id, gw, weight="distance")
            dist = float(nx.path_weight(G, path, "distance"))
            if best is None or dist < best[0]:
                best = (dist, path)
        if best is None:
            routes[h.id] = None
            continue
        dist, path = best
        n_haps = sum(1 for node in path if not str(node).startswith("gw:"))
        latency = cfg.haps_base_ms + cfg.per_haps_ms * n_haps + dist / cfg.speed_km_per_ms
        routes[h.id] = HapsRoute(latency_ms=latency, hops=n_haps, distance_km=dist)
    return routes


def cell_latency_and_hops(
    ground_mbps: float,
    ground_latency_ms: float,
    haps_mbps: dict[str, float],
    routes: dict[str, HapsRoute | None],
) -> tuple[float, float]:
    num_lat = ground_mbps * ground_latency_ms
    num_hops = 0.0
    den = ground_mbps
    for hid, mbps in haps_mbps.items():
        route = routes.get(hid)
        if route is None or mbps <= 0.0:
            continue
        num_lat += mbps * route.latency_ms
        num_hops += mbps * route.hops
        den += mbps
    if den <= 1e-12:
        return 0.0, 0.0
    return num_lat / den, num_hops / den
