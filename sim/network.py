"""Directed graph and two-pass max-flow routing."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from sim.config import NetworkConfig
from sim.grid import haversine_km
from sim.state import Gateway, Haps

LARGE_MBPS = 1e12
FlowKey = tuple[str, str, str]


@dataclass
class Residuals:
    access: dict[str, float] = field(default_factory=dict)
    isl: dict[tuple[str, str], float] = field(default_factory=dict)
    feeder: dict[tuple[str, str], float] = field(default_factory=dict)

    def copy(self) -> Residuals:
        return Residuals(
            access=dict(self.access),
            isl=dict(self.isl),
            feeder=dict(self.feeder),
        )


@dataclass
class FlowResult:
    served: dict[str, float]
    cell_haps: dict[tuple[str, str], float]
    links: dict[FlowKey, float]
    residuals: Residuals


def initial_residuals(
    haps: list[Haps],
    gateways: list[Gateway],
    cfg: NetworkConfig,
) -> Residuals:
    access = {h.id: cfg.haps_access_mbps for h in haps}
    isl: dict[tuple[str, str], float] = {}
    for a in haps:
        for b in haps:
            if a.id == b.id:
                continue
            if haversine_km(a.lat, a.lon, b.lat, b.lon) <= cfg.isl_range_km:
                isl[(a.id, b.id)] = cfg.isl_mbps
    feeder: dict[tuple[str, str], float] = {}
    for h in haps:
        for g in gateways:
            if haversine_km(h.lat, h.lon, g.lat, g.lon) <= cfg.feeder_range_km:
                feeder[(h.id, g.id)] = cfg.feeder_mbps
    return Residuals(access=access, isl=isl, feeder=feeder)


def cell_coverage(
    cell_lat: float,
    cell_lon: float,
    haps: list[Haps],
    coverage_km: float,
    allowed: set[str] | None = None,
) -> list[str]:
    out = []
    for h in haps:
        if allowed is not None and h.id not in allowed:
            continue
        if haversine_km(cell_lat, cell_lon, h.lat, h.lon) <= coverage_km:
            out.append(h.id)
    return out


def route_overflow(
    overflow: dict[str, float],
    coverage: dict[str, list[str]],
    haps: list[Haps],
    gateways: list[Gateway],
    residuals: Residuals,
) -> FlowResult:
    """One max-flow pass. `overflow` is per-cell Mbps for this class of traffic."""
    G = nx.DiGraph()
    source, sink = "SOURCE", "SINK"
    active = [cid for cid, mbps in overflow.items() if mbps > 1e-9]

    for cid in active:
        G.add_edge(source, f"cell:{cid}", capacity=float(overflow[cid]))
        for hid in coverage.get(cid, []):
            G.add_edge(f"cell:{cid}", f"haps_access:{hid}", capacity=LARGE_MBPS)

    for hid, cap in residuals.access.items():
        if cap > 1e-12:
            G.add_edge(f"haps_access:{hid}", f"haps_core:{hid}", capacity=float(cap))

    for (a, b), cap in residuals.isl.items():
        if cap > 1e-12:
            G.add_edge(f"haps_core:{a}", f"haps_core:{b}", capacity=float(cap))

    for (hid, gid), cap in residuals.feeder.items():
        if cap > 1e-12:
            G.add_edge(f"haps_core:{hid}", f"gw:{gid}", capacity=float(cap))

    for g in gateways:
        G.add_edge(f"gw:{g.id}", sink, capacity=LARGE_MBPS)

    served: dict[str, float] = {}
    cell_haps: dict[tuple[str, str], float] = {}
    links: dict[FlowKey, float] = {}
    leftover = residuals.copy()

    if not active or source not in G or sink not in G:
        return FlowResult(served, cell_haps, links, leftover)

    _value, flow_dict = nx.maximum_flow(G, source, sink)

    def edge_flow(u: str, v: str) -> float:
        return float(flow_dict.get(u, {}).get(v, 0.0) or 0.0)

    for cid in active:
        served[cid] = edge_flow(source, f"cell:{cid}")
        for hid in coverage.get(cid, []):
            fl = edge_flow(f"cell:{cid}", f"haps_access:{hid}")
            if fl > 1e-12:
                cell_haps[(cid, hid)] = fl

    for hid in leftover.access:
        fl = edge_flow(f"haps_access:{hid}", f"haps_core:{hid}")
        if fl > 1e-12:
            links[("access", hid, hid)] = fl
            leftover.access[hid] = max(0.0, leftover.access[hid] - fl)

    for a, b in leftover.isl:
        fl = edge_flow(f"haps_core:{a}", f"haps_core:{b}")
        if fl > 1e-12:
            links[("isl", a, b)] = fl
            leftover.isl[(a, b)] = max(0.0, leftover.isl[(a, b)] - fl)

    for hid, gid in leftover.feeder:
        fl = edge_flow(f"haps_core:{hid}", f"gw:{gid}")
        if fl > 1e-12:
            links[("feeder", hid, gid)] = fl
            leftover.feeder[(hid, gid)] = max(0.0, leftover.feeder[(hid, gid)] - fl)

    return FlowResult(served, cell_haps, links, leftover)


def merge_flows(*parts: dict[FlowKey, float]) -> dict[FlowKey, float]:
    out: dict[FlowKey, float] = {}
    for part in parts:
        for key, value in part.items():
            out[key] = out.get(key, 0.0) + value
    return out


def merge_cell_haps(
    *parts: dict[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for part in parts:
        for key, value in part.items():
            out[key] = out.get(key, 0.0) + value
    return out
