"""Ground backbone ("mesh"): H3-neighbour backhaul, endpoints, calibration, routing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sim.config import DemandConfig, MeshConfig
from sim.endpoints import endpoint_capacities
from sim.grid import haversine_km
from sim.network import FlowKey, Residuals
from sim.state import Cell, Gateway
from sim.topology import Pair

# Network simplex wants integers: capacities go in as whole bps.
BPS_PER_MBPS = 1_000_000
# Link costs are in millimetres plus a fixed sub-metre tie-break per link. The
# hex lattice has many equal-length routes, and with ties the solver fills
# arbitrary links; a unique optimum makes a normal tick reproduce calibration.
COST_PER_KM = 1_000_000
TIE_BREAK_MAX = 1000
# Cost of entering the HAPS layer: more than any ground route (10,000 km).
HAPS_COST = 10_000 * COST_PER_KM
CACHE_KEY_FIELD = b"calibration_key"
# Bump when the calibration method changes, to invalidate cached backhaul files.
CALIBRATION_VERSION = 2


def link_cost(a: str, b: str, km: float) -> int:
    lo, hi = sorted((a, b))
    tie = int(hashlib.sha1(f"{lo}{hi}".encode()).hexdigest()[:8], 16) % TIE_BREAK_MAX
    return int(round(km * 1000)) * TIE_BREAK_MAX + tie


def gateway_cells(gateways: list[Gateway], cells: list[Cell]) -> dict[str, str]:
    """Each gateway joins the backbone at its nearest grid cell."""
    out: dict[str, str] = {}
    for g in gateways:
        best = min(cells, key=lambda c: haversine_km(c.lat, c.lon, g.lat, g.lon), default=None)
        if best is not None:
            out[g.id] = best.cell_id
    return out


def hops_to_top_endpoints(
    cells: list[Cell], pairs: dict[Pair, float], weights: dict[str, float], top_n: int
) -> dict[str, int]:
    """Backhaul hops from each cell to its nearest top-N endpoint cell."""
    G = nx.Graph()
    G.add_nodes_from(c.cell_id for c in cells)
    G.add_edges_from(pairs)
    top = sorted(weights, key=weights.get, reverse=True)[:top_n]
    dist = nx.multi_source_dijkstra_path_length(G, set(top)) if top else {}
    # A cell cut off from every top endpoint (an island) gets no hop penalty.
    return {c.cell_id: int(dist.get(c.cell_id, 0)) for c in cells}


# --- calibration -------------------------------------------------------------


@dataclass
class Backhaul:
    distance_km: dict[Pair, float]
    calibrated_flow: dict[tuple[str, str], float]  # directed, Mbps
    capacity: dict[Pair, float]  # per undirected pair, available in each direction


def calibration_key(
    cells: list[Cell], demand_cfg: DemandConfig, mesh_cfg: MeshConfig, h3_resolution: int
) -> str:
    payload = {
        "version": CALIBRATION_VERSION,
        "cells": sorted((c.cell_id, round(c.population, 6)) for c in cells),
        "h3_resolution": h3_resolution,
        "demand": demand_cfg.model_dump(),
        "mesh": mesh_cfg.model_dump(
            include={
                "endpoint_exponent",
                "endpoint_headroom",
                "backhaul_headroom",
                "backhaul_min_mbps",
            }
        ),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def calibrate_backhaul(
    cells: list[Cell],
    pairs: dict[Pair, float],
    demand: dict[str, float],
    ground_access: dict[str, float],
    weights: dict[str, float],
    cfg: MeshConfig,
) -> Backhaul:
    """Min-cost max-flow at normal peak demand over an uncapacitated backbone.

    HAPS are left out: the backbone is sized for normal operation, when the
    HAPS layer carries nothing.
    """
    total = sum(demand.values())
    sink_cap = endpoint_capacities(weights, total, cfg.endpoint_headroom)
    G = nx.DiGraph()
    for c in cells:
        cid = c.cell_id
        G.add_edge("SOURCE", f"users:{cid}", capacity=int(demand[cid] * BPS_PER_MBPS), weight=0)
        G.add_edge(
            f"users:{cid}", f"net:{cid}",
            capacity=int(ground_access[cid] * BPS_PER_MBPS), weight=0,
        )
        G.add_edge(f"net:{cid}", "SINK", capacity=int(sink_cap[cid] * BPS_PER_MBPS), weight=0)
    for (a, b), d in pairs.items():
        cost = link_cost(a, b, d)
        G.add_edge(f"net:{a}", f"net:{b}", weight=cost)
        G.add_edge(f"net:{b}", f"net:{a}", weight=cost)

    flow = nx.max_flow_min_cost(G, "SOURCE", "SINK")

    calibrated: dict[tuple[str, str], float] = {}
    capacity: dict[Pair, float] = {}
    for a, b in pairs:
        f_ab = flow[f"net:{a}"][f"net:{b}"] / BPS_PER_MBPS
        f_ba = flow[f"net:{b}"][f"net:{a}"] / BPS_PER_MBPS
        calibrated[(a, b)] = f_ab
        calibrated[(b, a)] = f_ba
        capacity[(a, b)] = max(cfg.backhaul_headroom * max(f_ab, f_ba), cfg.backhaul_min_mbps)
    return Backhaul(distance_km=dict(pairs), calibrated_flow=calibrated, capacity=capacity)


def _write_backhaul(path: Path, bh: Backhaul, key: str) -> None:
    rows = [
        {
            "a": a,
            "b": b,
            "distance_km": d,
            "flow_ab_mbps": bh.calibrated_flow[(a, b)],
            "flow_ba_mbps": bh.calibrated_flow[(b, a)],
            "capacity_mbps": bh.capacity[(a, b)],
        }
        for (a, b), d in bh.distance_km.items()
    ]
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[CACHE_KEY_FIELD] = key.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)


def _read_backhaul(path: Path, key: str, pairs: dict[Pair, float]) -> Backhaul | None:
    if not path.exists():
        return None
    table = pq.read_table(path)
    if (table.schema.metadata or {}).get(CACHE_KEY_FIELD, b"").decode() != key:
        return None
    df = table.to_pandas()
    calibrated: dict[tuple[str, str], float] = {}
    capacity: dict[Pair, float] = {}
    for row in df.itertuples(index=False):
        calibrated[(row.a, row.b)] = float(row.flow_ab_mbps)
        calibrated[(row.b, row.a)] = float(row.flow_ba_mbps)
        capacity[(row.a, row.b)] = float(row.capacity_mbps)
    if set(capacity) != set(pairs):
        return None
    return Backhaul(distance_km=dict(pairs), calibrated_flow=calibrated, capacity=capacity)


def load_or_calibrate(
    path: Path,
    key: str,
    cells: list[Cell],
    pairs: dict[Pair, float],
    demand: dict[str, float],
    ground_access: dict[str, float],
    weights: dict[str, float],
    cfg: MeshConfig,
) -> Backhaul:
    cached = _read_backhaul(path, key, pairs)
    if cached is not None:
        return cached
    bh = calibrate_backhaul(cells, pairs, demand, ground_access, weights, cfg)
    _write_backhaul(path, bh, key)
    return bh


# --- per-tick routing ----------------------------------------------------------


@dataclass
class MeshResiduals:
    backhaul: dict[tuple[str, str], float]  # directed
    endpoint: dict[str, float]
    access: dict[str, float]  # users -> net, per cell
    haps: Residuals

    def copy(self) -> MeshResiduals:
        return MeshResiduals(
            backhaul=dict(self.backhaul),
            endpoint=dict(self.endpoint),
            access=dict(self.access),
            haps=self.haps.copy(),
        )


@dataclass
class MeshFlowResult:
    ground: dict[str, float] = field(default_factory=dict)  # users -> net
    haps: dict[str, float] = field(default_factory=dict)  # users -> any HAPS
    cell_haps: dict[tuple[str, str], float] = field(default_factory=dict)
    links: dict[FlowKey, float] = field(default_factory=dict)
    transit: dict[str, float] = field(default_factory=dict)
    crossed_backhaul_mbps: float = 0.0
    residuals: MeshResiduals | None = None


def _build_graph(
    demand: dict[str, float],
    res: MeshResiduals,
    coverage: dict[str, list[str]],
    gw_cell: dict[str, str],
    distance_km: dict[Pair, float],
) -> nx.DiGraph:
    """Integer bps capacities and link costs, for network simplex.

    Edges with no `capacity` are uncapacitated.
    """
    def bps(mbps: float) -> int:
        return int(mbps * BPS_PER_MBPS)

    G = nx.DiGraph()
    for cid, mbps in demand.items():
        if mbps > 1e-9:
            # Rounded up so truncation never shows up as unserved demand.
            G.add_edge("SOURCE", f"users:{cid}", capacity=math.ceil(mbps * BPS_PER_MBPS), weight=0)
            if res.access.get(cid, 0.0) > 1e-12:
                G.add_edge(f"users:{cid}", f"net:{cid}", capacity=bps(res.access[cid]), weight=0)
            for hid in coverage.get(cid, []):
                G.add_edge(f"users:{cid}", f"haps_access:{hid}", weight=HAPS_COST)
    for (a, b), cap in res.backhaul.items():
        if cap > 1e-12:
            cost = link_cost(a, b, distance_km[(a, b) if a < b else (b, a)])
            G.add_edge(f"net:{a}", f"net:{b}", capacity=bps(cap), weight=cost)
    for cid, cap in res.endpoint.items():
        if cap > 1e-12:
            G.add_edge(f"net:{cid}", "SINK", capacity=bps(cap), weight=0)
    for hid, cap in res.haps.access.items():
        if cap > 1e-12:
            G.add_edge(f"haps_access:{hid}", f"haps_core:{hid}", capacity=bps(cap), weight=0)
    for (a, b), cap in res.haps.isl.items():
        if cap > 1e-12:
            G.add_edge(f"haps_core:{a}", f"haps_core:{b}", capacity=bps(cap), weight=0)
    for (hid, gid), cap in res.haps.feeder.items():
        if cap > 1e-12:
            G.add_edge(f"haps_core:{hid}", f"gw:{gid}", capacity=bps(cap), weight=0)
    for gid, cid in gw_cell.items():
        G.add_edge(f"gw:{gid}", f"net:{cid}", weight=0)
    return G


def route_mesh_pass(
    demand: dict[str, float],
    coverage: dict[str, list[str]],
    gw_cell: dict[str, str],
    distance_km: dict[Pair, float],
    residuals: MeshResiduals,
) -> MeshFlowResult:
    """One traffic class: max flow, cheapest routes among max flows.

    Backhaul costs its length; HAPS access costs more than any ground
    route, so HAPS only carry what the ground can't.
    """
    out = MeshFlowResult()
    res = residuals.copy()
    out.residuals = res
    G = _build_graph(demand, res, coverage, gw_cell, distance_km)
    if "SOURCE" not in G or "SINK" not in G:
        return out
    fd = nx.max_flow_min_cost(G, "SOURCE", "SINK")

    def ef(u: str, v: str) -> float:
        return fd.get(u, {}).get(v, 0) / BPS_PER_MBPS

    users_in: dict[str, float] = {}
    gw_in: dict[str, float] = {}
    for cid, mbps in demand.items():
        if mbps <= 1e-9:
            continue
        g = min(ef(f"users:{cid}", f"net:{cid}"), mbps)
        if g > 1e-12:
            users_in[cid] = g
            out.ground[cid] = g
            res.access[cid] = max(0.0, res.access[cid] - g)
        for hid in coverage.get(cid, []):
            fl = ef(f"users:{cid}", f"haps_access:{hid}")
            if fl > 1e-12:
                out.haps[cid] = out.haps.get(cid, 0.0) + fl
                out.cell_haps[(cid, hid)] = fl

    # Flow both ways along a link would cost more, but cancel any to the net flow.
    bh_in: dict[str, float] = {}
    bh_out: dict[str, float] = {}
    for a, b in list(res.backhaul):
        if a > b:
            continue
        net = ef(f"net:{a}", f"net:{b}") - ef(f"net:{b}", f"net:{a}")
        if abs(net) <= 1e-12:
            continue
        u, v, fl = (a, b, net) if net > 0 else (b, a, -net)
        out.links[("backhaul", u, v)] = fl
        res.backhaul[(u, v)] = max(0.0, res.backhaul[(u, v)] - fl)
        bh_out[u] = bh_out.get(u, 0.0) + fl
        bh_in[v] = bh_in.get(v, 0.0) + fl

    sink: dict[str, float] = {}
    for cid in res.endpoint:
        fl = ef(f"net:{cid}", "SINK")
        if fl > 1e-12:
            sink[cid] = fl
            res.endpoint[cid] = max(0.0, res.endpoint[cid] - fl)

    for gid, cid in gw_cell.items():
        fl = ef(f"gw:{gid}", f"net:{cid}")
        if fl > 1e-12:
            gw_in[cid] = gw_in.get(cid, 0.0) + fl
    h = res.haps
    for hid in h.access:
        fl = ef(f"haps_access:{hid}", f"haps_core:{hid}")
        if fl > 1e-12:
            out.links[("access", hid, hid)] = fl
            h.access[hid] = max(0.0, h.access[hid] - fl)
    for a, b in h.isl:
        fl = ef(f"haps_core:{a}", f"haps_core:{b}")
        if fl > 1e-12:
            out.links[("isl", a, b)] = fl
            h.isl[(a, b)] = max(0.0, h.isl[(a, b)] - fl)
    for hid, gid in h.feeder:
        fl = ef(f"haps_core:{hid}", f"gw:{gid}")
        if fl > 1e-12:
            out.links[("feeder", hid, gid)] = fl
            h.feeder[(hid, gid)] = max(0.0, h.feeder[(hid, gid)] - fl)

    # Flows aren't labelled by origin, so assume traffic mixes at each cell:
    # what enters over backhaul leaves over backhaul or to the local endpoint
    # in proportion to those outflows.
    for cid in set(bh_in) | set(users_in) | set(gw_in):
        inflow = bh_in.get(cid, 0.0) + users_in.get(cid, 0.0) + gw_in.get(cid, 0.0)
        if inflow <= 1e-12:
            continue
        b_in = bh_in.get(cid, 0.0)
        through = b_in * bh_out.get(cid, 0.0) / inflow
        if through > 1e-12:
            out.transit[cid] = through
        out.crossed_backhaul_mbps += b_in * sink.get(cid, 0.0) / inflow
    return out
