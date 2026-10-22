"""Simulation engine: one tick of demand, ground, routing, and latency."""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import pandas as pd

from sim.config import AppConfig, load_config
from sim.demand import (
    active_fraction,
    allocate_ground,
    demand_mbps,
    peak_ground_capacity,
    set_tower_status as _set_tower_status,
)
from sim.grid import load_or_build_cells
from sim.latency import cell_latency_and_hops, haps_routes
from sim.network import (
    cell_coverage,
    initial_residuals,
    merge_cell_haps,
    merge_flows,
    route_overflow,
)
from sim.state import (
    Cell,
    CellResult,
    Gateway,
    Haps,
    LinkLoad,
    TickResult,
    Totals,
)

ROOT = Path(__file__).resolve().parents[1]


def _parse_hour(sim_time: float | str | datetime | time) -> tuple[float, str]:
    if isinstance(sim_time, datetime):
        hour = sim_time.hour + sim_time.minute / 60.0 + sim_time.second / 3600.0
        return hour, sim_time.strftime("%H:%M")
    if isinstance(sim_time, time):
        hour = sim_time.hour + sim_time.minute / 60.0 + sim_time.second / 3600.0
        return hour, sim_time.strftime("%H:%M")
    if isinstance(sim_time, str):
        parts = sim_time.strip().split(":")
        hour = int(parts[0]) + (int(parts[1]) / 60.0 if len(parts) > 1 else 0.0)
        label = f"{int(hour) % 24:02d}:{int(round((hour % 1) * 60)) % 60:02d}"
        return hour, sim_time
    hour = float(sim_time)
    label = f"{int(hour) % 24:02d}:{int(round((hour % 1) * 60)) % 60:02d}"
    return hour, label


def _cells_from_frame(df: pd.DataFrame) -> list[Cell]:
    cells = []
    for row in df.itertuples(index=False):
        cells.append(
            Cell(
                cell_id=str(row.cell_id),
                lat=float(row.lat),
                lon=float(row.lon),
                population=float(row.population),
            )
        )
    return cells


class Simulation:
    def __init__(
        self,
        config_path: str | Path | None = None,
        cells: list[Cell] | pd.DataFrame | None = None,
        project_root: str | Path | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.root = Path(project_root) if project_root else ROOT
        if config is not None:
            self.config = config
        else:
            path = Path(config_path) if config_path else self.root / "config.yaml"
            self.config = load_config(path)

        if cells is None:
            df = load_or_build_cells(self.config.grid, self.root)
            self.cells = _cells_from_frame(df)
        elif isinstance(cells, pd.DataFrame):
            self.cells = _cells_from_frame(cells)
        else:
            self.cells = cells

        for cell in self.cells:
            cell.ground_capacity_mbps = peak_ground_capacity(
                cell.population, self.config.demand
            )

        self.haps = [
            Haps(id=p.id, lat=p.lat, lon=p.lon) for p in self.config.haps
        ]
        self.gateways = [
            Gateway(id=p.id, lat=p.lat, lon=p.lon) for p in self.config.gateways
        ]
        self.routes = haps_routes(
            self.haps,
            self.gateways,
            self.config.network.isl_range_km,
            self.config.network.feeder_range_km,
            self.config.latency,
        )
        reachable = {hid for hid, route in self.routes.items() if route is not None}
        self.coverage = {
            cell.cell_id: cell_coverage(
                cell.lat,
                cell.lon,
                self.haps,
                self.config.network.haps_coverage_km,
                allowed=reachable,
            )
            for cell in self.cells
        }
        self._base_residuals = initial_residuals(
            self.haps, self.gateways, self.config.network
        )

    def set_tower_status(self, lat: float, lon: float, radius_km: float, value: float) -> None:
        _set_tower_status(self.cells, lat, lon, radius_km, value)

    def step(self, sim_time: float | str | datetime | time) -> TickResult:
        hour, label = _parse_hour(sim_time)
        frac = active_fraction(hour, self.config.demand)
        net = self.config.network
        lat_cfg = self.config.latency

        overflow_c: dict[str, float] = {}
        overflow_g: dict[str, float] = {}
        demand_c: dict[str, float] = {}
        demand_g: dict[str, float] = {}
        ground_c: dict[str, float] = {}
        ground_g: dict[str, float] = {}

        for cell in self.cells:
            crit, gen = demand_mbps(cell.population, frac, self.config.demand)
            demand_c[cell.cell_id] = crit
            demand_g[cell.cell_id] = gen
            usable = cell.ground_capacity_mbps * cell.tower_status
            g_c, g_g, o_c, o_g = allocate_ground(crit, gen, usable)
            ground_c[cell.cell_id] = g_c
            ground_g[cell.cell_id] = g_g
            if o_c > 1e-9:
                overflow_c[cell.cell_id] = o_c
            if o_g > 1e-9:
                overflow_g[cell.cell_id] = o_g

        residuals = self._base_residuals.copy()
        pass_c = route_overflow(overflow_c, self.coverage, self.haps, self.gateways, residuals)
        pass_g = route_overflow(
            overflow_g, self.coverage, self.haps, self.gateways, pass_c.residuals
        )
        haps_c = pass_c.served
        haps_g = pass_g.served
        cell_haps = merge_cell_haps(pass_c.cell_haps, pass_g.cell_haps)
        link_flow = merge_flows(pass_c.links, pass_g.links)

        cell_results: list[CellResult] = []
        total_demand = 0.0
        total_crit = 0.0
        served_all = 0.0
        served_crit = 0.0
        unserved_c = 0.0
        unserved_g = 0.0
        ground_traffic = 0.0
        haps_lat_num = 0.0
        haps_traffic = 0.0

        for cell in self.cells:
            cid = cell.cell_id
            d_c = demand_c[cid]
            d_g = demand_g[cid]
            g_served = ground_c[cid] + ground_g[cid]
            h_c = haps_c.get(cid, 0.0)
            h_g = haps_g.get(cid, 0.0)
            u_c = max(0.0, overflow_c.get(cid, 0.0) - h_c)
            u_g = max(0.0, overflow_g.get(cid, 0.0) - h_g)
            per_haps = {
                hid: mbps
                for (cell_id, hid), mbps in cell_haps.items()
                if cell_id == cid
            }
            latency, hops = cell_latency_and_hops(
                g_served, lat_cfg.ground_latency_ms, per_haps, self.routes
            )
            cell_results.append(
                CellResult(
                    cell_id=cid,
                    demand_critical_mbps=d_c,
                    demand_general_mbps=d_g,
                    ground_served_mbps=g_served,
                    haps_served_critical_mbps=h_c,
                    haps_served_general_mbps=h_g,
                    unserved_critical_mbps=u_c,
                    unserved_general_mbps=u_g,
                    latency_ms=latency,
                    hops=hops,
                )
            )
            total_demand += d_c + d_g
            total_crit += d_c
            served_all += g_served + h_c + h_g
            served_crit += ground_c[cid] + h_c
            unserved_c += u_c
            unserved_g += u_g
            ground_traffic += g_served
            for hid, mbps in per_haps.items():
                route = self.routes.get(hid)
                if route is None:
                    continue
                haps_lat_num += mbps * route.latency_ms
                haps_traffic += mbps

        orig = self._base_residuals
        links: list[LinkLoad] = []
        for hid, cap in orig.access.items():
            flow = link_flow.get(("access", hid, hid), 0.0)
            links.append(
                LinkLoad("access", hid, hid, flow, cap, flow / cap if cap else 0.0)
            )
        for (a, b), cap in orig.isl.items():
            flow = link_flow.get(("isl", a, b), 0.0)
            links.append(
                LinkLoad("isl", a, b, flow, cap, flow / cap if cap else 0.0)
            )
        for (hid, gid), cap in orig.feeder.items():
            flow = link_flow.get(("feeder", hid, gid), 0.0)
            links.append(
                LinkLoad("feeder", hid, gid, flow, cap, flow / cap if cap else 0.0)
            )

        over80 = sum(1 for link in links if link.utilisation >= 0.8)
        pct_served = 100.0 * served_all / total_demand if total_demand else 100.0
        pct_crit = 100.0 * served_crit / total_crit if total_crit else 100.0
        avg_ground = lat_cfg.ground_latency_ms if ground_traffic > 1e-12 else 0.0
        avg_haps = haps_lat_num / haps_traffic if haps_traffic > 1e-12 else 0.0

        return TickResult(
            sim_time=label,
            cells=cell_results,
            links=links,
            totals=Totals(
                demand_mbps=total_demand,
                pct_served=pct_served,
                pct_critical_served=pct_crit,
                unserved_critical_mbps=unserved_c,
                unserved_general_mbps=unserved_g,
                links_over_80pct=over80,
                avg_ground_latency_ms=avg_ground,
                avg_haps_latency_ms=avg_haps,
            ),
        )
