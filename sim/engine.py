"""Simulation engine: demand, ground, routing and latency per tick; HAPS
movement and battery between ticks."""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path

import pandas as pd

from sim.config import AppConfig, load_config
from sim.controllers import Action, CellObs, GatewayObs, HapsObs, Observation, PlatformSpec
from sim.dynamics import battery_after_tick, move_towards
from sim.demand import (
    active_fraction,
    allocate_ground,
    demand_mbps,
    peak_ground_capacity,
    set_tower_status as _set_tower_status,
)
from sim.endpoints import endpoint_capacities, endpoint_weights
from sim.grid import load_or_build_cells
from sim.region import DEFAULT_REGION, Region, load_region, region_config
from sim import mesh
from sim.latency import cell_latency_and_hops, haps_routes
from sim.network import (
    Residuals,
    cell_coverage,
    initial_residuals,
    merge_cell_haps,
    merge_flows,
    route_overflow,
)
from sim.topology import build_hubs, neighbour_pairs
from sim.traffic import TrafficModel, classify_areas
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
    """One region's network.

    With no `config` or `config_path`, config.yaml is merged with the region's
    overrides and assets. With no `cells`, the region's grid is loaded or built.
    The region defaults to "england" when either is needed.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        cells: list[Cell] | pd.DataFrame | None = None,
        project_root: str | Path | None = None,
        config: AppConfig | None = None,
        region: str | Region | None = None,
    ) -> None:
        self.root = Path(project_root) if project_root else ROOT
        if region is None and (cells is None or (config is None and config_path is None)):
            region = DEFAULT_REGION
        self.region = load_region(region) if isinstance(region, str) else region

        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(Path(config_path))
        else:
            self.config = region_config(self.region)

        if cells is None:
            df = load_or_build_cells(self.region, self.root)
            self.cells = _cells_from_frame(df)
        elif isinstance(cells, pd.DataFrame):
            self.cells = _cells_from_frame(cells)
        else:
            self.cells = cells

        self.processed_dir = (
            self.region.processed_dir(self.root)
            if self.region is not None
            else self.root / "data" / "processed" / "custom"
        )

        self.traffic: TrafficModel | None = None
        if self.config.demand_model == "area_types":
            if self.config.ground_model == "mesh":
                raise NotImplementedError(
                    "demand_model: area_types is only supported with ground_model: simple"
                )
            self._init_traffic()
        else:
            for cell in self.cells:
                cell.ground_capacity_mbps = peak_ground_capacity(
                    cell.population, self.config.demand
                )

        dyn = self.config.dynamics
        self.haps = [
            Haps(
                id=p.id,
                lat=p.lat,
                lon=p.lon,
                speed_kmh=dyn.speed_kmh,
                battery_pct=dyn.battery_start_pct,
                battery_floor_pct=dyn.battery_floor_pct,
            )
            for p in self.config.haps
        ]
        self.gateways = [
            Gateway(id=p.id, lat=p.lat, lon=p.lon) for p in self.config.gateways
        ]
        self._refresh_haps_geometry()
        self.sim_minutes = 0.0
        self.last_result: TickResult | None = None
        self._auto_low_power: set[str] = set()
        if self.config.ground_model == "mesh":
            self._init_mesh()

    def _init_traffic(self) -> None:
        frac = self.config.demand.active_fraction_peak
        peak_20 = {
            c.cell_id: sum(demand_mbps(c.population, frac, self.config.demand))
            for c in self.cells
        }
        backbone = self.region.backbone if self.region is not None else None
        self.hubs, self.metro = build_hubs(self.cells, peak_20, self.config.mesh, backbone)
        self.area_type = classify_areas(self.cells, self.hubs, self.metro, self.config.traffic)
        self.traffic = TrafficModel(
            self.cells, self.area_type, self.config.traffic, self.config.demand
        )
        for cell in self.cells:
            cell.ground_capacity_mbps = self.traffic.ground_capacity(cell)

    def _refresh_haps_geometry(self) -> None:
        """Routes, coverage and HAPS links for the current HAPS positions."""
        net = self.config.network
        self.routes = haps_routes(
            self.haps, self.gateways, net.isl_range_km, net.feeder_range_km, self.config.latency
        )
        reachable = {hid for hid, route in self.routes.items() if route is not None}
        self.coverage = {
            cell.cell_id: cell_coverage(
                cell.lat, cell.lon, self.haps, net.haps_coverage_km, allowed=reachable
            )
            for cell in self.cells
        }
        self._base_residuals = initial_residuals(self.haps, self.gateways, net)

    def _tick_residuals(self) -> Residuals:
        """HAPS link capacity now: low power halves access, gateway health scales feeders."""
        res = self._base_residuals.copy()
        factor = self.config.dynamics.low_power_access_factor
        for h in self.haps:
            if h.power_mode == "low" and h.id in res.access:
                res.access[h.id] *= factor
        status = {g.id: g.status for g in self.gateways}
        for key in res.feeder:
            if status[key[1]] != 1.0:
                res.feeder[key] *= status[key[1]]
        return res

    def _init_mesh(self) -> None:
        mcfg = self.config.mesh
        self.pairs = neighbour_pairs(self.cells)
        self.endpoint_weights = endpoint_weights(self.cells, mcfg.endpoint_exponent)
        self.gw_cell = mesh.gateway_cells(self.gateways, self.cells)
        self.ground_hops = mesh.hops_to_top_endpoints(
            self.cells, self.pairs, self.endpoint_weights, mcfg.latency_top_endpoints
        )
        peak = self.config.demand.active_fraction_peak
        peak_demand = {
            c.cell_id: sum(demand_mbps(c.population, peak, self.config.demand))
            for c in self.cells
        }
        self.backhaul = mesh.load_or_calibrate(
            self.processed_dir / "backhaul.parquet",
            mesh.calibration_key(
                self.cells, self.config.demand, mcfg, self.config.grid.h3_resolution
            ),
            self.cells,
            self.pairs,
            peak_demand,
            {c.cell_id: c.ground_capacity_mbps for c in self.cells},
            self.endpoint_weights,
            mcfg,
        )

    def set_tower_status(self, lat: float, lon: float, radius_km: float, value: float) -> None:
        _set_tower_status(self.cells, lat, lon, radius_km, value)

    def set_gateway_status(self, gateway_id: str, value: float) -> None:
        next(g for g in self.gateways if g.id == gateway_id).status = value

    def set_time(self, sim_time: float | str | datetime | time) -> None:
        hour, _label = _parse_hour(sim_time)
        self.sim_minutes = hour * 60.0
        self.last_result = None

    def assign(
        self,
        haps_id: str,
        target_lat: float,
        target_lon: float,
        role: str = "serve",
        power_mode: str = "normal",
    ) -> None:
        """Controller action: where a HAPS should go, and how it should run."""
        if power_mode not in ("normal", "low"):
            raise ValueError(f"unknown power_mode {power_mode!r}")
        h = next(h for h in self.haps if h.id == haps_id)
        h.target_lat, h.target_lon, h.role = target_lat, target_lon, role
        if power_mode != h.power_mode:
            h.power_mode = power_mode
            self._auto_low_power.discard(h.id)

    def observe(self) -> Observation:
        """What an operator sees: the latest tick plus platform state."""
        if self.last_result is None:
            self.last_result = self._evaluate()
        r = self.last_result
        by_id = {c.cell_id: c for c in self.cells}
        net, dyn = self.config.network, self.config.dynamics
        return Observation(
            sim_minutes=self.sim_minutes,
            cells=[
                CellObs(
                    cell_id=c.cell_id,
                    lat=by_id[c.cell_id].lat,
                    lon=by_id[c.cell_id].lon,
                    demand_mbps=c.demand_critical_mbps + c.demand_general_mbps,
                    ground_served_mbps=c.ground_served_mbps,
                    overflow_mbps=max(
                        0.0, c.demand_critical_mbps + c.demand_general_mbps - c.ground_served_mbps
                    ),
                    tower_status=by_id[c.cell_id].tower_status,
                )
                for c in r.cells
            ],
            haps=[
                HapsObs(
                    id=h.id, lat=h.lat, lon=h.lon, target_lat=h.target_lat,
                    target_lon=h.target_lon, speed_kmh=h.speed_kmh,
                    battery_pct=h.battery_pct, battery_floor_pct=h.battery_floor_pct,
                    power_mode=h.power_mode, role=h.role,
                )
                for h in self.haps
            ],
            gateways=[GatewayObs(g.id, g.lat, g.lon, g.status) for g in self.gateways],
            spec=PlatformSpec(
                coverage_km=net.haps_coverage_km,
                feeder_range_km=net.feeder_range_km,
                isl_range_km=net.isl_range_km,
                base_drain_pct_per_hour=dyn.base_drain_pct_per_hour,
                travel_drain_pct_per_km=dyn.travel_drain_pct_per_km,
            ),
        )

    def apply(self, actions: list[Action]) -> None:
        for a in actions:
            self.assign(a.haps_id, a.target_lat, a.target_lon, a.role, a.power_mode)

    def step(self, sim_time: float | str | datetime | time | None = None) -> TickResult:
        """With `sim_time`: evaluate that time of day, nothing moves.
        Without: advance one tick, moving HAPS and updating batteries."""
        if sim_time is not None:
            self.set_time(sim_time)
            if self.config.ground_model == "mesh":
                result = self._step_mesh(sim_time)
            else:
                result = self._evaluate()
        else:
            if self.config.ground_model == "mesh":
                raise NotImplementedError("time stepping is only supported with ground_model: simple")
            result = self._advance()
        self.last_result = result
        return result

    def _advance(self) -> TickResult:
        dyn = self.config.dynamics
        dt_h = dyn.tick_minutes / 60.0
        moved: dict[str, float] = {}
        for h in self.haps:
            h.lat, h.lon, moved[h.id] = move_towards(
                h.lat, h.lon, h.target_lat, h.target_lon, h.speed_kmh * dt_h
            )
        if any(km > 0.0 for km in moved.values()):
            self._refresh_haps_geometry()
        self.sim_minutes += dyn.tick_minutes
        result = self._evaluate()

        access_cap = {link.from_id: link.capacity_mbps for link in result.links if link.link_type == "access"}
        access_flow = {link.from_id: link.flow_mbps for link in result.links if link.link_type == "access"}
        mid_hour = (self.sim_minutes - dyn.tick_minutes / 2.0) / 60.0
        for h in self.haps:
            cap = access_cap.get(h.id, 0.0)
            used = access_flow.get(h.id, 0.0) / cap if cap > 0 else 0.0
            h.battery_pct = battery_after_tick(h, used, moved[h.id], mid_hour, dt_h, dyn)
            # Low power is a rule, not a controller decision: below the floor,
            # switch; once recharged past floor + margin, switch back.
            if h.battery_pct < h.battery_floor_pct and h.power_mode != "low":
                h.power_mode = "low"
                self._auto_low_power.add(h.id)
            elif (
                h.id in self._auto_low_power
                and h.battery_pct >= h.battery_floor_pct + dyn.low_power_exit_margin_pct
            ):
                h.power_mode = "normal"
                self._auto_low_power.discard(h.id)
        result.haps = [Haps(**h.to_dict()) for h in self.haps]
        return result

    def _label(self) -> str:
        m = int(round(self.sim_minutes)) % (24 * 60)
        return f"{m // 60:02d}:{m % 60:02d}"

    def _demand(self, hour: float) -> dict[str, tuple[float, float]]:
        if self.traffic is None:
            frac = active_fraction(hour, self.config.demand)
            return {c.cell_id: demand_mbps(c.population, frac, self.config.demand) for c in self.cells}
        return {
            c.cell_id: demand_mbps(
                c.population, self.traffic.active_fraction(c.cell_id, hour), self.config.demand
            )
            for c in self.cells
        }

    def _evaluate(self, label: str | None = None) -> TickResult:
        """Serve the current time's demand from the current state. No side effects."""
        hour = self.sim_minutes / 60.0
        label = label or self._label()
        demand = self._demand(hour)
        lat_cfg = self.config.latency

        overflow_c: dict[str, float] = {}
        overflow_g: dict[str, float] = {}
        demand_c: dict[str, float] = {}
        demand_g: dict[str, float] = {}
        ground_c: dict[str, float] = {}
        ground_g: dict[str, float] = {}

        for cell in self.cells:
            crit, gen = demand[cell.cell_id]
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

        residuals = self._tick_residuals()
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

        links = self._haps_links(link_flow, residuals)

        over80 = sum(1 for link in links if link.utilisation >= 0.8)
        pct_served = 100.0 * served_all / total_demand if total_demand else 100.0
        pct_crit = 100.0 * served_crit / total_crit if total_crit else 100.0
        avg_ground = lat_cfg.ground_latency_ms if ground_traffic > 1e-12 else 0.0
        avg_haps = haps_lat_num / haps_traffic if haps_traffic > 1e-12 else 0.0

        return TickResult(
            sim_time=label,
            sim_minutes=self.sim_minutes,
            haps=[Haps(**h.to_dict()) for h in self.haps],
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

    def _haps_links(
        self, link_flow: dict[tuple[str, str, str], float], orig: Residuals
    ) -> list[LinkLoad]:
        links: list[LinkLoad] = []
        for hid, cap in orig.access.items():
            flow = link_flow.get(("access", hid, hid), 0.0)
            links.append(LinkLoad("access", hid, hid, flow, cap, flow / cap if cap else 0.0))
        for (a, b), cap in orig.isl.items():
            flow = link_flow.get(("isl", a, b), 0.0)
            links.append(LinkLoad("isl", a, b, flow, cap, flow / cap if cap else 0.0))
        for (hid, gid), cap in orig.feeder.items():
            flow = link_flow.get(("feeder", hid, gid), 0.0)
            links.append(LinkLoad("feeder", hid, gid, flow, cap, flow / cap if cap else 0.0))
        return links

    def _step_mesh(self, sim_time: float | str | datetime | time) -> TickResult:
        hour, label = _parse_hour(sim_time)
        frac = active_fraction(hour, self.config.demand)
        mcfg = self.config.mesh
        status = {c.cell_id: c.tower_status for c in self.cells}

        demand_c: dict[str, float] = {}
        demand_g: dict[str, float] = {}
        for cell in self.cells:
            crit, gen = demand_mbps(cell.population, frac, self.config.demand)
            demand_c[cell.cell_id] = crit
            demand_g[cell.cell_id] = gen
        total_demand = sum(demand_c.values()) + sum(demand_g.values())
        total_crit = sum(demand_c.values())

        # Backhaul capacity this tick, per direction.
        backhaul_cap: dict[tuple[str, str], float] = {}
        for (a, b), cap in self.backhaul.capacity.items():
            if mcfg.transit_affected_by_outage:
                cap *= min(status[a], status[b])
            backhaul_cap[(a, b)] = cap
            backhaul_cap[(b, a)] = cap

        residuals = mesh.MeshResiduals(
            backhaul=dict(backhaul_cap),
            endpoint=endpoint_capacities(
                self.endpoint_weights, total_demand, mcfg.endpoint_headroom
            ),
            access={c.cell_id: c.ground_capacity_mbps * c.tower_status for c in self.cells},
            haps=self._base_residuals.copy(),
        )
        pass_c = mesh.route_mesh_pass(
            demand_c, self.coverage, self.gw_cell, self.pairs, residuals
        )
        pass_g = mesh.route_mesh_pass(
            demand_g, self.coverage, self.gw_cell, self.pairs, pass_c.residuals
        )
        cell_haps = merge_cell_haps(pass_c.cell_haps, pass_g.cell_haps)
        link_flow = merge_flows(pass_c.links, pass_g.links)
        per_cell_haps: dict[str, dict[str, float]] = {}
        for (cid, hid), mbps in cell_haps.items():
            per_cell_haps.setdefault(cid, {})[hid] = mbps

        cell_results: list[CellResult] = []
        served_all = 0.0
        served_crit = 0.0
        unserved_c = 0.0
        unserved_g = 0.0
        ground_traffic = 0.0
        ground_lat_num = 0.0
        haps_lat_num = 0.0
        haps_traffic = 0.0

        for cell in self.cells:
            cid = cell.cell_id
            d_c = demand_c[cid]
            d_g = demand_g[cid]
            g_c = pass_c.ground.get(cid, 0.0)
            g_g = pass_g.ground.get(cid, 0.0)
            h_c = pass_c.haps.get(cid, 0.0)
            h_g = pass_g.haps.get(cid, 0.0)
            u_c = max(0.0, d_c - g_c - h_c)
            u_g = max(0.0, d_g - g_g - h_g)
            g_served = g_c + g_g
            ground_ms = mcfg.ground_base_ms + mcfg.per_backhaul_hop_ms * self.ground_hops[cid]
            per_haps = per_cell_haps.get(cid, {})
            latency, hops = cell_latency_and_hops(g_served, ground_ms, per_haps, self.routes)
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
                    transit_mbps=pass_c.transit.get(cid, 0.0) + pass_g.transit.get(cid, 0.0),
                )
            )
            served_all += g_served + h_c + h_g
            served_crit += g_c + h_c
            unserved_c += u_c
            unserved_g += u_g
            ground_traffic += g_served
            ground_lat_num += g_served * ground_ms
            for hid, mbps in per_haps.items():
                route = self.routes.get(hid)
                if route is None:
                    continue
                haps_lat_num += mbps * route.latency_ms
                haps_traffic += mbps

        links = self._haps_links(link_flow, self._base_residuals)
        for (a, b), cap in backhaul_cap.items():
            flow = link_flow.get(("backhaul", a, b), 0.0)
            links.append(LinkLoad("backhaul", a, b, flow, cap, flow / cap if cap else 0.0))

        over80 = sum(1 for link in links if link.utilisation >= 0.8)
        bh_over80 = sum(
            1 for link in links if link.link_type == "backhaul" and link.utilisation >= 0.8
        )
        crossed = pass_c.crossed_backhaul_mbps + pass_g.crossed_backhaul_mbps
        return TickResult(
            sim_time=label,
            cells=cell_results,
            links=links,
            totals=Totals(
                demand_mbps=total_demand,
                pct_served=100.0 * served_all / total_demand if total_demand else 100.0,
                pct_critical_served=100.0 * served_crit / total_crit if total_crit else 100.0,
                unserved_critical_mbps=unserved_c,
                unserved_general_mbps=unserved_g,
                links_over_80pct=over80,
                avg_ground_latency_ms=(
                    ground_lat_num / ground_traffic if ground_traffic > 1e-12 else 0.0
                ),
                avg_haps_latency_ms=haps_lat_num / haps_traffic if haps_traffic > 1e-12 else 0.0,
                pct_crossing_backhaul=100.0 * crossed / served_all if served_all > 1e-12 else 0.0,
                backhaul_links_over_80pct=bh_over80,
            ),
        )
