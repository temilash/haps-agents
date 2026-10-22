"""HAPS controllers. They see only `Observation`s and act only through `Action`s,
which the engine applies with Simulation.assign()."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.grid import EARTH_RADIUS_KM


@dataclass
class CellObs:
    cell_id: str
    lat: float
    lon: float
    demand_mbps: float
    ground_served_mbps: float
    overflow_mbps: float  # demand the ground network couldn't carry
    tower_status: float


@dataclass
class HapsObs:
    id: str
    lat: float
    lon: float
    target_lat: float
    target_lon: float
    speed_kmh: float
    battery_pct: float
    battery_floor_pct: float
    power_mode: str
    role: str


@dataclass
class GatewayObs:
    id: str
    lat: float
    lon: float
    status: float  # health: 1 up, 0 down


@dataclass
class PlatformSpec:
    """What an operator knows about the platforms."""

    coverage_km: float
    feeder_range_km: float
    isl_range_km: float
    base_drain_pct_per_hour: float
    travel_drain_pct_per_km: float


@dataclass
class Observation:
    sim_minutes: float
    cells: list[CellObs]
    haps: list[HapsObs]
    gateways: list[GatewayObs]
    spec: PlatformSpec


@dataclass
class Action:
    haps_id: str
    target_lat: float
    target_lon: float
    role: str = "serve"
    power_mode: str = "normal"


class Controller:
    name = "base"

    def decide(self, observation: Observation) -> list[Action]:
        raise NotImplementedError


class StaticController(Controller):
    name = "static"

    def decide(self, observation: Observation) -> list[Action]:
        return []


def _distances_km(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Haversine from one point to many, matching sim.grid.haversine_km."""
    p1, l1 = np.radians(lat), np.radians(lon)
    p2, l2 = np.radians(lats), np.radians(lons)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def has_gateway_route(
    lat: float,
    lon: float,
    gateways: list[GatewayObs],
    relays: list[tuple[float, float]],
    spec: PlatformSpec,
) -> bool:
    """Within feeder range of a working gateway, or within ISL range of a relay
    position that is."""
    def feeds(la: float, lo: float) -> bool:
        return any(
            g.status > 0 and _distances_km(la, lo, np.array([g.lat]), np.array([g.lon]))[0]
            <= spec.feeder_range_km
            for g in gateways
        )

    if feeds(lat, lon):
        return True
    return any(
        _distances_km(lat, lon, np.array([r[0]]), np.array([r[1]]))[0] <= spec.isl_range_km
        and feeds(*r)
        for r in relays
    )


@dataclass
class GreedyController(Controller):
    """Every `interval_minutes`, send each HAPS (highest battery first) to the
    cell centre covering the most overflow not already claimed this round."""

    interval_minutes: float = 30.0
    horizon_minutes: float = 60.0
    hysteresis: float = 0.15
    battery_margin_pct: float = 5.0
    name: str = "greedy"
    _last_run: float | None = field(default=None, init=False, repr=False)
    # Positions considered in the last round, for tests and inspection.
    last_candidates: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def decide(self, observation: Observation) -> list[Action]:
        obs = observation
        if self._last_run is not None and obs.sim_minutes - self._last_run < self.interval_minutes - 1e-9:
            return []
        self._last_run = obs.sim_minutes
        spec = obs.spec
        lats = np.array([c.lat for c in obs.cells])
        lons = np.array([c.lon for c in obs.cells])
        unclaimed = np.array([max(0.0, c.overflow_mbps) for c in obs.cells])

        def covered(lat: float, lon: float) -> np.ndarray:
            return _distances_km(lat, lon, lats, lons) <= spec.coverage_km

        # Where each HAPS will be: targets for those already decided this round.
        planned = {h.id: (h.target_lat, h.target_lon) for h in obs.haps}
        actions: list[Action] = []
        self.last_candidates = {}
        for h in sorted(obs.haps, key=lambda h: (-h.battery_pct, h.id)):
            relays = [pos for hid, pos in planned.items() if hid != h.id]
            reach_km = h.speed_kmh * self.horizon_minutes / 60.0
            trip_km = _distances_km(h.lat, h.lon, lats, lons)
            trip_h = trip_km / h.speed_kmh if h.speed_kmh > 0 else np.inf
            battery_after = (
                h.battery_pct
                - spec.travel_drain_pct_per_km * trip_km
                - spec.base_drain_pct_per_hour * trip_h
            )
            ok = (trip_km <= reach_km) & (
                battery_after > h.battery_floor_pct + self.battery_margin_pct
            )
            for i in np.flatnonzero(ok):
                if not has_gateway_route(lats[i], lons[i], obs.gateways, relays, spec):
                    ok[i] = False
            self.last_candidates[h.id] = int(ok.sum())

            # Staying on the current course is the baseline to beat.
            stay = (h.target_lat, h.target_lon)
            stay_score = float(unclaimed[covered(*stay)].sum())
            best_score, best = stay_score, stay
            for i in np.flatnonzero(ok):
                score = float(unclaimed[covered(lats[i], lons[i])].sum())
                if score > best_score:
                    best_score, best = score, (float(lats[i]), float(lons[i]))
            if best != stay and best_score > stay_score * (1.0 + self.hysteresis):
                actions.append(Action(h.id, best[0], best[1], h.role, h.power_mode))
                planned[h.id] = best
            else:
                best = stay
            unclaimed[covered(*best)] = 0.0
        return actions


def make_controller(name: str, greedy_cfg=None) -> Controller:
    if name == "static":
        return StaticController()
    if name == "greedy":
        kw = greedy_cfg.model_dump() if greedy_cfg is not None else {}
        return GreedyController(**kw)
    raise ValueError(f"unknown controller {name!r}")
