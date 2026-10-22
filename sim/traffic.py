"""Area types and per-cell daily demand curves (demand_model: area_types)."""

from __future__ import annotations

import hashlib
import math

import h3
import numpy as np

from sim.config import AREA_TYPES, DemandConfig, TrafficConfig
from sim.demand import active_fraction, demand_mbps
from sim.grid import haversine_km
from sim.state import Cell
from sim.topology import Hub

CITY_CENTRE, NIGHTLIFE, SUBURBAN, RURAL = AREA_TYPES
# Grid used to integrate curves and find their peaks.
_GRID_HOURS = np.arange(0.0, 24.0, 1.0 / 12.0)


def classify_areas(
    cells: list[Cell],
    hubs: list[Hub],
    metro: dict[str, tuple[str, float]],
    cfg: TrafficConfig,
) -> dict[str, str]:
    """By population-density rank within the region and distance to the hubs.

    "Within a hub's radius" means being one of the hub's city cells.
    """
    density = {c.cell_id: c.population / h3.cell_area(c.cell_id, unit="km^2") for c in cells}
    ranked = sorted(cells, key=lambda c: (-density[c.cell_id], c.cell_id))
    n = len(ranked)
    out: dict[str, str] = {}
    for i, c in enumerate(ranked):
        top = i / n  # 0 = densest
        in_radius = c.cell_id in metro
        near = any(
            haversine_km(c.lat, c.lon, h.lat, h.lon) <= cfg.suburban_radius_factor * h.radius_km
            for h in hubs
        )
        if in_radius and top < cfg.city_centre_share:
            out[c.cell_id] = CITY_CENTRE
        elif in_radius and top < cfg.nightlife_share:
            out[c.cell_id] = NIGHTLIFE
        elif near or top < cfg.suburban_share:
            out[c.cell_id] = SUBURBAN
        else:
            out[c.cell_id] = RURAL
    return out


def curve_at(values: list[float], hour: float) -> float:
    """Periodic Catmull-Rom through hourly values; never negative."""
    n = len(values)
    h = (hour % 24.0) * n / 24.0
    i = int(math.floor(h))
    t = h - i
    p0, p1, p2, p3 = (values[(i + k) % n] for k in (-1, 0, 1, 2))
    v = 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
    )
    return max(0.0, v)


def day_curves(cfg: TrafficConfig, day_type: str) -> dict[str, tuple[list[float], float]]:
    """Hourly values and time shift (hours) for each area type on this day type."""
    curves = {t: list(getattr(cfg.curves, t)) for t in AREA_TYPES}
    if day_type == "weekday":
        return {t: (v, 0.0) for t, v in curves.items()}
    w = cfg.weekend
    cc = curves[CITY_CENTRE]
    lo = min(cc)
    for hour in range(len(cc)):
        if w.business_start_hour <= hour <= w.business_end_hour:
            cc[hour] = lo + w.business_scale * (cc[hour] - lo)
    return {t: (v, w.shift_hours) for t, v in curves.items()}


def _cell_variation(cell_id: str, seed: int) -> tuple[float, float, float]:
    """Three U(-1, 1) draws, fixed per cell and seed."""
    digest = hashlib.sha256(f"{seed}:{cell_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    u = rng.uniform(-1.0, 1.0, size=3)
    return float(u[0]), float(u[1]), float(u[2])


class TrafficModel:
    def __init__(
        self,
        cells: list[Cell],
        area_type: dict[str, str],
        cfg: TrafficConfig,
        demand_cfg: DemandConfig,
    ) -> None:
        self.cfg = cfg
        self.demand_cfg = demand_cfg
        self.area_type = area_type
        self.curves = day_curves(cfg, cfg.day_type)
        self.shift_h: dict[str, float] = {}
        self.amplitude: dict[str, float] = {}
        self.capacity_factor: dict[str, float] = {}
        for c in cells:
            u_shift, u_amp, u_cap = _cell_variation(c.cell_id, cfg.seed)
            self.shift_h[c.cell_id] = u_shift * cfg.time_shift_minutes / 60.0
            self.amplitude[c.cell_id] = 1.0 + u_amp * cfg.amplitude_jitter
            self.capacity_factor[c.cell_id] = 1.0 + u_cap * cfg.capacity_jitter

        curve_max = {
            t: max(curve_at(v, h) for h in _GRID_HOURS) for t, (v, _s) in self.curves.items()
        }
        self.scale = self._normalisation(cells) if cfg.normalise_to_single_curve else 1.0
        self.peak_fraction = {
            c.cell_id: self.scale * self.amplitude[c.cell_id] * curve_max[area_type[c.cell_id]]
            for c in cells
        }

    def _normalisation(self, cells: list[Cell]) -> float:
        """Scale so a weekday carries the single curve's total traffic.

        A curve's daily mean doesn't depend on its time shift, so the per-cell
        shifts drop out; the per-cell amplitudes don't.
        """
        weekday = day_curves(self.cfg, "weekday")
        mean = {t: float(np.mean([curve_at(v, h) for h in _GRID_HOURS])) for t, (v, _s) in weekday.items()}
        single = float(np.mean([active_fraction(h, self.demand_cfg) for h in _GRID_HOURS]))
        target = sum(c.population for c in cells) * single
        actual = sum(
            c.population * self.amplitude[c.cell_id] * mean[self.area_type[c.cell_id]]
            for c in cells
        )
        return target / actual if actual > 0 else 1.0

    def active_fraction(self, cell_id: str, hour: float) -> float:
        values, day_shift = self.curves[self.area_type[cell_id]]
        h = hour - day_shift - self.shift_h[cell_id]
        return self.scale * self.amplitude[cell_id] * curve_at(values, h)

    def ground_capacity(self, cell: Cell) -> float:
        """The cell's own daily peak demand * area headroom * a seeded factor."""
        crit, gen = demand_mbps(cell.population, self.peak_fraction[cell.cell_id], self.demand_cfg)
        headroom = self.cfg.ground_headroom[self.area_type[cell.cell_id]]
        return (crit + gen) * headroom * self.capacity_factor[cell.cell_id]
