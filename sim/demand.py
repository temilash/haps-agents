"""Demand, ground capacity, and overflow."""

from __future__ import annotations

import math

from sim.config import DemandConfig
from sim.grid import haversine_km
from sim.state import Cell


def active_fraction(hour: float, cfg: DemandConfig) -> float:
    """Smooth daily curve: ~trough at 04:00, ~peak at 20:00."""
    h = hour % 24.0
    lo = cfg.active_fraction_trough
    hi = cfg.active_fraction_peak
    trough_h = cfg.trough_hour
    peak_h = cfg.peak_hour

    if trough_h <= h <= peak_h:
        t = (h - trough_h) / (peak_h - trough_h)
        w = 0.5 * (1.0 - math.cos(math.pi * t))
        return lo + (hi - lo) * w

    span = (24.0 - peak_h) + trough_h
    elapsed = (h - peak_h) if h > peak_h else (24.0 - peak_h) + h
    t = elapsed / span
    w = 0.5 * (1.0 - math.cos(math.pi * t))
    return hi + (lo - hi) * w


def demand_mbps(population: float, frac: float, cfg: DemandConfig) -> tuple[float, float]:
    active_users = population * frac
    critical = active_users * cfg.critical_user_share * cfg.critical_mbps_per_user
    general = active_users * cfg.general_user_share * cfg.general_mbps_per_user
    return critical, general


def peak_ground_capacity(population: float, cfg: DemandConfig) -> float:
    critical, general = demand_mbps(population, cfg.active_fraction_peak, cfg)
    return (critical + general) * cfg.ground_overprovision


def allocate_ground(
    critical_mbps: float,
    general_mbps: float,
    usable_ground_mbps: float,
) -> tuple[float, float, float, float]:
    """Return (ground_crit, ground_gen, overflow_crit, overflow_gen)."""
    total = critical_mbps + general_mbps
    if total <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    served = min(total, max(0.0, usable_ground_mbps))
    share_c = critical_mbps / total
    ground_c = served * share_c
    ground_g = served - ground_c
    return ground_c, ground_g, critical_mbps - ground_c, general_mbps - ground_g


def set_tower_status(
    cells: list[Cell],
    lat: float,
    lon: float,
    radius_km: float,
    value: float,
) -> None:
    for cell in cells:
        if haversine_km(cell.lat, cell.lon, lat, lon) <= radius_km:
            cell.tower_status = value
