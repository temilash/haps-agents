"""HAPS movement, daylight and battery."""

from __future__ import annotations

import math

from sim.config import DynamicsConfig
from sim.grid import haversine_km
from sim.state import Haps


def move_towards(lat: float, lon: float, tlat: float, tlon: float, max_km: float) -> tuple[float, float, float]:
    """Up to max_km along the great circle to the target: (lat, lon, km moved)."""
    d = haversine_km(lat, lon, tlat, tlon)
    if d <= max_km or d <= 1e-9:
        return tlat, tlon, d
    f = max_km / d
    p1, l1, p2, l2 = map(math.radians, (lat, lon, tlat, tlon))
    delta = d / 6371.0  # angular distance, same Earth radius as haversine_km
    a = math.sin((1 - f) * delta) / math.sin(delta)
    b = math.sin(f * delta) / math.sin(delta)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    return (
        math.degrees(math.atan2(z, math.hypot(x, y))),
        math.degrees(math.atan2(y, x)),
        max_km,
    )


def daylight_factor(hour: float, cfg: DynamicsConfig) -> float:
    """0 at night, rising smoothly to 1 midway between sunrise and sunset."""
    h = hour % 24.0
    if not cfg.sunrise_hour < h < cfg.sunset_hour:
        return 0.0
    return math.sin(math.pi * (h - cfg.sunrise_hour) / (cfg.sunset_hour - cfg.sunrise_hour)) ** 2


def battery_after_tick(
    haps: Haps,
    access_used_share: float,
    km_moved: float,
    mid_tick_hour: float,
    tick_hours: float,
    cfg: DynamicsConfig,
) -> float:
    serving = cfg.serving_drain_pct_per_hour * access_used_share
    if haps.power_mode == "low":
        serving *= cfg.low_power_serving_factor
    drain = (cfg.base_drain_pct_per_hour + serving) * tick_hours
    drain += cfg.travel_drain_pct_per_km * km_moved
    charge = cfg.solar_charge_pct_per_hour * daylight_factor(mid_tick_hour, cfg) * tick_hours
    return min(100.0, max(0.0, haps.battery_pct - drain + charge))
