"""Load config.yaml into pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class GridConfig(BaseModel):
    h3_resolution: int = 5


class DemandConfig(BaseModel):
    trough_hour: float = 4.0
    peak_hour: float = 20.0
    active_fraction_trough: float = 0.02
    active_fraction_peak: float = 0.15
    critical_user_share: float = 0.05
    critical_mbps_per_user: float = 0.1
    general_user_share: float = 0.95
    general_mbps_per_user: float = 2.0
    ground_overprovision: float = 1.3


class NetworkConfig(BaseModel):
    haps_coverage_km: float = 50
    haps_access_mbps: float = 5000
    isl_range_km: float = 250
    isl_mbps: float = 10000
    feeder_range_km: float = 150
    feeder_mbps: float = 10000
    haps_count: int = 4


class LatencyConfig(BaseModel):
    ground_latency_ms: float = 20
    haps_base_ms: float = 20
    per_haps_ms: float = 3
    speed_km_per_ms: float = 300


class MeshConfig(BaseModel):
    endpoint_exponent: float = 1.5
    endpoint_headroom: float = 1.2
    backhaul_headroom: float = 1.5
    backhaul_min_mbps: float = 500
    transit_affected_by_outage: bool = True
    ground_base_ms: float = 10
    per_backhaul_hop_ms: float = 0.5
    latency_top_endpoints: int = 10
    # Hubs (used when the region has no backbone.yaml)
    hub_count: int | None = None  # None: clamp(round(area_km2 / 10000), 4, 20)
    hub_min_separation_km: float = 40
    hub_radius_km: float = 10
    big_hub_radius_km: float = 25
    hub_degree: int = 3
    big_hub_count: int = 4
    # Neighbour-link pruning
    mesh_mode: Literal["full", "pruned"] = "pruned"
    urban_share: float = 0.10
    suburban_share: float = 0.30
    suburban_extra_links: int = 2
    rural_extra_links: int = 1
    # Results and routing
    isolation_threshold: float = 0.2
    routing_algorithm: Literal["mincost", "maxflow"] = "mincost"


AREA_TYPES = ("city_centre", "nightlife", "suburban", "rural")


class AreaCurves(BaseModel):
    """Active fraction of the population at each hour 00:00..23:00."""

    city_centre: list[float] = [
        0.04, 0.03, 0.03, 0.03, 0.03, 0.04, 0.06, 0.10, 0.16, 0.20, 0.20, 0.20,
        0.23, 0.22, 0.20, 0.20, 0.19, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04,
    ]
    nightlife: list[float] = [
        0.18, 0.12, 0.06, 0.03, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
        0.09, 0.09, 0.08, 0.08, 0.09, 0.11, 0.14, 0.17, 0.20, 0.22, 0.22, 0.21,
    ]
    suburban: list[float] = [
        0.05, 0.03, 0.02, 0.02, 0.02, 0.03, 0.06, 0.12, 0.12, 0.08, 0.06, 0.06,
        0.06, 0.06, 0.06, 0.07, 0.09, 0.11, 0.13, 0.17, 0.19, 0.19, 0.16, 0.10,
    ]
    rural: list[float] = [
        0.03, 0.02, 0.015, 0.015, 0.015, 0.02, 0.04, 0.06, 0.07, 0.07, 0.07, 0.07,
        0.07, 0.07, 0.07, 0.07, 0.08, 0.09, 0.10, 0.11, 0.11, 0.10, 0.07, 0.05,
    ]


class WeekendConfig(BaseModel):
    shift_hours: float = 1.0
    business_scale: float = 0.6  # city_centre business peak cut by 40%
    business_start_hour: float = 8.0
    business_end_hour: float = 18.0


class TrafficConfig(BaseModel):
    day_type: Literal["weekday", "weekend"] = "weekday"
    # Area classification, by population-density rank within the region.
    city_centre_share: float = 0.05
    nightlife_share: float = 0.15
    suburban_share: float = 0.40
    suburban_radius_factor: float = 2.0
    curves: AreaCurves = Field(default_factory=AreaCurves)
    # Scale curves so a weekday carries the same total traffic as the single curve.
    normalise_to_single_curve: bool = True
    weekend: WeekendConfig = Field(default_factory=WeekendConfig)
    # Seeded per-cell variation.
    seed: int = 0
    time_shift_minutes: float = 30.0
    amplitude_jitter: float = 0.10
    # Ground access = the cell's own daily peak * headroom * U(1 - jitter, 1 + jitter).
    ground_headroom: dict[str, float] = Field(
        default_factory=lambda: {
            "city_centre": 1.2, "nightlife": 1.1, "suburban": 1.0, "rural": 0.7,
        }
    )
    capacity_jitter: float = 0.10


class DynamicsConfig(BaseModel):
    tick_minutes: float = 10.0
    speed_kmh: float = 100.0
    battery_start_pct: float = 80.0
    battery_floor_pct: float = 25.0
    # Leave automatic low power once the battery is back this far above the floor.
    low_power_exit_margin_pct: float = 10.0
    base_drain_pct_per_hour: float = 2.0
    serving_drain_pct_per_hour: float = 1.0  # at full use of access capacity
    travel_drain_pct_per_km: float = 0.02
    solar_charge_pct_per_hour: float = 6.0
    sunrise_hour: float = 7.5
    sunset_hour: float = 17.5
    low_power_access_factor: float = 0.5
    low_power_serving_factor: float = 0.5


class GreedyConfig(BaseModel):
    interval_minutes: float = 30.0
    horizon_minutes: float = 60.0
    hysteresis: float = 0.15
    battery_margin_pct: float = 5.0


class PlatformConfig(BaseModel):
    id: str
    lat: float
    lon: float


class AppConfig(BaseModel):
    ground_model: Literal["simple", "mesh"] = "simple"
    # single: one daily curve, ground sized for 20:00 (the original model).
    # area_types: per-area curves, ground sized for each cell's own peak.
    demand_model: Literal["single", "area_types"] = "single"
    controller: Literal["static", "greedy"] = "static"
    grid: GridConfig = Field(default_factory=GridConfig)
    demand: DemandConfig = Field(default_factory=DemandConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    traffic: TrafficConfig = Field(default_factory=TrafficConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    greedy: GreedyConfig = Field(default_factory=GreedyConfig)
    haps: list[PlatformConfig] = Field(default_factory=list)
    gateways: list[PlatformConfig] = Field(default_factory=list)


def load_config_dict(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> AppConfig:
    return AppConfig.model_validate(load_config_dict(path))


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
