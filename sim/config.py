"""Load config.yaml into pydantic models."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GridConfig(BaseModel):
    h3_resolution: int = 5
    cells_path: str = "data/processed/cells.parquet"
    boundary_path: str = "data/raw/england.geojson"
    population_path: str = "data/raw/population.tif"


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


class LatencyConfig(BaseModel):
    ground_latency_ms: float = 20
    haps_base_ms: float = 20
    per_haps_ms: float = 3
    speed_km_per_ms: float = 300


class PlatformConfig(BaseModel):
    id: str
    lat: float
    lon: float


class AppConfig(BaseModel):
    grid: GridConfig = Field(default_factory=GridConfig)
    demand: DemandConfig = Field(default_factory=DemandConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    haps: list[PlatformConfig]
    gateways: list[PlatformConfig]


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return AppConfig.model_validate(data)
