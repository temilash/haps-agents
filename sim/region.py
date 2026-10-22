"""Regions: everything place-specific lives in regions/<name>/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from sim.config import AppConfig, deep_merge, load_config_dict

ROOT = Path(__file__).resolve().parents[1]
REGIONS_DIR = ROOT / "regions"
DEFAULT_REGION = "england"


class City(BaseModel):
    name: str
    lat: float
    lon: float
    population: float


class SyntheticSpec(BaseModel):
    """Population used when the region has no population raster."""

    sigma_km: float = 30.0
    cities: list[City] = Field(default_factory=list)
    # Used only without `cities`: cluster centres are seeded from the region name.
    clusters: int | None = None
    density_per_km2: float = 300.0


class DemoOutage(BaseModel):
    name: str
    lat: float
    lon: float
    radius_km: float = 15.0
    tower_status: float = 0.1


class DemoSpec(BaseModel):
    simple_outage: DemoOutage | None = None
    outages: list[DemoOutage] = Field(default_factory=list)


class RegionSpec(BaseModel):
    name: str
    h3_resolution: int = 5
    boundary: str = "boundary.geojson"
    population: str = "population.tif"
    overrides: dict = Field(default_factory=dict)
    synthetic: SyntheticSpec = Field(default_factory=SyntheticSpec)
    demo: DemoSpec = Field(default_factory=DemoSpec)


@dataclass
class Region:
    name: str
    dir: Path
    spec: RegionSpec
    backbone: dict | None
    assets: dict | None

    @property
    def boundary_path(self) -> Path:
        return self.dir / self.spec.boundary

    @property
    def population_path(self) -> Path:
        return self.dir / self.spec.population

    def processed_dir(self, root: Path) -> Path:
        return root / "data" / "processed" / self.name


def _optional_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_region(name: str = DEFAULT_REGION, regions_dir: Path = REGIONS_DIR) -> Region:
    rdir = regions_dir / name
    path = rdir / "region.yaml"
    if not path.exists():
        known = sorted(p.name for p in regions_dir.iterdir() if (p / "region.yaml").exists())
        raise FileNotFoundError(f"No region {name!r} in {regions_dir} (have: {known})")
    spec = RegionSpec.model_validate(_optional_yaml(path))
    return Region(
        name=spec.name,
        dir=rdir,
        spec=spec,
        backbone=_optional_yaml(rdir / "backbone.yaml"),
        assets=_optional_yaml(rdir / "assets.yaml"),
    )


def region_config(region: Region, base_path: Path = ROOT / "config.yaml") -> AppConfig:
    """config.yaml, then the region's overrides, resolution and assets."""
    data = deep_merge(load_config_dict(base_path), region.spec.overrides)
    data = deep_merge(data, {"grid": {"h3_resolution": region.spec.h3_resolution}})
    if region.assets is not None:
        data["haps"] = region.assets.get("haps", [])
        data["gateways"] = region.assets.get("gateways", [])
    return AppConfig.model_validate(data)
