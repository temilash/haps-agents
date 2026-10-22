from __future__ import annotations

import shutil
from pathlib import Path

import h3
import yaml
from shapely.geometry import Point

from sim.engine import Simulation
from sim.grid import load_boundary, load_or_build_cells
from sim.region import REGIONS_DIR, load_region, region_config


def test_test_island_synthetic_population(tmp_path: Path, capsys):
    region = load_region("test_island")
    df = load_or_build_cells(region, tmp_path)
    assert "synthetic population" in capsys.readouterr().err
    boundary = load_boundary(region.boundary_path)
    assert all(boundary.contains(Point(lon, lat)) for lat, lon in zip(df.lat, df.lon))
    area = len(df) * h3.average_hexagon_area(5, unit="km^2")
    assert 30_000 < area < 45_000  # ~200 x 200 km
    assert df.population.sum() > 0
    # Seeded from the region name: rebuilding gives the same population.
    shutil.rmtree(tmp_path / "data")
    again = load_or_build_cells(region, tmp_path)
    assert again.population.tolist() == df.population.tolist()
    assert (tmp_path / "data/processed/test_island/cells.parquet").exists()


def _copy_region(tmp_path: Path, name: str, **changes) -> Path:
    regions = tmp_path / "regions"
    shutil.copytree(REGIONS_DIR / name, regions / name)
    path = regions / name / "region.yaml"
    spec = yaml.safe_load(path.read_text())
    spec.update(changes)
    path.write_text(yaml.safe_dump(spec))
    return regions


def test_region_overrides_and_assets(tmp_path: Path):
    regions = _copy_region(
        tmp_path, "england", overrides={"network": {"haps_coverage_km": 60}}
    )
    cfg = region_config(load_region("england", regions))
    assert cfg.network.haps_coverage_km == 60
    assert cfg.network.isl_mbps == 10000  # untouched defaults survive the merge
    assert [h.id for h in cfg.haps] == ["H1", "H2", "H3", "H4"]
    assert region_config(load_region("test_island")).haps == []


def test_cells_cache_rebuilt_when_region_changes(tmp_path: Path):
    regions = _copy_region(tmp_path, "test_island")
    n5 = len(load_or_build_cells(load_region("test_island", regions), tmp_path))
    _copy_region(tmp_path / "v2", "test_island", h3_resolution=4)
    n4 = len(load_or_build_cells(load_region("test_island", tmp_path / "v2/regions"), tmp_path))
    assert n4 < n5


def test_simulation_runs_on_test_island(tmp_path: Path):
    sim = Simulation(region="test_island", project_root=tmp_path)
    result = sim.step("20:00")
    assert len(result.cells) == len(sim.cells)
    assert 0.0 < result.totals.pct_served <= 100.0
