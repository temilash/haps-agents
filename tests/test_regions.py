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


# --- boundary-driven grid ---------------------------------------------------------


def test_missing_boundary_explains_where_to_get_it(tmp_path: Path):
    import pytest

    regions = _copy_region(tmp_path, "test_island")
    (regions / "test_island" / "boundary.geojson").unlink()
    with pytest.raises(FileNotFoundError, match="Countries Boundaries UK BUC"):
        load_or_build_cells(load_region("test_island", regions), tmp_path)


def test_england_feature_is_picked_from_uk_file():
    import pytest

    region = load_region("england")
    england = load_boundary(region.boundary_path, "England", "CTRY23NM")
    wales = load_boundary(region.boundary_path, "Wales", None)  # any *NM property
    assert england.contains(Point(-0.13, 51.51)) and not england.contains(Point(-3.18, 51.48))
    assert wales.contains(Point(-3.18, 51.48))
    with pytest.raises(ValueError, match="England"):
        load_boundary(region.boundary_path, "Atlantis", None)


def test_england_grid_is_inside_the_boundary(tmp_path: Path):
    region = load_region("england")
    df = load_or_build_cells(region, tmp_path)
    boundary = load_boundary(region.boundary_path, "England", "CTRY23NM")
    assert all(boundary.contains(Point(lon, lat)) for lat, lon in zip(df.lat, df.lon))
    ids = set(df.cell_id)
    for lat, lon in [(54.5, 1.0), (54.0, -4.0), (50.3, -1.5), (55.95, -3.19), (51.48, -3.18)]:
        assert h3.latlng_to_cell(lat, lon, 5) not in ids  # seas, Edinburgh, Cardiff
    area = sum(h3.cell_area(c, unit="km^2") for c in df.cell_id)
    assert 120_000 < area < 140_000  # England is ~130,600 km2
    total = sum(c.population for c in region.spec.synthetic.cities)
    assert df.population.sum() == pytest_approx(total)


def pytest_approx(x):
    import pytest

    return pytest.approx(x, rel=1e-9)


def test_population_raster_is_clipped_to_the_boundary(tmp_path: Path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    regions = _copy_region(tmp_path, "test_island")
    region = load_region("test_island", regions)
    boundary = load_boundary(region.boundary_path)
    # 100 people per 0.05-degree pixel, over the island and a sea margin.
    min_lon, min_lat, max_lon, max_lat = boundary.bounds
    step = 0.05
    west, north = min_lon - 1.0, max_lat + 1.0
    cols = int(round((max_lon + 1.0 - west) / step))
    rows = int(round((north - (min_lat - 1.0)) / step))
    with rasterio.open(
        region.population_path, "w", driver="GTiff", height=rows, width=cols, count=1,
        dtype="float32", crs="EPSG:4326", transform=from_origin(west, north, step, step), nodata=-1,
    ) as dst:
        dst.write(np.full((1, rows, cols), 100.0, dtype="float32"))

    inside = sum(
        boundary.contains(Point(west + (c + 0.5) * step, north - (r + 0.5) * step))
        for r in range(rows)
        for c in range(cols)
    )
    df = load_or_build_cells(region, tmp_path)
    assert inside < rows * cols  # the raster really does reach into the sea
    assert df.population.sum() == pytest_approx(100.0 * inside)
    assert all(boundary.contains(Point(lon, lat)) for lat, lon in zip(df.lat, df.lon))
