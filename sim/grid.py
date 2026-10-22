"""H3 hex grid and population per cell."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from sim.cache import file_digest, make_key, read_keyed, write_keyed
from sim.region import Region, SyntheticSpec

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def _cell_table(cell_ids: list[str], population: np.ndarray) -> pd.DataFrame:
    lats = []
    lons = []
    for cid in cell_ids:
        lat, lon = h3.cell_to_latlng(cid)
        lats.append(float(lat))
        lons.append(float(lon))
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "lat": lats,
            "lon": lons,
            "population": np.asarray(population, dtype=float),
        }
    )


def load_boundary(path: Path):
    with path.open() as f:
        geo = json.load(f)
    if geo.get("type") == "FeatureCollection":
        geoms = [shape(ft["geometry"]) for ft in geo["features"] if ft.get("geometry")]
        return unary_union(geoms)
    if geo.get("type") == "Feature":
        return shape(geo["geometry"])
    return shape(geo)


def boundary_cells(boundary, resolution: int) -> list[str]:
    return sorted(h3.geo_to_cells(boundary, resolution))


def _seeded_centres(
    boundary, seed_name: str, n: int, total_population: float
) -> list[tuple[float, float, float]]:
    """Random (lat, lon, population) cluster centres inside the boundary.

    Seeded from the region name, so a region always gets the same population.
    Cluster sizes fall off as 1/rank, like city sizes.
    """
    seed = int(hashlib.sha256(seed_name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    min_lon, min_lat, max_lon, max_lat = boundary.bounds
    points: list[tuple[float, float]] = []
    while len(points) < n:
        lat = rng.uniform(min_lat, max_lat)
        lon = rng.uniform(min_lon, max_lon)
        if boundary.contains(Point(lon, lat)):
            points.append((lat, lon))
    shares = 1.0 / np.arange(1, n + 1)
    shares /= shares.sum()
    return [(lat, lon, total_population * w) for (lat, lon), w in zip(points, shares)]


def synthetic_cells(
    boundary, resolution: int, spec: SyntheticSpec, seed_name: str
) -> pd.DataFrame:
    """Gaussian population blobs at the spec's cities, or at seeded centres."""
    cell_ids = boundary_cells(boundary, resolution)
    lats = np.empty(len(cell_ids))
    lons = np.empty(len(cell_ids))
    for i, cid in enumerate(cell_ids):
        lat, lon = h3.cell_to_latlng(cid)
        lats[i] = lat
        lons[i] = lon

    if spec.cities:
        centres = [(c.lat, c.lon, c.population) for c in spec.cities]
    else:
        area = len(cell_ids) * h3.average_hexagon_area(resolution, unit="km^2")
        n = spec.clusters or min(30, max(3, round(area / 4000)))
        centres = _seeded_centres(boundary, seed_name, n, spec.density_per_km2 * area)

    pop = np.zeros(len(cell_ids), dtype=float)
    for clat, clon, cpop in centres:
        dist = np.array(
            [haversine_km(lat, lon, clat, clon) for lat, lon in zip(lats, lons)]
        )
        weights = np.exp(-0.5 * (dist / spec.sigma_km) ** 2)
        total = weights.sum()
        if total > 0:
            pop += cpop * (weights / total)
    return _cell_table(cell_ids, pop)


def cells_from_worldpop(boundary, raster_path: Path, resolution: int) -> pd.DataFrame:
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        nodata = src.nodata
        mask = np.isfinite(data) & (data > 0)
        if nodata is not None:
            mask &= data != nodata
        rows, cols = np.where(mask)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        if src.crs and src.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            xs, ys = warp_transform(src.crs, "EPSG:4326", xs, ys)
        lons = np.asarray(xs, dtype=float)
        lats = np.asarray(ys, dtype=float)
        values = data[rows, cols].astype(float)

    cell_ids = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(lats, lons)]
    grouped = (
        pd.DataFrame({"cell_id": cell_ids, "population": values})
        .groupby("cell_id", as_index=False)["population"]
        .sum()
    )
    latlon = [h3.cell_to_latlng(cid) for cid in grouped["cell_id"]]
    grouped["lat"] = [p[0] for p in latlon]
    grouped["lon"] = [p[1] for p in latlon]
    inside = [
        boundary.contains(Point(lon, lat)) or boundary.touches(Point(lon, lat))
        for lat, lon in zip(grouped["lat"], grouped["lon"])
    ]
    return grouped.loc[inside, ["cell_id", "lat", "lon", "population"]].reset_index(drop=True)


def load_or_build_cells(region: Region, root: Path, *, write: bool = True) -> pd.DataFrame:
    """Cells for a region, cached in data/processed/<region>/cells.parquet."""
    cells_path = region.processed_dir(root) / "cells.parquet"
    key = make_key(
        {
            "spec": region.spec.model_dump(exclude={"overrides", "demo"}),
            "boundary": file_digest(region.boundary_path),
            "population": file_digest(region.population_path),
        }
    )
    cached = read_keyed(cells_path, key)
    if cached is not None:
        return cached

    boundary = load_boundary(region.boundary_path)
    resolution = region.spec.h3_resolution
    if region.population_path.exists():
        df = cells_from_worldpop(boundary, region.population_path, resolution)
    else:
        print(
            f"WARNING: {region.population_path} missing; "
            f"using a synthetic population for region {region.name!r}.",
            file=sys.stderr,
        )
        df = synthetic_cells(boundary, resolution, region.spec.synthetic, region.name)

    if write:
        write_keyed(df, cells_path, key)
    return df
