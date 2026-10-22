"""H3 hex grid and population per cell."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from sim.config import GridConfig

EARTH_RADIUS_KM = 6371.0

# Rough resident populations used only for the synthetic fallback blobs.
CITIES = [
    ("London", 51.5074, -0.1278, 9_000_000),
    ("Birmingham", 52.4862, -1.8904, 1_140_000),
    ("Manchester", 53.4808, -2.2426, 550_000),
    ("Leeds", 53.8008, -1.5491, 790_000),
    ("Liverpool", 53.4084, -2.9916, 500_000),
    ("Newcastle", 54.9783, -1.6178, 300_000),
    ("Sheffield", 53.3811, -1.4701, 580_000),
    ("Bristol", 51.4545, -2.5879, 470_000),
    ("Nottingham", 52.9548, -1.1581, 330_000),
    ("Leicester", 52.6369, -1.1398, 370_000),
    ("Southampton", 50.9097, -1.4044, 250_000),
    ("Plymouth", 50.3755, -4.1427, 260_000),
    ("Norwich", 52.6309, 1.2974, 140_000),
    ("Carlisle", 54.892, -2.932, 75_000),
    ("Hull", 53.7676, -0.3274, 260_000),
]

BLOB_SIGMA_KM = 30.0
ENGLAND_LAT = (49.9, 55.8)
ENGLAND_LON = (-5.8, 1.8)


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


def synthetic_cells(resolution: int = 5) -> pd.DataFrame:
    poly = h3.LatLngPoly(
        [
            (ENGLAND_LAT[0], ENGLAND_LON[0]),
            (ENGLAND_LAT[0], ENGLAND_LON[1]),
            (ENGLAND_LAT[1], ENGLAND_LON[1]),
            (ENGLAND_LAT[1], ENGLAND_LON[0]),
        ]
    )
    cell_ids = sorted(h3.polygon_to_cells(poly, resolution))
    lats = np.empty(len(cell_ids))
    lons = np.empty(len(cell_ids))
    for i, cid in enumerate(cell_ids):
        lat, lon = h3.cell_to_latlng(cid)
        lats[i] = lat
        lons[i] = lon

    pop = np.zeros(len(cell_ids), dtype=float)
    for _name, clat, clon, cpop in CITIES:
        dist = np.array(
            [haversine_km(lat, lon, clat, clon) for lat, lon in zip(lats, lons)]
        )
        weights = np.exp(-0.5 * (dist / BLOB_SIGMA_KM) ** 2)
        total = weights.sum()
        if total > 0:
            pop += cpop * (weights / total)
    return _cell_table(cell_ids, pop)


def _load_boundary(path: Path):
    with path.open() as f:
        geo = json.load(f)
    if geo.get("type") == "FeatureCollection":
        geoms = [shape(ft["geometry"]) for ft in geo["features"] if ft.get("geometry")]
        return unary_union(geoms)
    if geo.get("type") == "Feature":
        return shape(geo["geometry"])
    return shape(geo)


def cells_from_worldpop(boundary_path: Path, raster_path: Path, resolution: int) -> pd.DataFrame:
    import rasterio
    from rasterio.warp import transform as warp_transform

    boundary = _load_boundary(boundary_path)
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


def load_or_build_cells(
    grid: GridConfig,
    root: Path,
    *,
    write: bool = True,
) -> pd.DataFrame:
    cells_path = root / grid.cells_path
    if cells_path.exists():
        return pd.read_parquet(cells_path)

    boundary = root / grid.boundary_path
    raster = root / grid.population_path
    if boundary.exists() and raster.exists():
        df = cells_from_worldpop(boundary, raster, grid.h3_resolution)
    else:
        print(
            "WARNING: data/raw/england.geojson or data/raw/population.tif missing; "
            "using synthetic fallback grid.",
            file=sys.stderr,
        )
        df = synthetic_cells(grid.h3_resolution)

    if write:
        cells_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cells_path, index=False)
    return df
