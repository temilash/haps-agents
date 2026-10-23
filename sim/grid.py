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


BOUNDARY_HELP = """\
Region boundary not found: {path}

Every region needs a boundary; the grid is built from it so no cell lies in the sea.
For England, download the ONS "Countries (December 2023) Boundaries UK BUC" GeoJSON:
  https://geoportal.statistics.gov.uk/  (search "Countries Boundaries UK BUC", Download > GeoJSON)
or directly:
  https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Countries_December_2023_Boundaries_UK_BUC/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson
Save it as {path}. It holds all four UK countries; region.yaml's
boundary_feature_name (e.g. England) picks one."""


def _feature_names(props: dict, name_property: str | None) -> list[str]:
    if name_property is not None:
        return [str(props.get(name_property))]
    return [
        str(v)
        for k, v in props.items()
        if isinstance(v, str) and (k.upper().endswith("NM") or k.lower() == "name")
    ]


def load_boundary(
    path: Path,
    feature_name: str | list[str] | None = None,
    name_property: str | None = None,
):
    """The region's outline: the named feature(s) merged, or all of them for
    None / "all"."""
    if not path.exists():
        raise FileNotFoundError(BOUNDARY_HELP.format(path=path))
    with path.open() as f:
        geo = json.load(f)
    if geo.get("type") == "FeatureCollection":
        features = [ft for ft in geo["features"] if ft.get("geometry")]
    elif geo.get("type") == "Feature":
        features = [geo]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": geo}]
    if feature_name is not None and feature_name != "all":
        wanted = [feature_name] if isinstance(feature_name, str) else list(feature_name)
        names = {
            id(ft): _feature_names(ft.get("properties") or {}, name_property) for ft in features
        }
        missing = [w for w in wanted if not any(w in names[id(ft)] for ft in features)]
        if missing:
            found = sorted({n for ns in names.values() for n in ns})
            raise ValueError(f"No feature named {missing} in {path} (found: {found})")
        features = [ft for ft in features if any(w in names[id(ft)] for w in wanted)]
    geom = unary_union([shape(ft["geometry"]) for ft in features])
    return geom if geom.is_valid else geom.buffer(0)


def boundary_cells(boundary, resolution: int) -> list[str]:
    """Cells whose centre is inside the boundary (h3 polygon_to_cells)."""
    shape_ = h3.geo_to_h3shape(boundary.__geo_interface__)
    return sorted(h3.polygon_to_cells(shape_, resolution))


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


def raster_cells(boundary, raster_path: Path, resolution: int) -> pd.DataFrame:
    """Population per boundary cell from a raster clipped to the boundary.

    Pixels outside the boundary are dropped. A pixel inside the boundary can
    still fall in a coastal cell whose centre is offshore (so not in the grid);
    its population goes to the nearest grid cell, so none is lost.
    """
    import rasterio
    import rasterio.mask
    from rasterio.warp import transform as warp_transform
    from rasterio.warp import transform_geom

    cell_ids = boundary_cells(boundary, resolution)
    with rasterio.open(raster_path) as src:
        geom = boundary.__geo_interface__
        if src.crs and src.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            geom = transform_geom("EPSG:4326", src.crs, geom)
        data, transform = rasterio.mask.mask(src, [geom], crop=True, filled=False)
        band = data[0]
        mask = ~np.ma.getmaskarray(band)
        values = np.ma.getdata(band).astype(float)
        mask &= np.isfinite(values) & (values > 0)
        if src.nodata is not None:
            mask &= values != src.nodata
        rows, cols = np.where(mask)
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        if src.crs and src.crs.to_string() not in ("EPSG:4326", "OGC:CRS84"):
            xs, ys = warp_transform(src.crs, "EPSG:4326", xs, ys)
        lons = np.asarray(xs, dtype=float)
        lats = np.asarray(ys, dtype=float)
        values = values[rows, cols]

    pixel_cells = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(lats, lons)]
    per_cell = pd.Series(values).groupby(pd.Series(pixel_cells)).sum()
    in_grid = set(cell_ids)
    centres = {cid: h3.cell_to_latlng(cid) for cid in cell_ids}
    totals = dict.fromkeys(cell_ids, 0.0)
    moved = 0.0
    for cid, pop in per_cell.items():
        if cid not in in_grid:
            lat, lon = h3.cell_to_latlng(cid)
            cid = min(centres, key=lambda c: haversine_km(lat, lon, *centres[c]))
            moved += pop
        totals[cid] += pop
    total = float(values.sum())
    print(
        f"  raster clipped to boundary: {total:,.0f} people; "
        f"{moved:,.0f} ({100 * moved / total if total else 0:.1f}%) in offshore-centred "
        "coastal cells moved to the nearest land cell",
        file=sys.stderr,
    )
    return _cell_table(cell_ids, np.array([totals[c] for c in cell_ids]))


def landmasses(cell_ids: list[str]) -> list[set[str]]:
    """Groups of cells joined by H3 neighbour links, biggest first."""
    import networkx as nx

    ids = set(cell_ids)
    G = nx.Graph()
    G.add_nodes_from(cell_ids)
    for cid in cell_ids:
        G.add_edges_from((cid, nb) for nb in h3.grid_ring(cid, 1) if nb in ids)
    return sorted(nx.connected_components(G), key=lambda c: (-len(c), min(c)))


def drop_small_components(df: pd.DataFrame, min_population: float) -> pd.DataFrame:
    pop = dict(zip(df.cell_id, df.population))
    dropped = []
    for comp in landmasses(list(df.cell_id)):
        total = sum(pop[c] for c in comp)
        if total < min_population:
            lat, lon = h3.cell_to_latlng(min(comp))
            dropped.append((len(comp), total, lat, lon))
    if not dropped:
        return df
    for n, total, lat, lon in dropped:
        print(
            f"  dropped landmass of {n} cell(s), population {total:,.0f}, near ({lat:.2f}, {lon:.2f})",
            file=sys.stderr,
        )
    keep = set().union(*(c for c in landmasses(list(df.cell_id)) if sum(pop[x] for x in c) >= min_population))
    return df[df.cell_id.isin(keep)].reset_index(drop=True)


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

    spec = region.spec
    boundary = load_boundary(
        region.boundary_path, spec.boundary_feature_name, spec.boundary_name_property
    )
    resolution = spec.h3_resolution
    if region.population_path.exists():
        df = raster_cells(boundary, region.population_path, resolution)
        source = f"from {region.population_path.name}"
    else:
        print(
            f"WARNING: {region.population_path} missing; "
            f"using a synthetic population for region {region.name!r}.",
            file=sys.stderr,
        )
        df = synthetic_cells(boundary, resolution, spec.synthetic, region.name)
        source = "synthetic"
    df = drop_small_components(df, spec.min_component_population)
    print(
        f"Region {region.name}: {len(df)} cells inside the boundary, "
        f"population {df['population'].sum() / 1e6:.2f} million ({source})",
        file=sys.stderr,
    )

    if write:
        write_keyed(df, cells_path, key)
    return df
