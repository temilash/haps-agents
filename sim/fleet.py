"""Starting positions for the HAPS fleet."""

from __future__ import annotations

import sys

import numpy as np

from sim.config import NetworkConfig, PlatformConfig
from sim.controllers import _distances_km
from sim.state import Cell, Gateway


def place_fleet(
    count: int,
    listed: list[PlatformConfig],
    cells: list[Cell],
    peak_demand: dict[str, float],
    gateways: list[Gateway],
    net: NetworkConfig,
) -> list[PlatformConfig]:
    """Listed positions first; each further HAPS goes to the cell centre that
    covers the most peak demand no HAPS covers yet, among centres with a route
    to a gateway (within feeder range of one, or within ISL range of an
    already placed HAPS that is)."""
    fleet = list(listed[:count])
    if len(fleet) == count:
        return fleet
    lats = np.array([c.lat for c in cells])
    lons = np.array([c.lon for c in cells])
    demand = np.array([peak_demand[c.cell_id] for c in cells])
    near_gateway = np.zeros(len(cells), dtype=bool)
    for g in gateways:
        if g.status > 0:
            near_gateway |= _distances_km(g.lat, g.lon, lats, lons) <= net.feeder_range_km
    covers = np.array(
        [_distances_km(c.lat, c.lon, lats, lons) <= net.haps_coverage_km for c in cells]
    )

    covered = np.zeros(len(cells), dtype=bool)
    relay_ok = near_gateway.copy()  # centres from which a route exists
    for p in fleet:
        covered |= _distances_km(p.lat, p.lon, lats, lons) <= net.haps_coverage_km
        if any(
            _distances_km(p.lat, p.lon, np.array([g.lat]), np.array([g.lon]))[0] <= net.feeder_range_km
            for g in gateways
            if g.status > 0
        ):
            relay_ok |= _distances_km(p.lat, p.lon, lats, lons) <= net.isl_range_km

    used = {p.id for p in fleet}
    n = 1
    while len(fleet) < count:
        scores = covers @ np.where(covered, 0.0, demand)
        scores[~relay_ok] = -1.0
        best = int(np.argmax(scores))
        if scores[best] < 0:
            print(
                f"WARNING: placed {len(fleet)} of {count} HAPS; no position left with a "
                "route to a gateway",
                file=sys.stderr,
            )
            break
        while f"H{n}" in used:
            n += 1
        hid = f"H{n}"
        used.add(hid)
        fleet.append(PlatformConfig(id=hid, lat=float(lats[best]), lon=float(lons[best])))
        covered |= covers[best]
        if near_gateway[best]:
            relay_ok |= _distances_km(lats[best], lons[best], lats, lons) <= net.isl_range_km
    return fleet
