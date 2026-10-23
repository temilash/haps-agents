"""Compact map data shared by the replay export and the live server."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import h3

from sim.engine import Simulation


def sig(x: float, digits: int = 3) -> float:
    """Round to significant figures, keeping files small."""
    if x == 0 or not math.isfinite(x):
        return 0
    v = round(x, -int(math.floor(math.log10(abs(x)))) + digits - 1)
    return int(v) if v == int(v) else v


def write_json(path: Path, data) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, separators=(",", ":"))
    path.write_text(text)
    return len(text)


def cells_geojson(sim: Simulation) -> dict:
    """One polygon per cell, with population, area type and hub."""
    hub_of = {cid: hub for cid, (hub, _km) in sim.topology.metro.items()}
    area = getattr(sim, "area_type", {})
    features = []
    for c in sim.cells:
        ring = [[round(lng, 5), round(lat, 5)] for lat, lng in h3.cell_to_boundary(c.cell_id)]
        ring.append(ring[0])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "cell_id": c.cell_id,
                    "population": round(c.population),
                    "area_type": area.get(c.cell_id),
                    "hub": hub_of.get(c.cell_id),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def gateways_info(sim: Simulation) -> list[dict]:
    active = {g.id for g in sim._active_gateways()}
    return [
        {"id": g.id, "lat": round(g.lat, 4), "lon": round(g.lon, 4), "active": g.id in active}
        for g in sim.gateways
    ]


def static_info(sim: Simulation) -> dict:
    """Hubs, backbone and subsea links, gateways."""
    topo = sim.topology
    pos = {h.name: [round(h.lat, 4), round(h.lon, 4)] for h in topo.hubs}
    cell_pos = {c.cell_id: [round(c.lat, 4), round(c.lon, 4)] for c in sim.cells}

    def node_pos(node: str) -> list[float]:
        kind, name = node.split(":", 1)
        return pos[name] if kind == "pop" else cell_pos[name]

    return {
        "region": sim.region.name if sim.region is not None else None,
        "coverage_km": sim.config.network.haps_coverage_km,
        "gateways": gateways_info(sim),
        "hubs": [
            {"name": h.name, "lat": round(h.lat, 4), "lon": round(h.lon, 4), "radius_km": h.radius_km}
            for h in topo.hubs
        ],
        "backbone": [[pos[a], pos[b]] for a, b in topo.backbone],
        "subsea": [
            {"id": sid, "a": node_pos(a), "b": node_pos(b)}
            for sid, (a, b) in sorted(topo.subsea_ids().items())
        ],
    }


def compact_frame(record: dict, gateways: dict[str, list[float]]) -> dict:
    """One tick (TickResult.to_dict()) as a map frame.

    cells: {cell_id: [demand, overflow, % served, unserved critical, isolated]}
    links: HAPS links only (isl, feeder), each ISL pair once.
    """
    haps_pos = {h["id"]: [round(h["lat"], 4), round(h["lon"], 4)] for h in record["haps"]}
    cells = {}
    for c in record["cells"]:
        demand = c["demand_critical_mbps"] + c["demand_general_mbps"]
        served = c["ground_served_mbps"] + c["haps_served_critical_mbps"] + c["haps_served_general_mbps"]
        cells[c["cell_id"]] = [
            sig(demand),
            sig(max(0.0, demand - c["ground_served_mbps"])),
            round(100.0 * served / demand, 1) if demand > 0 else 100,
            sig(c["unserved_critical_mbps"]),
            1 if c.get("isolated") else 0,
        ]
    links: dict[tuple, dict] = {}
    for link in record["links"]:
        kind = link["link_type"]
        if kind == "isl":
            key = (kind, *sorted((link["from_id"], link["to_id"])))
            a, b = haps_pos.get(key[1]), haps_pos.get(key[2])
        elif kind == "feeder":
            key = (kind, link["from_id"], link["to_id"])
            a, b = haps_pos.get(link["from_id"]), gateways.get(link["to_id"])
        else:
            continue
        if a is None or b is None:
            continue
        util = round(link["utilisation"], 3)
        if key not in links or util > links[key]["util"]:
            links[key] = {"type": kind, "a": a, "b": b, "util": util}
    t = record["totals"]
    return {
        "t": record["sim_time"],
        "totals": {
            "demand_mbps": sig(t["demand_mbps"], 4),
            "pct_served": round(t["pct_served"], 3),
            "pct_critical_served": round(t["pct_critical_served"], 3),
            "unserved_critical_mbps": sig(t["unserved_critical_mbps"], 4),
            "unserved_general_mbps": sig(t["unserved_general_mbps"], 4),
            "isolated_cells": t.get("isolated_cells", 0),
        },
        "cells": cells,
        "haps": [
            {
                "id": h["id"],
                "lat": round(h["lat"], 4),
                "lon": round(h["lon"], 4),
                "battery": round(h["battery_pct"], 1),
                "mode": h["power_mode"],
                "target_lat": round(h["target_lat"], 4),
                "target_lon": round(h["target_lon"], 4),
                "pinned": bool(h.get("pinned", False)),
            }
            for h in record["haps"]
        ],
        "links": list(links.values()),
    }


def export_static(sim: Simulation, out: Path) -> None:
    n = write_json(out / "cells.geojson", cells_geojson(sim))
    print(f"  cells.geojson: {len(sim.cells)} cells, {n / 1e6:.1f} MB")
    st = static_info(sim)
    write_json(out / "static.json", st)
    print(
        f"  static.json: {len(st['hubs'])} hubs, {len(st['backbone'])} backbone links, "
        f"{len(st['subsea'])} subsea links, {len(st['gateways'])} gateways"
    )


def export_run(
    log: Path, gateways: dict[str, list[float]], out: Path, cell_ids: set[str] | None = None
) -> dict | None:
    """Compact frames for one run, or None if it was recorded on another grid."""
    frames = []
    controller = None
    n_haps = 0
    with log.open() as f:
        for line in f:
            r = json.loads(line)
            if cell_ids is not None and not frames and {c["cell_id"] for c in r["cells"]} != cell_ids:
                print(f"  skipping {log.stem}: recorded on a different grid; re-run it", file=sys.stderr)
                return None
            controller = controller or r.get("controller")
            n_haps = max(n_haps, len(r["haps"]))
            gws = {**gateways, **{g["id"]: [g["lat"], g["lon"]] for g in r.get("gateways", [])}}
            frames.append(compact_frame(r, gws))
    if not frames:
        return None
    n = write_json(out / "runs" / f"{log.stem}.json", {"run_id": log.stem, "controller": controller, "frames": frames})
    print(f"  runs/{log.stem}.json: {len(frames)} frames, {n / 1e6:.1f} MB (from {log.stat().st_size / 1e6:.0f} MB)")
    return {"run_id": log.stem, "controller": controller, "haps": n_haps, "frames": len(frames)}


def update_index(out: Path, entries: list[dict]) -> int:
    """Add or replace runs in runs/index.json; drop entries whose file is gone."""
    path = out / "runs" / "index.json"
    index = json.loads(path.read_text()) if path.exists() else []
    new = {e["run_id"] for e in entries}
    index = [e for e in index if e["run_id"] not in new and (out / "runs" / f"{e['run_id']}.json").exists()]
    index += entries
    write_json(path, sorted(index, key=lambda e: e["run_id"]))
    return len(index)
