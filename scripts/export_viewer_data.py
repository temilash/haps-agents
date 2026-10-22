#!/usr/bin/env python3
"""Export a region's grid and saved runs as compact static files for web/index.html."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.engine import Simulation  # noqa: E402
from sim.region import DEFAULT_REGION, load_region  # noqa: E402
from sim.topology import auto_backbone, backbone_from_spec  # noqa: E402

WEB = ROOT / "web" / "data"


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


def export_static(sim: Simulation, out: Path) -> None:
    hub_of = {cid: hub for cid, (hub, _km) in sim.metro.items()}
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
                    "area_type": sim.area_type[c.cell_id],
                    "hub": hub_of.get(c.cell_id),
                },
            }
        )
    n = write_json(out / "cells.geojson", {"type": "FeatureCollection", "features": features})
    print(f"  cells.geojson: {len(features)} cells, {n / 1e6:.1f} MB")

    region = sim.region
    if region.backbone is not None:
        links = backbone_from_spec(region.backbone.get("links", []), sim.hubs)
    else:
        links = auto_backbone(sim.hubs, sim.config.mesh)
    pos = {h.name: [round(h.lat, 4), round(h.lon, 4)] for h in sim.hubs}
    static = {
        "region": region.name,
        "coverage_km": sim.config.network.haps_coverage_km,
        "gateways": [{"id": g.id, "lat": g.lat, "lon": g.lon} for g in sim.gateways],
        "hubs": [
            {"name": h.name, "lat": round(h.lat, 4), "lon": round(h.lon, 4), "radius_km": h.radius_km}
            for h in sim.hubs
        ],
        "backbone": [[pos[a], pos[b]] for a, b in links],
    }
    write_json(out / "static.json", static)
    print(f"  static.json: {len(sim.hubs)} hubs, {len(links)} backbone links, {len(sim.gateways)} gateways")


def export_run(log: Path, gateways: dict[str, list[float]], out: Path) -> dict:
    frames = []
    controller = None
    with log.open() as f:
        for line in f:
            r = json.loads(line)
            controller = controller or r.get("controller")
            haps_pos = {h["id"]: [round(h["lat"], 4), round(h["lon"], 4)] for h in r["haps"]}
            cells = {}
            for c in r["cells"]:
                demand = c["demand_critical_mbps"] + c["demand_general_mbps"]
                served = c["ground_served_mbps"] + c["haps_served_critical_mbps"] + c["haps_served_general_mbps"]
                cells[c["cell_id"]] = [
                    sig(demand),
                    sig(max(0.0, demand - c["ground_served_mbps"])),
                    round(100.0 * served / demand, 1) if demand > 0 else 100,
                    sig(c["unserved_critical_mbps"]),
                ]
            # ISL is reported per direction; draw each pair once at its busier direction.
            links: dict[tuple, dict] = {}
            for link in r["links"]:
                kind = link["link_type"]
                if kind == "isl":
                    key = (kind, *sorted((link["from_id"], link["to_id"])))
                    a, b = haps_pos[key[1]], haps_pos[key[2]]
                elif kind == "feeder":
                    key = (kind, link["from_id"], link["to_id"])
                    a, b = haps_pos[link["from_id"]], gateways[link["to_id"]]
                else:
                    continue
                util = round(link["utilisation"], 3)
                if key not in links or util > links[key]["util"]:
                    links[key] = {"type": kind, "a": a, "b": b, "util": util}
            t = r["totals"]
            frames.append(
                {
                    "t": r["sim_time"],
                    "totals": {
                        "demand_mbps": sig(t["demand_mbps"], 4),
                        "pct_served": round(t["pct_served"], 3),
                        "pct_critical_served": round(t["pct_critical_served"], 3),
                        "unserved_critical_mbps": sig(t["unserved_critical_mbps"], 4),
                        "unserved_general_mbps": sig(t["unserved_general_mbps"], 4),
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
                        }
                        for h in r["haps"]
                    ],
                    "links": list(links.values()),
                }
            )
    n = write_json(out / "runs" / f"{log.stem}.json", {"run_id": log.stem, "controller": controller, "frames": frames})
    print(f"  runs/{log.stem}.json: {len(frames)} frames, {n / 1e6:.1f} MB (from {log.stat().st_size / 1e6:.0f} MB)")
    return {"run_id": log.stem, "controller": controller, "frames": len(frames)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="run ids (log file names without .jsonl); default: all saved runs")
    args = parser.parse_args()

    region = load_region(args.region)
    sim = Simulation(region=region)
    if sim.traffic is None:
        sys.exit("export needs demand_model: area_types (for area types and hubs)")
    out = WEB / region.name
    print(f"Exporting {region.name} to {out}")
    export_static(sim, out)

    runs_dir = sim.processed_dir / "runs"
    run_ids = args.runs or sorted(p.stem for p in runs_dir.glob("*.jsonl"))
    gateways = {g.id: [round(g.lat, 4), round(g.lon, 4)] for g in sim.gateways}
    index_path = out / "runs" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [e for e in index if e["run_id"] not in run_ids]
    for run_id in run_ids:
        log = runs_dir / f"{run_id}.jsonl"
        if not log.exists():
            sys.exit(f"no saved run {log}")
        index.append(export_run(log, gateways, out))
    write_json(index_path, sorted(index, key=lambda e: e["run_id"]))
    print(f"  runs/index.json: {len(index)} runs")


if __name__ == "__main__":
    main()
