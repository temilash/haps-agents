#!/usr/bin/env python3
"""Export a region's grid and saved runs as compact static files for web/index.html."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.engine import Simulation  # noqa: E402
from sim.frames import export_run, export_static, update_index  # noqa: E402
from sim.region import DEFAULT_REGION, load_region  # noqa: E402

WEB = ROOT / "web" / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="run ids (log file names without .jsonl); default: all saved runs")
    args = parser.parse_args()

    region = load_region(args.region)
    sim = Simulation(region=region)
    out = WEB / region.name
    print(f"Exporting {region.name} to {out}")
    export_static(sim, out)

    runs_dir = sim.processed_dir / "runs"
    run_ids = args.runs if args.runs else sorted(p.stem for p in runs_dir.glob("*.jsonl"))
    gateways = {g.id: [round(g.lat, 4), round(g.lon, 4)] for g in sim.gateways}
    cell_ids = {c.cell_id for c in sim.cells}
    entries = []
    for run_id in run_ids:
        log = runs_dir / f"{run_id}.jsonl"
        if not log.exists():
            sys.exit(f"no saved run {log}")
        entry = export_run(log, gateways, out, cell_ids)
        if entry is not None:
            entries.append(entry)
    print(f"  runs/index.json: {update_index(out, entries)} runs")


if __name__ == "__main__":
    main()
