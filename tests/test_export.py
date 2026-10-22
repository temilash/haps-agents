from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sim.controllers import GreedyController
from sim.engine import Simulation
from sim.runner import run

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_viewer_data", ROOT / "scripts/export_viewer_data.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_export_static_and_run(tmp_path: Path):
    sim = Simulation(region="england", project_root=tmp_path)
    log = tmp_path / "runs" / "r1_greedy.jsonl"
    run(sim, GreedyController(), start="19:00", hours=1, log_path=log)

    out = tmp_path / "web"
    export.export_static(sim, out)
    cells = json.loads((out / "cells.geojson").read_text())
    assert len(cells["features"]) == len(sim.cells)
    props = cells["features"][0]["properties"]
    assert set(props) == {"cell_id", "population", "area_type", "hub"}
    static = json.loads((out / "static.json").read_text())
    assert len(static["hubs"]) == 14 and len(static["backbone"]) == 25

    gateways = {g.id: [g.lat, g.lon] for g in sim.gateways}
    entry = export.export_run(log, gateways, out)
    assert entry == {"run_id": "r1_greedy", "controller": "greedy", "frames": 6}
    frames = json.loads((out / "runs/r1_greedy.json").read_text())["frames"]
    f = frames[-1]
    assert f["t"] == "20:00"
    assert len(f["cells"]) == len(sim.cells) and len(next(iter(f["cells"].values()))) == 4
    assert {h["id"] for h in f["haps"]} == {"H1", "H2", "H3", "H4"}
    assert {link["type"] for link in f["links"]} <= {"isl", "feeder"}
    # ISL pairs are drawn once.
    isl = [tuple(sorted(map(tuple, (l["a"], l["b"])))) for l in f["links"] if l["type"] == "isl"]
    assert len(isl) == len(set(isl))


def test_sig_rounding():
    assert export.sig(123456.0) == 123000
    assert export.sig(0.0123456) == 0.0123
    assert export.sig(0.0) == 0
