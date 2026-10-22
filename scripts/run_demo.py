#!/usr/bin/env python3
"""Run a few ticks and print totals plus the busiest links."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.config import load_config  # noqa: E402
from sim.engine import Simulation  # noqa: E402

RURAL = (54.45, -2.10)
STEVENAGE = (51.90, -0.20)  # London-north corridor


def print_tick(title: str, result, elapsed_s: float) -> None:
    t = result.totals
    print(f"\n=== {title} ===")
    print(f"  demand_mbps:            {t.demand_mbps:,.1f}")
    print(f"  pct_served:             {t.pct_served:.4f}")
    print(f"  pct_critical_served:    {t.pct_critical_served:.4f}")
    print(f"  unserved_critical_mbps: {t.unserved_critical_mbps:,.3f}")
    print(f"  unserved_general_mbps:  {t.unserved_general_mbps:,.3f}")
    print(f"  links_over_80pct:       {t.links_over_80pct}")
    print(f"  avg_ground_latency_ms:  {t.avg_ground_latency_ms:.3f}")
    print(f"  avg_haps_latency_ms:    {t.avg_haps_latency_ms:.3f}")
    if any(link.link_type == "backhaul" for link in result.links):
        print(f"  unserved_mbps:          {t.unserved_critical_mbps + t.unserved_general_mbps:,.3f}")
        print(f"  pct_crossing_backhaul:  {t.pct_crossing_backhaul:.2f}")
        print(f"  backhaul_over_80pct:    {t.backhaul_links_over_80pct}")
    print(f"  tick_s:                 {elapsed_s:.3f}")
    top = sorted(result.links, key=lambda link: link.utilisation, reverse=True)[:5]
    print("  most loaded links:")
    for link in top:
        print(
            f"    {link.link_type:8} {link.from_id} -> {link.to_id}  "
            f"flow={link.flow_mbps:,.1f}  cap={link.capacity_mbps:,.0f}  "
            f"util={100 * link.utilisation:.2f}%"
        )


def run_named(sim: Simulation, sim_time: str, title: str) -> None:
    t0 = time.perf_counter()
    result = sim.step(sim_time)
    print_tick(title, result, time.perf_counter() - t0)


def run_simple(cfg) -> None:
    sim = Simulation(config=cfg)
    run_named(sim, "12:00", "12:00 normal")
    run_named(sim, "20:00", "20:00 normal")
    sim.set_tower_status(54.89, -2.93, 30.0, 0.1)
    run_named(sim, "20:00", "20:00 Carlisle towers at 0.1")


def run_mesh(cfg) -> None:
    t0 = time.perf_counter()
    sim = Simulation(config=cfg)
    print(f"startup_s (incl. backhaul calibration or cache load): {time.perf_counter() - t0:.3f}")
    run_named(sim, "20:00", "20:00 normal")
    for (lat, lon), name in ((RURAL, "rural"), (STEVENAGE, "Stevenage")):
        sim = Simulation(config=cfg)
        sim.set_tower_status(lat, lon, 15.0, 0.1)
        run_named(sim, "20:00", f"20:00 {name} outage ({lat}, {lon}), 15 km at 0.1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-model", choices=["simple", "mesh"], help="override config.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / "config.yaml")
    if args.ground_model:
        cfg.ground_model = args.ground_model
    if cfg.ground_model == "mesh":
        run_mesh(cfg)
    else:
        run_simple(cfg)


if __name__ == "__main__":
    main()
