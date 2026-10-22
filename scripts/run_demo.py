#!/usr/bin/env python3
"""Run a few ticks and print totals plus the busiest links."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.engine import Simulation  # noqa: E402
from sim.region import DEFAULT_REGION, Region, load_region, region_config  # noqa: E402


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


def run_simple(cfg, region: Region) -> None:
    sim = Simulation(config=cfg, region=region)
    run_named(sim, "12:00", "12:00 normal")
    run_named(sim, "20:00", "20:00 normal")
    o = region.spec.demo.simple_outage
    if o is not None:
        sim.set_tower_status(o.lat, o.lon, o.radius_km, o.tower_status)
        run_named(sim, "20:00", f"20:00 {o.name} towers at {o.tower_status}")


def run_mesh(cfg, region: Region) -> None:
    t0 = time.perf_counter()
    sim = Simulation(config=cfg, region=region)
    print(f"startup_s (incl. backhaul calibration or cache load): {time.perf_counter() - t0:.3f}")
    run_named(sim, "20:00", "20:00 normal")
    for o in region.spec.demo.outages:
        sim = Simulation(config=cfg, region=region)
        sim.set_tower_status(o.lat, o.lon, o.radius_km, o.tower_status)
        run_named(
            sim, "20:00",
            f"20:00 {o.name} outage ({o.lat}, {o.lon}), {o.radius_km:g} km at {o.tower_status}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--ground-model", choices=["simple", "mesh"], help="override config")
    args = parser.parse_args()
    region = load_region(args.region)
    cfg = region_config(region)
    if args.ground_model:
        cfg.ground_model = args.ground_model
    if cfg.ground_model == "mesh":
        run_mesh(cfg, region)
    else:
        run_simple(cfg, region)


if __name__ == "__main__":
    main()
