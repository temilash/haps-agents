#!/usr/bin/env python3
"""Simulate a day with a HAPS controller, alongside the static baseline."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.controllers import make_controller  # noqa: E402
from sim.engine import Simulation  # noqa: E402
from sim.grid import haversine_km  # noqa: E402
from sim.region import DEFAULT_REGION, load_region, region_config  # noqa: E402
from sim.runner import run, volume_tb  # noqa: E402


def nearest_hub(sim: Simulation, lat: float, lon: float) -> str:
    hubs = getattr(sim, "hubs", None)
    if not hubs:
        return f"{lat:.2f},{lon:.2f}"
    h = min(hubs, key=lambda h: haversine_km(lat, lon, h.lat, h.lon))
    km = haversine_km(lat, lon, h.lat, h.lon)
    return h.name if km < 15 else f"{h.name}+{km:.0f}km"


def simulate(cfg, region, controller_name: str, start: str, hours: float, run_id: str):
    sim = Simulation(config=cfg, region=region)
    controller = make_controller(controller_name, cfg.greedy)
    log = sim.processed_dir / "runs" / f"{run_id}_{controller_name}_h{len(sim.haps)}.jsonl"
    t0 = time.perf_counter()
    results = run(sim, controller, start=start, hours=hours, log_path=log)
    return sim, results, time.perf_counter() - t0, log


def totals(results, tick_minutes: float) -> dict:
    t = [r.totals for r in results]
    demand = sum(x.demand_mbps for x in t)
    unserved = sum(x.unserved_critical_mbps + x.unserved_general_mbps for x in t)
    return {
        "served_tb": volume_tb(demand - unserved, tick_minutes),
        "unserved_tb": volume_tb(unserved, tick_minutes),
        "avg_pct_served": sum(x.pct_served for x in t) / len(t),
        "volume_pct_served": 100.0 * (demand - unserved) / demand if demand else 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--controller", choices=["static", "greedy"], default="greedy")
    parser.add_argument("--start", default="06:00")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--haps", type=int, help="fleet size (overrides fleet.count)")
    args = parser.parse_args()

    region = load_region(args.region)
    cfg = region_config(region)
    if args.haps is not None:
        cfg.fleet.count = args.haps
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    names = [args.controller] + (["static"] if args.controller != "static" else [])
    runs = {name: simulate(cfg, region, name, args.start, args.hours, run_id) for name in names}
    tick = cfg.dynamics.tick_minutes
    per_hour = int(round(60 / tick))

    head = f"{'time':>5} {'demand':>9} " + " ".join(
        f"| {n + ': served% crit% unserved':<34}" for n in names
    )
    n_haps = len(runs[names[0]][0].haps)
    print(f"\nRegion {region.name}, {n_haps} HAPS, {cfg.traffic.day_type}, {cfg.demand_model} demand, "
          f"tick {tick:g} min. Demand and unserved in Gbps.\n")
    print(head)
    first = runs[names[0]][1]
    for i in range(per_hour - 1, len(first), per_hour):
        r0 = first[i]
        row = f"{r0.sim_time:>5} {r0.totals.demand_mbps / 1e3:9,.0f} "
        for n in names:
            t = runs[n][1][i].totals
            unserved = (t.unserved_critical_mbps + t.unserved_general_mbps) / 1e3
            row += f"| {t.pct_served:7.3f} {t.pct_critical_served:7.3f} {unserved:9.2f}         "
        print(row)
        for n in names:
            sim, results = runs[n][0], runs[n][1]
            haps = "  ".join(
                f"{h.id} {nearest_hub(sim, h.lat, h.lon)} {h.battery_pct:.0f}% {h.power_mode}"
                for h in results[i].haps
            )
            print(f"{'':16}{n:>7}: {haps}")

    print("\nDaily totals")
    print(f"  {'':8} {'served TB':>10} {'unserved TB':>12} {'avg % served':>13} {'volume % served':>16} {'run time s':>11}")
    for n in names:
        sim, results, secs, log = runs[n]
        d = totals(results, tick)
        print(f"  {n:8} {d['served_tb']:10,.1f} {d['unserved_tb']:12,.3f} "
              f"{d['avg_pct_served']:13.4f} {d['volume_pct_served']:16.4f} {secs:11.1f}")
    for n in names:
        print(f"  log: {runs[n][3]}")


if __name__ == "__main__":
    main()
