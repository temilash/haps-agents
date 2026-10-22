#!/usr/bin/env python3
"""Run a few ticks and print totals plus the busiest links."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.engine import Simulation  # noqa: E402


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
    print(f"  tick_s:                 {elapsed_s:.3f}")
    top = sorted(result.links, key=lambda link: link.utilisation, reverse=True)[:5]
    print("  most loaded links:")
    for link in top:
        print(
            f"    {link.link_type:6} {link.from_id} -> {link.to_id}  "
            f"flow={link.flow_mbps:,.1f}  cap={link.capacity_mbps:,.0f}  "
            f"util={100 * link.utilisation:.2f}%"
        )


def run_named(sim: Simulation, sim_time: str, title: str) -> None:
    t0 = time.perf_counter()
    result = sim.step(sim_time)
    print_tick(title, result, time.perf_counter() - t0)


def main() -> None:
    sim = Simulation()
    run_named(sim, "12:00", "12:00 normal")
    run_named(sim, "20:00", "20:00 normal")
    sim.set_tower_status(54.89, -2.93, 30.0, 0.1)
    run_named(sim, "20:00", "20:00 Carlisle towers at 0.1")


if __name__ == "__main__":
    main()
