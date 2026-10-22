"""Run a controller over simulated time and log every tick."""

from __future__ import annotations

import json
from pathlib import Path

from sim.controllers import Controller
from sim.engine import Simulation
from sim.state import TickResult


def _rounded(value, places: int = 3):
    if isinstance(value, float):
        return round(value, places)
    if isinstance(value, dict):
        return {k: _rounded(v, places) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v, places) for v in value]
    return value


def run(
    sim: Simulation,
    controller: Controller,
    start: str = "06:00",
    hours: float = 24.0,
    log_path: Path | None = None,
) -> list[TickResult]:
    """Each tick: the controller sees the latest state and acts, then time advances."""
    sim.set_time(start)
    ticks = int(round(hours * 60.0 / sim.config.dynamics.tick_minutes))
    results: list[TickResult] = []
    log = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w")
    try:
        for _ in range(ticks):
            sim.apply(controller.decide(sim.observe()))
            result = sim.step()
            results.append(result)
            if log is not None:
                record = {"controller": controller.name, **result.to_dict()}
                log.write(json.dumps(_rounded(record), separators=(",", ":")) + "\n")
    finally:
        if log is not None:
            log.close()
    return results


def volume_tb(mbps: float, minutes: float) -> float:
    """Data carried at `mbps` for `minutes`, in terabytes."""
    return mbps * minutes * 60.0 / 8.0 / 1e6
