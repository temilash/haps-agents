"""Traffic destinations: every cell, weighted by population as a proxy for
where services are hosted."""

from __future__ import annotations

from sim.state import Cell


def endpoint_weights(cells: list[Cell], exponent: float) -> dict[str, float]:
    """population ** exponent, normalised to sum to 1."""
    raw = {c.cell_id: max(0.0, c.population) ** exponent for c in cells}
    total = sum(raw.values())
    if total <= 0.0:
        return {cid: 1.0 / len(raw) for cid in raw} if raw else {}
    return {cid: w / total for cid, w in raw.items()}


def endpoint_capacities(
    weights: dict[str, float], total_demand_mbps: float, headroom: float
) -> dict[str, float]:
    """How much traffic each cell can terminate this tick."""
    return {cid: w * total_demand_mbps * headroom for cid, w in weights.items()}
