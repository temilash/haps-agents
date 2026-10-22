"""Plain dataclasses for simulation state and tick output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Cell:
    cell_id: str
    lat: float
    lon: float
    population: float
    ground_capacity_mbps: float = 0.0
    tower_status: float = 1.0


@dataclass
class Haps:
    id: str
    lat: float
    lon: float


@dataclass
class Gateway:
    id: str
    lat: float
    lon: float


@dataclass
class LinkLoad:
    link_type: str
    from_id: str
    to_id: str
    flow_mbps: float
    capacity_mbps: float
    utilisation: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CellResult:
    cell_id: str
    demand_critical_mbps: float
    demand_general_mbps: float
    ground_served_mbps: float
    haps_served_critical_mbps: float
    haps_served_general_mbps: float
    unserved_critical_mbps: float
    unserved_general_mbps: float
    latency_ms: float
    hops: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Totals:
    demand_mbps: float
    pct_served: float
    pct_critical_served: float
    unserved_critical_mbps: float
    unserved_general_mbps: float
    links_over_80pct: int
    avg_ground_latency_ms: float
    avg_haps_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TickResult:
    sim_time: str
    cells: list[CellResult] = field(default_factory=list)
    links: list[LinkLoad] = field(default_factory=list)
    totals: Totals = field(
        default_factory=lambda: Totals(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sim_time": self.sim_time,
            "cells": [c.to_dict() for c in self.cells],
            "links": [link.to_dict() for link in self.links],
            "totals": self.totals.to_dict(),
        }
