"""Live simulation server: REST controls, WebSocket frames, and the viewer.

    uvicorn api.server:app --reload      # then open http://localhost:8000

One Simulation per server. The sim loop is an asyncio task; each tick runs in
a worker thread behind a lock, so requests are answered between ticks. Every
change made through the API applies from the next tick.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.controllers import Controller, make_controller  # noqa: E402
from sim.engine import Simulation  # noqa: E402
from sim.frames import (  # noqa: E402
    cells_geojson,
    compact_frame,
    export_run,
    export_static,
    gateways_info,
    static_info,
    update_index,
)
from sim.region import DEFAULT_REGION, REGIONS_DIR, load_region, region_config  # noqa: E402
from sim.runner import _rounded  # noqa: E402

SPEEDS = (0.5, 1.0, 2.0, 5.0)


# --- request bodies ------------------------------------------------------------


class Position(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class OptionalPosition(BaseModel):
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def both_or_neither(self) -> OptionalPosition:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("give both lat and lon, or neither to place automatically")
        return self


class ResetRequest(BaseModel):
    region: str = DEFAULT_REGION
    controller: Literal["static", "greedy"] | None = None
    haps: int | None = Field(default=None, ge=0, le=100)
    start_time: str = "06:00"
    day_type: Literal["weekday", "weekend"] | None = None

    @field_validator("start_time")
    @classmethod
    def hh_mm(cls, v: str) -> str:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
            raise ValueError("start_time must be HH:MM, 00:00-23:59")
        return v

    @field_validator("region")
    @classmethod
    def known_region(cls, v: str) -> str:
        if not (REGIONS_DIR / v / "region.yaml").exists():
            raise ValueError(f"unknown region {v!r}")
        return v


class SpeedRequest(BaseModel):
    ticks_per_second: float

    @field_validator("ticks_per_second")
    @classmethod
    def allowed(cls, v: float) -> float:
        if v not in SPEEDS:
            raise ValueError(f"ticks_per_second must be one of {list(SPEEDS)}")
        return v


class ControllerRequest(BaseModel):
    name: Literal["static", "greedy"]


class OutageRequest(Position):
    radius_km: float = Field(default=15.0, gt=0, le=300)
    tower_status: float = Field(default=0.1, ge=0, le=1)


class SubseaCutRequest(BaseModel):
    cut: bool


# --- session ------------------------------------------------------------------


class Session:
    """The one live simulation, its controller, clients and log."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.sim: Simulation | None = None
        self.controller: Controller | None = None
        self.running = False
        self.ticks_per_second = 1.0
        self.tick_ms: float | None = None
        self.latest: dict | None = None
        self.outages: list[dict] = []
        self.subsea_cut = False
        self.clients: set[WebSocket] = set()
        self.log = None
        self.log_path: Path | None = None
        self.wake = asyncio.Event()
        self._region_sims: dict[str, Simulation] = {}

    # Called in a worker thread, under the lock.

    def reset(self, req: ResetRequest) -> None:
        self._finish_log()
        region = load_region(req.region)
        cfg = region_config(region)
        if req.controller is not None:
            cfg.controller = req.controller
        if req.haps is not None:
            cfg.fleet.count = req.haps
        if req.day_type is not None:
            cfg.traffic.day_type = req.day_type
        self.sim = Simulation(config=cfg, region=region, project_root=self.root)
        self.sim.set_time(req.start_time)
        self.controller = make_controller(cfg.controller, cfg.greedy)
        self.running = False
        self.tick_ms = None
        self.outages = []
        self.subsea_cut = False
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = self.sim.processed_dir / "runs" / f"live_{stamp}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        self.sim.observe()  # evaluate the start time, for the first frame
        self.latest = self._frame(self.sim.last_result.to_dict())

    def tick(self) -> dict:
        sim = self.sim
        t0 = time.perf_counter()
        sim.apply(self.controller.decide(sim.observe()))
        result = sim.step()
        self.tick_ms = 1000.0 * (time.perf_counter() - t0)
        record = {
            "controller": self.controller.name,
            **result.to_dict(),
            "gateways": [{"id": g.id, "lat": g.lat, "lon": g.lon} for g in sim.gateways],
        }
        self.log.write(json.dumps(_rounded(record), separators=(",", ":")) + "\n")
        self.log.flush()
        self.latest = self._frame(record)
        return self.latest

    def _frame(self, record: dict) -> dict:
        gateways = {g.id: [round(g.lat, 4), round(g.lon, 4)] for g in self.sim.gateways}
        return {
            "type": "frame",
            **compact_frame(record, gateways),
            **self.summary(),
        }

    def summary(self) -> dict:
        sim = self.sim
        result = sim.last_result
        return {
            "region": sim.region.name,
            "controller": self.controller.name,
            "running": self.running,
            "ticks_per_second": self.ticks_per_second,
            "sim_time": sim._label(),
            "tick_ms": None if self.tick_ms is None else round(self.tick_ms, 1),
            "haps": [
                {
                    "id": h.id, "lat": round(h.lat, 4), "lon": round(h.lon, 4),
                    "target_lat": round(h.target_lat, 4), "target_lon": round(h.target_lon, 4),
                    "battery": round(h.battery_pct, 1), "mode": h.power_mode, "pinned": h.pinned,
                }
                for h in sim.haps
            ],
            "gateways": gateways_info(sim),
            "outages": self.outages,
            "subsea_cut": self.subsea_cut,
            "isolated_cells": len(sim.isolated_cells()),
            "pct_served": None if result is None else round(result.totals.pct_served, 3),
            "log": self.log_path.name if self.log_path else None,
        }

    def region_sim(self, name: str) -> Simulation:
        if self.sim is not None and self.sim.region.name == name:
            return self.sim
        if name not in self._region_sims:
            self._region_sims[name] = Simulation(region=name, project_root=self.root)
        return self._region_sims[name]

    def _finish_log(self) -> None:
        """Close the live log and export it for the replay viewer."""
        if self.log is None:
            return
        self.log.close()
        self.log = None
        path, sim = self.log_path, self.sim
        if path.stat().st_size == 0:
            path.unlink()
            return
        out = self.root / "web" / "data" / sim.region.name
        if not (out / "cells.geojson").exists():
            export_static(Simulation(region=sim.region, project_root=self.root), out)
        gateways = {g.id: [g.lat, g.lon] for g in sim.gateways}
        entry = export_run(path, gateways, out, {c.cell_id for c in sim.cells})
        if entry is not None:
            entry["controller"] = f"live ({entry['controller']})"
            update_index(out, [entry])

    # Async side.

    async def run(self, fn: Callable[[], Any]) -> Any:
        """Run fn in a worker thread, under the lock."""

        def locked():
            with self.lock:
                return fn()

        return await asyncio.get_running_loop().run_in_executor(None, locked)

    async def broadcast(self, message: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)

    async def loop(self) -> None:
        while True:
            if not self.running:
                await self.wake.wait()
                self.wake.clear()
                continue
            start = time.perf_counter()
            frame = await self.run(self.tick)
            # A pause may have landed while the tick ran: report the state now.
            await self.broadcast({**frame, "running": self.running})
            # A slow tick is followed straight away by the next; ticks never queue.
            remaining = 1.0 / self.ticks_per_second - (time.perf_counter() - start)
            if remaining > 0:
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=remaining)
                    self.wake.clear()
                except asyncio.TimeoutError:
                    pass


# --- app ------------------------------------------------------------------------


def create_app(project_root: Path | None = None, region: str | None = None) -> FastAPI:
    root = Path(project_root) if project_root else ROOT
    start_region = region or os.environ.get("HAPS_REGION", DEFAULT_REGION)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session = Session(root)
        app.state.session = session
        await session.run(lambda: session.reset(ResetRequest(region=start_region)))
        task = asyncio.create_task(session.loop())
        try:
            yield
        finally:
            task.cancel()
            await session.run(session._finish_log)

    app = FastAPI(title="HAPS live simulation", lifespan=lifespan)

    def session_of(app_: FastAPI) -> Session:
        return app_.state.session

    async def change(fn: Callable[[Session], Any]) -> dict:
        """Apply a change under the lock, tell clients, return the summary."""
        s = session_of(app)

        def apply():
            try:
                fn(s)
            except KeyError as e:
                raise HTTPException(404, str(e.args[0]) if e.args else "not found")
            except ValueError as e:
                raise HTTPException(422, str(e))
            summary = s.summary()
            if s.latest is not None:
                s.latest.update(summary)  # new clients get the current state
            return summary

        summary = await s.run(apply)
        await s.broadcast({"type": "state", **summary})
        return summary

    def inside(s: Session, lat: float, lon: float) -> None:
        if not s.sim.in_region(lat, lon):
            raise ValueError(f"position ({lat:.3f}, {lon:.3f}) is outside region {s.sim.region.name}")

    # Regions and static data

    @app.get("/api/regions")
    async def regions():
        s = session_of(app)
        out = []
        for d in sorted(REGIONS_DIR.iterdir()):
            if (d / "region.yaml").exists():
                out.append(
                    {
                        "name": d.name,
                        "built": (root / "data" / "processed" / d.name / "cells.parquet").exists(),
                        "current": s.sim is not None and s.sim.region.name == d.name,
                    }
                )
        return out

    @app.get("/api/regions/{region}/cells")
    async def region_cells(region: str):
        s = session_of(app)
        if not (REGIONS_DIR / region / "region.yaml").exists():
            raise HTTPException(404, f"unknown region {region!r}")
        return await s.run(lambda: cells_geojson(s.region_sim(region)))

    @app.get("/api/static")
    async def static():
        s = session_of(app)
        return await s.run(lambda: static_info(s.sim))

    @app.get("/api/sim")
    async def state():
        s = session_of(app)
        return await s.run(s.summary)

    # Running the simulation

    @app.post("/api/sim/reset")
    async def reset(req: ResetRequest):
        s = session_of(app)
        await s.run(lambda: s.reset(req))
        await s.broadcast(s.latest)
        return await s.run(s.summary)

    @app.post("/api/sim/play")
    async def play():
        summary = await change(lambda s: setattr(s, "running", True))
        session_of(app).wake.set()
        return summary

    @app.post("/api/sim/pause")
    async def pause():
        return await change(lambda s: setattr(s, "running", False))

    @app.post("/api/sim/step")
    async def step():
        s = session_of(app)
        frame = await s.run(s.tick)
        await s.broadcast(frame)
        return await s.run(s.summary)

    @app.post("/api/sim/speed")
    async def speed(req: SpeedRequest):
        summary = await change(lambda s: setattr(s, "ticks_per_second", req.ticks_per_second))
        session_of(app).wake.set()
        return summary

    @app.post("/api/controller")
    async def controller(req: ControllerRequest):
        def set_controller(s: Session):
            s.sim.config.controller = req.name
            s.controller = make_controller(req.name, s.sim.config.greedy)

        return await change(set_controller)

    # HAPS

    @app.post("/api/haps")
    async def add_haps(req: OptionalPosition | None = None):
        req = req or OptionalPosition()

        def add(s: Session):
            if req.lat is not None:
                inside(s, req.lat, req.lon)
            s.sim.add_haps(req.lat, req.lon)

        return await change(add)

    @app.delete("/api/haps/{haps_id}")
    async def remove_haps(haps_id: str):
        return await change(lambda s: s.sim.remove_haps(haps_id))

    @app.post("/api/haps/{haps_id}/target")
    async def haps_target(haps_id: str, req: Position):
        def pin(s: Session):
            s.sim._haps(haps_id)  # 404 before 422
            inside(s, req.lat, req.lon)
            s.sim.pin_haps(haps_id, req.lat, req.lon)

        return await change(pin)

    @app.post("/api/haps/{haps_id}/unpin")
    async def haps_unpin(haps_id: str):
        return await change(lambda s: s.sim.unpin_haps(haps_id))

    # Gateways

    @app.post("/api/gateways")
    async def add_gateway(req: Position):
        def add(s: Session):
            inside(s, req.lat, req.lon)
            s.sim.add_gateway(req.lat, req.lon)

        return await change(add)

    @app.patch("/api/gateways/{gateway_id}")
    async def move_gateway(gateway_id: str, req: Position):
        def move(s: Session):
            s.sim._gateway(gateway_id)
            inside(s, req.lat, req.lon)
            s.sim.move_gateway(gateway_id, req.lat, req.lon)

        return await change(move)

    @app.delete("/api/gateways/{gateway_id}")
    async def remove_gateway(gateway_id: str):
        return await change(lambda s: s.sim.remove_gateway(gateway_id))

    # Outages and cables

    @app.post("/api/outage")
    async def outage(req: OutageRequest):
        def add(s: Session):
            inside(s, req.lat, req.lon)
            s.sim.set_tower_status(req.lat, req.lon, req.radius_km, req.tower_status)
            s.outages.append(req.model_dump())

        return await change(add)

    @app.post("/api/outage/clear")
    async def outage_clear():
        def clear(s: Session):
            s.sim.clear_outages()
            s.outages = []

        return await change(clear)

    @app.post("/api/subsea_cut")
    async def subsea_cut(req: SubseaCutRequest):
        def cut(s: Session):
            s.sim.set_subsea_cut("all" if req.cut else None)
            s.subsea_cut = req.cut

        return await change(cut)

    # Frames

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        s = session_of(app)
        await websocket.accept()
        s.clients.add(websocket)
        try:
            if s.latest is not None:
                await websocket.send_json(await s.run(lambda: {**s.latest, **s.summary()}))
            while True:
                await websocket.receive_text()  # clients don't send anything we use
        except WebSocketDisconnect:
            pass
        finally:
            s.clients.discard(websocket)

    app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
    return app


app = create_app()
