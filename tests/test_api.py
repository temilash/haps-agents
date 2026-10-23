from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.server import create_app
from sim.grid import haversine_km

LONDON = {"lat": 51.507, "lon": -0.128}
YORK = {"lat": 53.96, "lon": -1.08}


@pytest.fixture()
def client(tmp_path: Path):
    with TestClient(create_app(project_root=tmp_path, region="england")) as c:
        yield c


def _ids(summary: dict, key: str) -> list[str]:
    return [x["id"] for x in summary[key]]


def test_reset(client):
    s = client.post(
        "/api/sim/reset",
        json={"region": "england", "controller": "greedy", "haps": 2, "start_time": "08:00", "day_type": "weekend"},
    ).json()
    assert s["region"] == "england" and s["controller"] == "greedy"
    assert s["sim_time"] == "08:00" and not s["running"]
    assert _ids(s, "haps") == ["H1", "H2"]
    s = client.post("/api/sim/step").json()
    assert s["sim_time"] == "08:10" and s["tick_ms"] > 0


def test_add_and_remove_haps(client):
    start = client.get("/api/sim").json()
    n = len(start["haps"])
    s = client.post("/api/haps", json=YORK).json()
    assert len(s["haps"]) == n + 1
    new = s["haps"][-1]
    assert new["id"] == f"H{n + 1}" and (new["lat"], new["lon"]) == (YORK["lat"], YORK["lon"])
    s = client.post("/api/haps").json()  # auto-placed
    assert len(s["haps"]) == n + 2
    s = client.delete(f"/api/haps/H{n + 1}").json()
    assert len(s["haps"]) == n + 1 and f"H{n + 1}" not in _ids(s, "haps")
    assert client.delete("/api/haps/H99").status_code == 404


def test_pinned_haps_is_left_alone_by_greedy(client):
    client.post("/api/sim/reset", json={"region": "england", "controller": "greedy", "start_time": "19:00"})
    # Overflow in the south, so greedy has a reason to move HAPS.
    client.post("/api/outage", json={**LONDON, "radius_km": 60, "tower_status": 0.0})
    s = client.post("/api/haps/H1/target", json=YORK).json()
    h1 = next(h for h in s["haps"] if h["id"] == "H1")
    assert h1["pinned"] and (h1["target_lat"], h1["target_lon"]) == (YORK["lat"], YORK["lon"])
    before = {h["id"]: (h["target_lat"], h["target_lon"]) for h in s["haps"]}
    for _ in range(4):
        s = client.post("/api/sim/step").json()
    after = {h["id"]: (h["target_lat"], h["target_lon"]) for h in s["haps"]}
    assert after["H1"] == (YORK["lat"], YORK["lon"])
    assert any(after[i] != before[i] for i in after if i != "H1")  # greedy did act
    h1 = next(h for h in s["haps"] if h["id"] == "H1")
    assert haversine_km(h1["lat"], h1["lon"], *YORK.values()) < haversine_km(53.4808, -2.2426, *YORK.values())

    s = client.post("/api/haps/H1/unpin").json()
    assert not next(h for h in s["haps"] if h["id"] == "H1")["pinned"]


def test_outage_changes_served_traffic_on_next_step(client):
    client.post("/api/sim/reset", json={"region": "england", "start_time": "19:50"})
    before = client.post("/api/sim/step").json()["pct_served"]
    s = client.post("/api/outage", json={**LONDON, "radius_km": 50, "tower_status": 0.0}).json()
    assert s["pct_served"] == before  # applies from the next tick
    assert s["outages"] == [{**LONDON, "radius_km": 50.0, "tower_status": 0.0}]
    after = client.post("/api/sim/step").json()["pct_served"]
    assert after < before - 1.0
    s = client.post("/api/outage/clear").json()
    assert s["outages"] == []
    assert client.post("/api/sim/step").json()["pct_served"] > after


def test_gateway_add_move_delete(client):
    s = client.post("/api/gateways", json=YORK).json()
    assert _ids(s, "gateways") == ["GW-1", "GW-2", "GW-3"]
    s = client.patch("/api/gateways/GW-3", json=LONDON).json()
    gw = next(g for g in s["gateways"] if g["id"] == "GW-3")
    assert (gw["lat"], gw["lon"]) == (LONDON["lat"], LONDON["lon"])
    s = client.delete("/api/gateways/GW-3").json()
    assert _ids(s, "gateways") == ["GW-1", "GW-2"]
    assert client.delete("/api/gateways/GW-3").status_code == 404
    assert client.patch("/api/gateways/GW-9", json=LONDON).status_code == 404


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("post", "/api/sim/speed", {"ticks_per_second": 3}),
        ("post", "/api/outage", {**LONDON, "radius_km": 10, "tower_status": 2}),
        ("post", "/api/outage", {"lat": 40.0, "lon": -30.0}),  # outside the region
        ("post", "/api/haps", {"lat": 53.0}),  # lat without lon
        ("post", "/api/haps", {"lat": 48.85, "lon": 2.35}),  # Paris
        ("post", "/api/haps/H1/target", {"lat": 95, "lon": 0}),
        ("post", "/api/gateways", {"lat": "north", "lon": 0}),
        ("post", "/api/sim/reset", {"start_time": "25:00"}),
        ("post", "/api/sim/reset", {"region": "atlantis"}),
        ("post", "/api/controller", {"name": "clever"}),
    ],
)
def test_invalid_input_returns_422(client, method, path, body):
    r = getattr(client, method)(path, json=body)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]


def test_speed_play_pause_and_controller(client):
    assert client.post("/api/sim/speed", json={"ticks_per_second": 5}).json()["ticks_per_second"] == 5
    assert client.post("/api/controller", json={"name": "greedy"}).json()["controller"] == "greedy"
    assert client.post("/api/sim/play").json()["running"]
    assert not client.post("/api/sim/pause").json()["running"]


def test_websocket_frames(client):
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "frame" and first["t"] == "06:00"
        assert len(next(iter(first["cells"].values()))) == 5
        client.post("/api/sim/step")
        frame = ws.receive_json()
        assert frame["type"] == "frame" and frame["t"] == "06:10" and frame["tick_ms"] > 0
        client.post("/api/haps", json=YORK)
        state = ws.receive_json()
        assert state["type"] == "state" and len(state["haps"]) == 5


def test_static_regions_and_cells(client):
    st = client.get("/api/static").json()
    assert len(st["hubs"]) == 14 and len(st["gateways"]) == 2
    regions = {r["name"]: r for r in client.get("/api/regions").json()}
    assert regions["england"]["current"] and regions["england"]["built"]
    cells = client.get("/api/regions/england/cells").json()
    assert len(cells["features"]) == 571
    assert client.get("/api/regions/atlantis/cells").status_code == 404
    assert client.get("/").status_code == 200  # the viewer


def test_live_session_is_recorded_for_replay(client, tmp_path):
    client.post("/api/sim/step")
    client.post("/api/sim/step")
    log = client.get("/api/sim").json()["log"]
    client.post("/api/sim/reset", json={"region": "england"})  # finishes the session
    run_id = log.removesuffix(".jsonl")
    index = json.loads((tmp_path / "web/data/england/runs/index.json").read_text())
    entry = next(e for e in index if e["run_id"] == run_id)
    assert entry["frames"] == 2 and entry["controller"].startswith("live")


def test_subsea_cut_isolates_cells(tmp_path):
    with TestClient(create_app(project_root=tmp_path, region="uk")) as c:
        assert c.get("/api/sim").json()["isolated_cells"] == 0
        s = c.post("/api/subsea_cut", json={"cut": True}).json()
        assert s["subsea_cut"] and s["isolated_cells"] > 50
        assert not next(g for g in s["gateways"] if g["id"] == "GW-Belfast")["active"]
        frame = None
        with c.websocket_connect("/ws") as ws:
            ws.receive_json()
            c.post("/api/sim/step")
            frame = ws.receive_json()
        assert sum(v[4] for v in frame["cells"].values()) == s["isolated_cells"]
        s = c.post("/api/subsea_cut", json={"cut": False}).json()
        assert s["isolated_cells"] == 0
