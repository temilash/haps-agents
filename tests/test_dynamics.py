from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.config import AppConfig, DynamicsConfig
from sim.controllers import GreedyController, StaticController, make_controller
from sim.dynamics import daylight_factor, move_towards
from sim.engine import Simulation
from sim.grid import haversine_km
from sim.latency import haps_routes
from sim.region import load_region, region_config
from sim.runner import run
from sim.state import Gateway, Haps
from sim.traffic import curve_at


def _england(**changes):
    cfg = region_config(load_region("england"))
    for path, value in changes.items():
        obj = cfg
        *parents, leaf = path.split(".")
        for p in parents:
            obj = getattr(obj, p)
        setattr(obj, leaf, value)
    return cfg


def _sim(tmp: Path, cfg=None) -> Simulation:
    return Simulation(config=cfg or _england(), region="england", project_root=tmp)


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("dyn")


@pytest.fixture(scope="module")
def static_day(root):
    sim = _sim(root)
    return sim, run(sim, StaticController())


@pytest.fixture(scope="module")
def greedy_day(root):
    sim = _sim(root)
    ctrl = make_controller("greedy", sim.config.greedy)
    picks = []  # (minutes, haps_id, target, planned positions of all HAPS)
    original = ctrl.decide

    def recording(obs):
        actions = original(obs)
        planned = {h.id: (h.target_lat, h.target_lon) for h in obs.haps}
        for a in actions:
            planned[a.haps_id] = (a.target_lat, a.target_lon)
        picks.extend((obs.sim_minutes, a.haps_id, planned) for a in actions)
        return actions

    ctrl.decide = recording
    return sim, run(sim, ctrl), picks


# --- traffic ---------------------------------------------------------------------


def test_area_types_peak_at_different_hours(root):
    sim = _sim(root)
    hours = np.arange(0.0, 24.0, 0.25)
    peaks = {}
    for area in ("city_centre", "nightlife", "suburban", "rural"):
        ids = [c.cell_id for c in sim.cells if sim.area_type[c.cell_id] == area]
        assert ids, area
        totals = [sum(sim.traffic.active_fraction(cid, h) for cid in ids) for h in hours]
        peaks[area] = float(hours[int(np.argmax(totals))])
    assert 9.0 <= peaks["city_centre"] <= 17.0
    assert 19.0 <= peaks["suburban"] <= 23.0
    assert peaks["nightlife"] >= 20.0 or peaks["nightlife"] <= 1.0
    assert len(set(peaks.values())) == 4


def test_weekday_traffic_matches_single_curve_and_weekend_is_later(root):
    from sim.demand import active_fraction

    sim = _sim(root)
    weekend = _sim(root, _england(**{"traffic.day_type": "weekend"}))
    hours = np.arange(0.0, 24.0, 1.0 / 12.0)
    pop = {c.cell_id: c.population for c in sim.cells}
    day = sum(pop[c] * sim.traffic.active_fraction(c, h) for c in pop for h in hours)
    single = sum(pop.values()) * sum(active_fraction(h, sim.config.demand) for h in hours)
    assert day == pytest.approx(single, rel=0.01)
    cc = [c for c in pop if sim.area_type[c] == "city_centre"]
    wk = [sum(sim.traffic.active_fraction(c, h) for c in cc) for h in hours]
    we = [sum(weekend.traffic.active_fraction(c, h) for c in cc) for h in hours]
    assert max(we) < 0.8 * max(wk)  # business peak cut by 40%
    sub = [c for c in pop if sim.area_type[c] == "suburban"]
    peak_wk = hours[int(np.argmax([sum(sim.traffic.active_fraction(c, h) for c in sub) for h in hours]))]
    peak_we = hours[int(np.argmax([sum(weekend.traffic.active_fraction(c, h) for c in sub) for h in hours]))]
    assert peak_we - peak_wk == pytest.approx(1.0, abs=0.25)


def test_per_cell_variation_is_seeded_and_bounded(root):
    a, b = _sim(root), _sim(root)
    assert a.traffic.shift_h == b.traffic.shift_h
    assert max(abs(v) for v in a.traffic.shift_h.values()) <= 0.5
    assert all(0.9 <= v <= 1.1 for v in a.traffic.amplitude.values())
    other = _sim(root, _england(**{"traffic.seed": 1}))
    assert other.traffic.shift_h != a.traffic.shift_h


def test_curve_interpolation_passes_through_hourly_values():
    values = [float(i % 5) for i in range(24)]
    for h in range(24):
        assert curve_at(values, float(h)) == pytest.approx(values[h])
    assert curve_at(values, 24.0) == pytest.approx(values[0])


def test_routine_overflow_mostly_rural(root):
    sim = _sim(root)
    result = sim.step("20:00")
    overflow = {"rural": 0.0, "suburban": 0.0, "city_centre": 0.0, "nightlife": 0.0}
    demand = dict.fromkeys(overflow, 0.0)
    for c in result.cells:
        d = c.demand_critical_mbps + c.demand_general_mbps
        overflow[sim.area_type[c.cell_id]] += d - c.ground_served_mbps
        demand[sim.area_type[c.cell_id]] += d
    share = {k: overflow[k] / demand[k] for k in overflow}
    assert share["rural"] > share["suburban"] > 0
    assert share["city_centre"] == pytest.approx(0.0, abs=1e-9)


def test_headroom_2_means_no_overflow_all_day(root):
    cfg = _england()
    cfg.traffic.ground_headroom = dict.fromkeys(cfg.traffic.ground_headroom, 2.0)
    sim = _sim(root, cfg)
    for _ in range(int(24 * 60 / cfg.dynamics.tick_minutes)):
        obs = sim.observe()
        assert max(c.overflow_mbps for c in obs.cells) == pytest.approx(0.0, abs=1e-9)
        sim.step()


# --- movement and battery ----------------------------------------------------------


def test_move_towards_follows_great_circle():
    lat, lon, km = move_towards(51.5, -0.13, 53.48, -2.24, 50.0)
    assert km == 50.0
    assert haversine_km(51.5, -0.13, lat, lon) == pytest.approx(50.0, abs=1e-6)
    total = haversine_km(51.5, -0.13, 53.48, -2.24)
    assert haversine_km(lat, lon, 53.48, -2.24) == pytest.approx(total - 50.0, abs=1e-6)
    assert move_towards(51.5, -0.13, 51.6, -0.13, 50.0)[:2] == (51.6, -0.13)


def test_haps_never_moves_more_than_max_per_tick(greedy_day):
    sim, results, _picks = greedy_day
    dyn = sim.config.dynamics
    max_km = dyn.speed_kmh * dyn.tick_minutes / 60.0
    moves = 0
    for before, after in zip(results, results[1:]):
        for a, b in zip(before.haps, after.haps):
            km = haversine_km(a.lat, a.lon, b.lat, b.lon)
            assert km <= max_km + 1e-6
            moves += km > 0
    assert moves > 0  # greedy did move


def test_daylight_factor():
    cfg = DynamicsConfig()
    assert daylight_factor(3.0, cfg) == 0.0
    assert daylight_factor(22.0, cfg) == 0.0
    assert daylight_factor(12.5, cfg) == pytest.approx(1.0)
    assert 0.0 < daylight_factor(9.0, cfg) < daylight_factor(11.0, cfg) < 1.0


def test_battery_recharges_in_daylight_and_drains_at_night(static_day):
    _sim_, results = static_day
    by_time = {r.sim_time: r for r in results}

    def delta(t0: str, t1: str) -> float:
        return by_time[t1].haps[0].battery_pct - by_time[t0].haps[0].battery_pct

    assert delta("12:00", "13:00") > 0
    assert delta("02:00", "03:00") < 0


def test_low_battery_switches_to_low_power_and_halves_access(root):
    cfg = _england(**{"dynamics.battery_start_pct": 25.05})
    sim = _sim(root, cfg)
    sim.set_tower_status(53.48, -2.24, 50.0, 0.0)  # load H1
    sim.set_time("22:00")
    before = sim.step()
    h1_cap = {l.from_id: l.capacity_mbps for l in before.links if l.link_type == "access"}["H1"]
    sim.step()
    h1 = next(h for h in sim.haps if h.id == "H1")
    assert h1.battery_pct < h1.battery_floor_pct and h1.power_mode == "low"
    after = sim.step()
    cap = {l.from_id: l.capacity_mbps for l in after.links if l.link_type == "access"}["H1"]
    assert cap == pytest.approx(h1_cap * cfg.dynamics.low_power_access_factor)


# --- controllers ---------------------------------------------------------------------


def test_greedy_targets_always_have_a_gateway_route(greedy_day):
    sim, _results, picks = greedy_day
    assert picks
    net = sim.config.network
    for _minutes, hid, planned in picks:
        haps = [Haps(i, lat, lon) for i, (lat, lon) in planned.items()]
        gws = [Gateway(g.id, g.lat, g.lon) for g in sim.gateways]
        routes = haps_routes(haps, gws, net.isl_range_km, net.feeder_range_km, sim.config.latency)
        assert routes[hid] is not None, (hid, planned[hid])


def test_greedy_ignores_overflow_out_of_gateway_reach(root):
    # One HAPS and one gateway in the north; the only outage is in Cornwall.
    cfg = AppConfig.model_validate(
        {
            **_england().model_dump(),
            "haps": [{"id": "H1", "lat": 53.48, "lon": -2.24}],
            "gateways": [{"id": "GW-1", "lat": 53.80, "lon": -1.55}],
            "fleet": {"count": None},  # just the one listed HAPS
        }
    )
    cfg.traffic.ground_headroom = dict.fromkeys(cfg.traffic.ground_headroom, 2.0)
    cfg.greedy.horizon_minutes = 24 * 60  # anywhere is reachable in time
    sim = _sim(root, cfg)
    sim.set_tower_status(50.3, -4.5, 60.0, 0.0)
    sim.set_time("20:00")
    ctrl = GreedyController(horizon_minutes=24 * 60)
    for action in ctrl.decide(sim.observe()):
        assert haversine_km(action.target_lat, action.target_lon, 53.80, -1.55) <= 150.0
        assert haversine_km(action.target_lat, action.target_lon, 50.3, -4.5) > 200.0


def test_greedy_serves_at_least_as_much_as_static(static_day, greedy_day):
    def unserved(results):
        return sum(r.totals.unserved_critical_mbps + r.totals.unserved_general_mbps for r in results)

    assert unserved(greedy_day[1]) <= unserved(static_day[1])


def test_static_never_moves(static_day):
    _sim_, results = static_day
    assert all(
        (h.lat, h.lon) == (results[0].haps[i].lat, results[0].haps[i].lon)
        for r in results
        for i, h in enumerate(r.haps)
    )


def test_run_log_written(root, tmp_path):
    import json

    sim = _sim(root)
    log = tmp_path / "run.jsonl"
    run(sim, StaticController(), hours=1, log_path=log)
    lines = log.read_text().splitlines()
    assert len(lines) == 6
    first = json.loads(lines[0])
    assert first["sim_time"] == "06:10" and first["controller"] == "static"
    assert {h["id"] for h in first["haps"]} == {"H1", "H2", "H3", "H4"}
