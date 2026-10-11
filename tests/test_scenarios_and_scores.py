import numpy as np
import pytest

from eval.comfort_score import _metrics, end_point, replay, score_scenario
from gym_env.half_car.core.ode_model import HalfCarODE
from utils.dataset import region_of
from utils.scenario_loader import enabled_eval_suites, load_scenario_any, _find_root


def test_repo_root_does_not_need_fetched_data():
    assert (_find_root() / "pyproject.toml").is_file()


def test_synthetic_suites_ship_with_the_repo():
    suites = dict(enabled_eval_suites(["01_single_bump", "03_three_bumps"]))
    assert "c" in suites["01_single_bump"]
    assert len(suites["03_three_bumps"]) >= 30


def test_scenario_c_is_a_raised_crosswalk():
    road, track, speed, _ = load_scenario_any("c")
    assert track is None
    assert speed == pytest.approx(30.0 / 3.6)
    (x0, height, width), = road._bumps
    assert (height, width) == (pytest.approx(0.1), pytest.approx(5.0))


def test_flat_road_gives_no_acceleration():
    road, *_ = load_scenario_any("c")
    road._bumps = []
    speeds = [8.0] * 600
    z_b, phi, s_pos = replay(road, speeds, 8.0)
    m = _metrics(z_b, phi, s_pos, speeds, 30.0)
    assert m["a_w"] < 1e-3 and m["completed"]


def test_constant_speed_score_on_scenario_c_is_stable():
    rows = score_scenario("01_single_bump", "c", {}, [])
    const = next(r for r in rows if r["source"] == "const")
    assert const["a_w"] == pytest.approx(0.4624, abs=2e-3)
    assert const["vdv"] == pytest.approx(1.2811, abs=5e-3)
    assert const["travel_time_s"] == pytest.approx(5.76, abs=0.05)
    assert const["a_w_vs_const_pct"] == 0.0


def test_slower_crossing_is_more_comfortable_on_a_tall_bump():
    scores = {}
    for kmh in (30.0, 12.0):
        road, *_ = load_scenario_any("c")
        v = kmh / 3.6
        speeds = [v] * int(np.ceil(end_point(road, HalfCarODE().wheelbase) / (v * 0.02)) + 5)
        z_b, phi, s_pos = replay(road, speeds, v)
        scores[kmh] = _metrics(z_b, phi, s_pos, speeds, end_point(road, HalfCarODE().wheelbase))["a_w"]
    assert scores[12.0] < scores[30.0]


def test_region_of_locations():
    assert region_of("nardo_sb01") == "italy"
    assert region_of("munich_sb07") == "germany"
