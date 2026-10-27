import json

import numpy as np
import pytest

from eval.comfort_score import human_replay, human_speeds
from gym_env.common.env_params import DT
from utils.scenario_loader import load_scenario_json


def _window(tmp_path, speed_fn, seconds=12.0, s0=1.0):
    export = tmp_path / "export"
    export.mkdir(parents=True)
    t = np.arange(0.0, seconds, 0.01)
    v = speed_fn(t)
    s = s0 + np.concatenate([[0.0], np.cumsum(v[:-1]) * 0.01])
    ts = (t * 1e6 + 1e7).astype(np.int64)
    wheel = {"contact_point_trajectory": {"poses": [[int(a), [0.0] * 7] for a in ts]},
             "wheel_excitement": [[int(a), 0, float(b), 0.0] for a, b in zip(ts, s)]}
    (export / "wheel_excitement.json").write_text(json.dumps({"wheel_results": {"cariad_wheel_FL": wheel}}))
    return tmp_path, s - s0


def test_without_an_offset_the_replay_is_the_recorded_drive(tmp_path):
    win, _ = _window(tmp_path, lambda t: 5.0 + 0.0 * t)
    legacy, n_legacy = human_speeds(win, 30.0)
    shifted, n_shifted, t_start = human_replay(win, 30.0, 0.0)
    assert t_start == 0.0 and n_legacy == n_shifted
    assert np.allclose(legacy, shifted)


def test_the_replay_starts_where_the_road_starts(tmp_path):
    win, _ = _window(tmp_path, lambda t: 5.0 + 0.0 * t)
    v, n_rec, t_start = human_replay(win, 30.0, 10.0)
    assert abs(t_start - 2.0) < 0.05
    assert abs(v[0] - 5.0) < 0.05


def test_the_shifted_drive_covers_the_recording_after_the_offset(tmp_path):
    win, s = _window(tmp_path, lambda t: 3.0 + 0.5 * t)
    offset = 6.0
    v, n_rec, _ = human_replay(win, 200.0, offset)
    assert abs(np.sum(v[:n_rec]) * DT - (s[-1] - offset)) < 0.3


def test_the_speed_at_the_start_is_the_speed_when_the_drive_reaches_the_offset(tmp_path):
    win, _ = _window(tmp_path, lambda t: 3.0 + 0.5 * t)
    full, _, _ = human_replay(win, 200.0, 0.0)
    _, _, t_start = human_replay(win, 200.0, 6.0)
    v, _, _ = human_replay(win, 200.0, 6.0)
    assert abs(v[0] - full[int(round(t_start / DT))]) < 0.05


def test_an_offset_beyond_the_recorded_drive_is_an_error(tmp_path):
    win, s = _window(tmp_path, lambda t: 5.0 + 0.0 * t)
    with pytest.raises(ValueError):
        human_replay(win, 30.0, float(s[-1]) + 5.0)


def _scenario(tmp_path, prior=None):
    z = [0.05] * 10 + [0.0] * 60 + [0.0, 0.01, 0.03, 0.05, 0.06, 0.05, 0.03, 0.01] + [0.0] * 60
    meta = {"trim": {"offset_m": prior}} if prior is not None else {}
    d = {"name": "synthetic_sb01", "meta": meta, "road": {"ds_m": 0.05, "z": z},
         "bumps": [{"s_bump_m": 3.9, "s_start_m": 3.7, "s_end_m": 4.1, "peak_z_m": 0.06, "width_m": 0.4}]}
    p = tmp_path / "synthetic_sb01.json"
    p.write_text(json.dumps(d))
    return p


def test_the_loader_reports_how_much_road_it_removed(tmp_path):
    road, _ = load_scenario_json(_scenario(tmp_path))
    assert road._trim_offset_m > 0.0
    assert abs(road._profile[1][0]) <= 0.01


def test_a_file_that_was_already_trimmed_keeps_its_recorded_offset(tmp_path):
    road, _ = load_scenario_json(_scenario(tmp_path, prior=2.85))
    again, _ = load_scenario_json(_scenario(tmp_path))
    assert abs(road._trim_offset_m - (again._trim_offset_m + 2.85)) < 1e-9
