import gymnasium as gym
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from matplotlib.backend_bases import KeyEvent  # noqa: E402

from gym_env.vehicles import get_vehicle  # noqa: E402


@pytest.fixture
def raw_env():
    env = gym.make(get_vehicle("half").env_id, render_mode="human")
    env.reset(seed=0)
    for _ in range(3):
        env.step(np.zeros(1, dtype=np.float32))
        env.render()
    yield env.unwrapped
    env.close()


def _press(raw, key):
    KeyEvent("key_press_event", raw._fig.canvas, key)._process()


def test_key_repeat_counts_once(raw_env):
    _press(raw_env, "left")
    assert raw_env._nav_request == "prev"
    raw_env._nav_request = None
    _press(raw_env, "left")
    assert raw_env._nav_request is None


def test_speed_keys(raw_env):
    start = raw_env._render_speed
    _press(raw_env, "f")
    faster = raw_env._render_speed
    _press(raw_env, "s")
    _press(raw_env, "s")
    assert faster > start > raw_env._render_speed


def test_no_navigation_past_the_ends(raw_env):
    raw_env._nav_allowed = {"prev": False, "next": True}
    _press(raw_env, "left")
    assert raw_env._nav_request is None
    _press(raw_env, "right")
    assert raw_env._nav_request == "next"


def test_replay_is_always_allowed(raw_env):
    raw_env._nav_allowed = {"prev": False, "next": False}
    _press(raw_env, "r")
    assert raw_env._nav_request == "replay"


def test_new_episode_restarts_the_clock(raw_env):
    first_anchor = raw_env._pace[0]
    raw_env.reset(seed=1)
    raw_env.step(np.zeros(1, dtype=np.float32))
    raw_env.render()
    assert raw_env._pace[0] > first_anchor
    assert raw_env._pace[3] == raw_env._episode_count
