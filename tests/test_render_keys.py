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


def test_a_repeated_navigation_key_counts_once(raw_env):
    _press(raw_env, "left")
    assert raw_env._nav_request == "prev"
    raw_env._nav_request = None
    _press(raw_env, "left")
    assert raw_env._nav_request is None


def test_s_slows_down_and_f_speeds_up(raw_env):
    start = raw_env._render_speed
    _press(raw_env, "f")
    faster = raw_env._render_speed
    _press(raw_env, "s")
    _press(raw_env, "s")
    assert faster > start > raw_env._render_speed
