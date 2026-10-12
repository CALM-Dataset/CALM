import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gym_env.vehicles import get_vehicle

VEHICLES = ["quarter", "half"]


def make(vehicle):
    return gym.make(get_vehicle(vehicle).env_id)


@pytest.mark.parametrize("vehicle", VEHICLES)
def test_the_environment_follows_the_gymnasium_api(vehicle):
    check_env(make(vehicle).unwrapped, skip_render_check=True)


@pytest.mark.parametrize("vehicle", VEHICLES)
def test_a_seed_reproduces_the_episode(vehicle):
    runs = []
    for _ in range(2):
        env = make(vehicle)
        obs, _ = env.reset(seed=3)
        rewards = []
        for _ in range(100):
            obs, r, *_ = env.step(np.zeros(1, dtype=np.float32))
            rewards.append(r)
        runs.append((obs, rewards))
    assert np.allclose(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]


@pytest.mark.parametrize("vehicle", VEHICLES)
def test_random_driving_always_ends_the_episode(vehicle):
    env = make(vehicle)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(20000):
        _, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
        if terminated or truncated:
            return
    pytest.fail("the episode never ended")
