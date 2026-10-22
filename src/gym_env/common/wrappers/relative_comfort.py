from __future__ import annotations

import copy
from typing import Callable

import gymnasium as gym
import numpy as np

from gym_env.common.env_params import DT
from gym_env.common.reward.reward_params import RewardConfig, load_reward_config

_COMFORT_KEYS = ("J_heave", "J_wheel", "J_pitch")


def road_key(road) -> tuple:
    prof = getattr(road, "_profile", None)
    prof_key = None if prof is None else tuple(float(np.sum(np.asarray(a, dtype=float))) for a in prof)
    return (
        round(float(road.speed), 4),
        tuple(tuple(round(float(x), 5) for x in b) for b in getattr(road, "_bumps", [])),
        prof_key,
    )


def hold_reference(make_env: Callable[[], gym.Env], road) -> tuple[float, float]:
    env = make_env()
    env.reset(options={"road": copy.deepcopy(road)})
    cost, done = 0.0, False
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    while not done:
        _, _, terminated, truncated, info = env.step(zero)
        cost -= sum(info.get(k, 0.0) for k in _COMFORT_KEYS)
        done = terminated or truncated
    t = float(env.unwrapped._t)
    env.close()
    return cost, t


class RelativeComfort(gym.Wrapper):

    def __init__(self, env: gym.Env, make_ref_env: Callable[[], gym.Env],
                 reward_config: RewardConfig | None = None):
        super().__init__(env)
        self._cfg = reward_config or load_reward_config()
        self._make_ref = make_ref_env
        self._cache: dict[tuple, tuple[float, float]] = {}
        self._comfort_gain = 1.0
        self._time_step_cost = 0.0

    def reset(self, **kwargs):
        road = (kwargs.get("options") or {}).get("road")
        if road is None or self._cfg.rel_scale <= 0.0:
            self._comfort_gain, self._time_step_cost = 1.0, 0.0
        else:
            key = road_key(road)
            if key not in self._cache:
                self._cache[key] = hold_reference(self._make_ref, road)
            cost, t_hold = self._cache[key]
            self._comfort_gain = self._cfg.rel_scale / max(cost, self._cfg.rel_min_cost)
            self._time_step_cost = self._cfg.rel_time_price * self._cfg.rel_scale * DT / max(t_hold, DT)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        comfort = sum(info.get(k, 0.0) for k in _COMFORT_KEYS)
        reward = reward + comfort * (self._comfort_gain - 1.0) - self._time_step_cost
        return obs, reward, terminated, truncated, info
