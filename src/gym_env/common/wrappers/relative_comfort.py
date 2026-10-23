from __future__ import annotations

import copy
from typing import Callable

import gymnasium as gym
import numpy as np

from gym_env.common.env_params import DT
from gym_env.common.reward.reward_params import RewardConfig, load_reward_config

_HEAVE = "J_heave"
_FRONTIER_POINTS = 8


def road_key(road) -> tuple:
    prof = getattr(road, "_profile", None)
    prof_key = None if prof is None else tuple(float(np.sum(np.asarray(a, dtype=float))) for a in prof)
    return (
        round(float(road.speed), 4),
        tuple(tuple(round(float(x), 5) for x in b) for b in getattr(road, "_bumps", [])),
        prof_key,
    )


def hold_reference(make_env: Callable[[], gym.Env], road, speed: float | None = None) -> tuple[float, float]:
    env = make_env()
    ref_road = copy.deepcopy(road)
    if speed is not None:
        ref_road.set_speed(float(speed))
    env.reset(options={"road": ref_road})
    cost, done = 0.0, False
    hold = env.unwrapped.hold_action
    while not done:
        _, _, terminated, truncated, info = env.step(hold)
        cost -= info.get(_HEAVE, 0.0)
        done = terminated or truncated
    t = float(env.unwrapped._t)
    env.close()
    return cost, t


def constant_speed_frontier(make_env: Callable[[], gym.Env], road, v_low: float) -> list[tuple[float, float]]:
    v_top = float(road.speed)
    speeds = np.linspace(min(v_low, v_top), v_top, _FRONTIER_POINTS)
    pts = []
    for v in speeds:
        cost, t = hold_reference(make_env, road, speed=float(v))
        pts.append((t, cost))
    pts.sort()
    return pts


class RelativeComfort(gym.Wrapper):

    def __init__(self, env: gym.Env, make_ref_env: Callable[[], gym.Env],
                 reward_config: RewardConfig | None = None):
        super().__init__(env)
        self._cfg = reward_config or load_reward_config()
        self._make_ref = make_ref_env
        self._cache: dict[tuple, list[tuple[float, float]]] = {}
        self._frontier: list[tuple[float, float]] = []
        self._comfort_gain = 1.0
        self._time_step_cost = 0.0
        self._j_hold = 1.0
        self._j_ep = 0.0
        self._t_hold = 1.0

    def reset(self, **kwargs):
        road = (kwargs.get("options") or {}).get("road")
        self._j_ep = 0.0
        self._frontier = []
        if road is None or self._cfg.rel_scale <= 0.0:
            self._comfort_gain, self._time_step_cost = 1.0, 0.0
        else:
            key = road_key(road)
            if key not in self._cache:
                self._cache[key] = constant_speed_frontier(self._make_ref, road, self._cfg.v_hard_min)
            pts = self._cache[key]
            t_hold, j_hold = pts[0][0], pts[0][1]
            self._frontier = pts
            self._t_hold = max(t_hold, DT)
            self._j_hold = max(j_hold, self._cfg.rel_min_cost)
            self._comfort_gain = self._cfg.rel_scale / self._j_hold
            self._time_step_cost = self._cfg.rel_time_price * self._cfg.rel_scale * DT / max(t_hold, DT)
        return self.env.reset(**kwargs)

    def _equal_time_cost(self, t: float) -> float:
        ts = [p[0] for p in self._frontier]
        js = [p[1] for p in self._frontier]
        return max(float(np.interp(t, ts, js)), self._cfg.rel_min_cost)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        heave = info.get(_HEAVE, 0.0)
        self._j_ep -= heave
        reward = reward + heave * (self._comfort_gain - 1.0) - self._time_step_cost
        if (terminated or truncated) and self._frontier and self._cfg.rel_frontier \
                and self.env.unwrapped._termination_reason == "completed":
            t = float(self.env.unwrapped._t)
            reward -= self._cfg.rel_scale * self._j_ep * (1.0 / self._equal_time_cost(t) - 1.0 / self._j_hold)
            late = max(t / self._t_hold - 1.0, 0.0)
            reward -= self._cfg.rel_time_quad * self._cfg.rel_scale * late ** 2
        return obs, reward, terminated, truncated, info
