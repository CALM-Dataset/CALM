from dataclasses import replace

import numpy as np

from gym_env.common import TerminationWrapper
from gym_env.common.reward.reward_params import load_reward_config
from gym_env.common.wrappers.relative_comfort import RelativeComfort, _COMFORT_KEYS
from gym_env.half_car.envs import HalfCarEnv
from road.road_generator import RoadGenerator


def _make():
    return TerminationWrapper(HalfCarEnv(action_mode="slowdown"))


def _hold_return(road, cfg):
    env = RelativeComfort(_make(), _make, reward_config=cfg)
    env.reset(seed=0, options={"road": road})
    comfort, steps, done = 0.0, 0, False
    while not done:
        _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        comfort += sum(info[k] for k in _COMFORT_KEYS) * env._comfort_gain
        steps += 1
        done = terminated or truncated
    return comfort, env._time_step_cost * steps


def _road(height):
    return RoadGenerator.from_scenario(vehicle_speed=6.0, flat_start=10.0, obstacles=[(height, 2.0)])


def test_holding_speed_costs_the_same_on_any_road():
    cfg = replace(load_reward_config(), rel_scale=20.0, rel_time_price=0.0, rel_min_cost=1e-6)
    low, _ = _hold_return(_road(0.04), cfg)
    high, _ = _hold_return(_road(0.10), cfg)
    assert abs(low + 20.0) < 1.0
    assert abs(high + 20.0) < 1.0


def test_holding_speed_pays_the_time_price_once():
    cfg = replace(load_reward_config(), rel_scale=20.0, rel_time_price=0.5, rel_min_cost=1e-6)
    _, time_cost = _hold_return(_road(0.06), cfg)
    assert abs(time_cost - 0.5 * 20.0) < 0.5


def test_wrapper_is_inert_when_switched_off():
    cfg = replace(load_reward_config(), rel_scale=0.0)
    plain = _make()
    wrapped = RelativeComfort(_make(), _make, reward_config=cfg)
    road_a, road_b = _road(0.06), _road(0.06)
    plain.reset(seed=1, options={"road": road_a})
    wrapped.reset(seed=1, options={"road": road_b})
    for _ in range(50):
        a = plain.step(np.zeros(1, dtype=np.float32))[1]
        b = wrapped.step(np.zeros(1, dtype=np.float32))[1]
        assert abs(a - b) < 1e-9
