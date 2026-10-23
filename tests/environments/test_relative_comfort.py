from dataclasses import replace

import numpy as np

from gym_env.common import TerminationWrapper
from gym_env.common.reward.reward_params import load_reward_config
from gym_env.common.wrappers.relative_comfort import RelativeComfort
from gym_env.half_car.envs import HalfCarEnv
from road.road_generator import RoadGenerator


def _make(cfg=None):
    return TerminationWrapper(HalfCarEnv(action_mode="slowdown", reward_config=cfg))


def _hold_return(road, cfg):
    env = RelativeComfort(_make(cfg), lambda: _make(cfg), reward_config=cfg)
    env.reset(seed=0, options={"road": road})
    comfort, steps, done = 0.0, 0, False
    while not done:
        _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
        comfort += sum(info[k] for k in ("J_heave",)) * env._comfort_gain
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


def _episode_comfort(road, cfg, speed_action):
    env = RelativeComfort(_make(cfg), lambda: _make(cfg), reward_config=cfg)
    env.reset(seed=0, options={"road": road})
    total, done = 0.0, False
    base = 0.0
    while not done:
        _, r, terminated, truncated, info = env.step(np.array([speed_action], dtype=np.float32))
        base += r
        done = terminated or truncated
    return base


def test_holding_speed_matches_the_equal_time_reference():
    from dataclasses import replace as rep
    cfg = rep(load_reward_config(), rel_scale=20.0, rel_time_price=0.0, rel_min_cost=1e-6,
              rel_frontier=True, Q_a=0.0, w_jerk=0.0, w_action_smooth=0.0, time_cost=0.0,
              Q_pitch=0.0, Q_zWddot=0.0)
    on = _episode_comfort(_road(0.06), cfg, 0.0)
    off = _episode_comfort(_road(0.06), rep(cfg, rel_frontier=False), 0.0)
    assert abs(on - off) < 1.0


def test_a_constant_slower_speed_scores_the_same_as_holding():
    from dataclasses import replace as rep
    cfg = rep(load_reward_config(), rel_scale=20.0, rel_time_price=0.0, rel_min_cost=1e-6,
              rel_frontier=True, Q_a=0.0, w_jerk=0.0, w_action_smooth=0.0, time_cost=0.0,
              Q_pitch=0.0, Q_zWddot=0.0)
    bonus = cfg.terminal_bonus
    hold = _episode_comfort(_road(0.06), cfg, 0.0) - bonus
    slow = _episode_comfort(_road(0.06), cfg, -0.5) - bonus
    assert abs(hold + 20.0) < 2.0
    assert abs(slow + 20.0) < 4.0


def test_quadratic_time_cost_grows_faster_than_the_delay():
    from dataclasses import replace as rep
    cfg = rep(load_reward_config(), rel_scale=20.0, rel_time_price=0.0, rel_min_cost=1e-6,
              rel_frontier=True, Q_a=0.0, w_jerk=0.0, w_action_smooth=0.0, time_cost=0.0,
              Q_pitch=0.0, Q_zWddot=0.0)
    quad = rep(cfg, rel_time_quad=2.0)

    def extra(action):
        return _episode_comfort(_road(0.06), cfg, action) - _episode_comfort(_road(0.06), quad, action)

    def delay(action):
        env = RelativeComfort(_make(cfg), lambda: _make(cfg), reward_config=cfg)
        env.reset(seed=0, options={"road": _road(0.06)})
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(np.array([action], dtype=np.float32))
            done = terminated or truncated
        return env.unwrapped._t / env._t_hold - 1.0

    assert abs(extra(0.0)) < 1e-6
    mild, deep = extra(-0.3), extra(-0.8)
    assert 0.0 < mild < deep
    assert deep / mild > delay(-0.8) / delay(-0.3)
