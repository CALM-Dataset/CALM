import numpy as np

from gym_env.common import PreviewWrapper, TerminationWrapper
from gym_env.common.wrappers.decision import DecisionWrapper, with_decisions
from gym_env.half_car.envs import HalfCarEnv
from road.road_generator import RoadGenerator


def _road(bumps=1):
    obstacles = [(0.06, 2.0)] * bumps
    return RoadGenerator.from_scenario(vehicle_speed=6.0, flat_start=60.0, obstacles=obstacles, gaps=[60.0] * (bumps - 1))


def _raw_return(action, road):
    env = TerminationWrapper(HalfCarEnv(action_mode="choice"))
    env.reset(seed=1, options={"road": road})
    total, steps, done = 0.0, 0, False
    while not done:
        _, reward, terminated, truncated, _ = env.step(np.array([action]))
        total += reward
        steps += 1
        done = terminated or truncated
    return total, steps


def _wrapped(road, record_inner=False):
    env = DecisionWrapper(TerminationWrapper(HalfCarEnv(action_mode="choice")), record_inner=record_inner)
    obs, info = env.reset(seed=1, options={"road": road})
    return env, obs, info


def test_other_action_modes_are_left_alone():
    env = TerminationWrapper(HalfCarEnv(action_mode="slowdown"))
    assert with_decisions(env) is env


def test_a_single_bump_is_one_decision():
    env, _, _ = _wrapped(_road())
    _, _, terminated, truncated, _ = env.step(np.array([3]))
    assert terminated and not truncated


def test_each_bump_is_one_decision():
    env, _, _ = _wrapped(_road(bumps=2))
    decisions, done = 0, False
    while not done:
        _, _, terminated, truncated, _ = env.step(np.array([3]))
        decisions += 1
        done = terminated or truncated
    assert decisions == 2


def test_the_decision_return_equals_the_raw_episode_return():
    for action in (0, 2, 5):
        raw_total, _ = _raw_return(action, _road())
        env, _, _ = _wrapped(_road())
        _, reward, _, _, _ = env.step(np.array([action]))
        assert abs(reward - raw_total) < 1e-9


def test_the_inner_records_cover_every_step():
    action = 4
    _, raw_steps = _raw_return(action, _road())
    env, _, reset_info = _wrapped(_road(), record_inner=True)
    _, _, _, _, info = env.step(np.array([action]))
    assert len(reset_info.get("inner", [])) + len(info["inner"]) == raw_steps


def test_the_observation_at_a_decision_shows_the_bump_ahead():
    env = PreviewWrapper(TerminationWrapper(HalfCarEnv(action_mode="choice")))
    wrapped = DecisionWrapper(env)
    obs, _ = wrapped.reset(seed=1, options={"road": _road()})
    at_start = PreviewWrapper(TerminationWrapper(HalfCarEnv(action_mode="choice")))
    first, _ = at_start.reset(seed=1, options={"road": _road()})
    assert not np.array_equal(obs, first)
