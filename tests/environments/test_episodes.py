import gymnasium as gym
import numpy as np

from gym_env.common.reward.reward_params import load_reward_config
from gym_env.half_car.envs import HalfCarEnv
from gym_env.common.reward.reward import compute_reward
from gym_env.common import TerminationWrapper
from utils.paths import ROOT


def _run(env, action, steps=6000):
    env.reset(seed=0)
    for _ in range(steps):
        _, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        if terminated or truncated:
            return terminated, truncated, info, reward
    raise AssertionError("episode did not end")


def test_reaching_the_end_terminates():
    terminated, truncated, info, _ = _run(TerminationWrapper(gym.make("gym_env/HalfCar")), 0.0)
    assert terminated and not truncated
    assert info["termination_reason"] == "completed"


def test_running_out_of_time_truncates():
    terminated, truncated, info, _ = _run(TerminationWrapper(HalfCarEnv(max_episode_steps=50)), 0.0)
    assert truncated and not terminated
    assert info["termination_reason"] == "timeout"


def test_a_crash_terminates():
    env = TerminationWrapper(HalfCarEnv())
    env.reset(seed=0)
    env.unwrapped._state[6] = 1.0
    _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
    assert terminated and not truncated
    assert info["termination_reason"] == "crash_susp"


def test_observations_stay_in_the_space():
    env = TerminationWrapper(gym.make("gym_env/HalfCar"))
    rng = np.random.default_rng(0)
    low, high = env.observation_space.low, env.observation_space.high
    for episode in range(6):
        obs, _ = env.reset(seed=episode)
        for _ in range(800):
            obs, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1, 1).astype(np.float32))
            assert np.all(obs >= low) and np.all(obs <= high)
            if terminated or truncated:
                break


def test_body_acceleration_costs_reward():
    cfg = load_reward_config()
    calm, _ = compute_reward(5.0, 0.0, 0.0, 0.0, 0.0, cfg, v_ref=5.0)
    rough, _ = compute_reward(5.0, 5.0, 5.0, 0.0, 0.0, cfg, v_ref=5.0)
    assert rough < calm


def test_same_seed_same_training_episodes():

    from gym_env.common.wrappers.curriculum import load_curriculum_config
    from train.environment import _make_env

    cfg = load_curriculum_config(ROOT / "config" / "curriculum" / "curriculum.yaml")

    def rollout(seed):
        env = _make_env(seed, None, {"render_mode": "none"}, cfg)()
        rng = np.random.default_rng(0)
        out = []
        for _ in range(4):
            out.append(env.reset()[0])
            for _ in range(30):
                out.append(env.step(rng.uniform(-1, 1, 1).astype(np.float32))[0])
        return np.array(out)

    assert np.array_equal(rollout(5), rollout(5))
    assert not np.array_equal(rollout(5), rollout(6))


def test_slowdown_mode_never_exceeds_scenario_speed_and_can_slow():
    env = gym.make("gym_env/HalfCar", action_mode="slowdown")
    env.reset(seed=3)
    top = env.unwrapped._v_episode_ref
    speeds = []
    for u in [1.0] * 20 + [-1.0] * 60:
        _, _, terminated, truncated, info = env.step(np.full(1, u, dtype=np.float32))
        speeds.append(info["speed"])
        if terminated or truncated:
            break
    assert max(speeds) <= top + 1e-9
    assert min(speeds) < top


def test_time_cost_makes_slowing_down_cost_reward():
    from dataclasses import replace
    cfg = replace(load_reward_config(), time_cost=10.0, track_scale=0.0)
    fast, _ = compute_reward(10.0, 0.0, 0.0, 0.0, 0.0, cfg, v_ref=10.0, dt=0.02)
    slow, _ = compute_reward(5.0, 0.0, 0.0, 0.0, 0.0, cfg, v_ref=10.0, dt=0.02)
    assert fast == slow
    assert fast < 0.0
    free, _ = compute_reward(10.0, 0.0, 0.0, 0.0, 0.0, replace(cfg, time_cost=0.0), v_ref=10.0, dt=0.02)
    assert free > fast


def test_weighted_heave_ignores_a_constant_offset():
    from eval.iso2631 import WkStream
    f = WkStream(0.02)
    out = [f.step(3.0) for _ in range(2000)]
    assert abs(out[-1]) < 1e-3


def test_action_repeat_sums_rewards_and_advances_several_steps():
    from gym_env.common.wrappers.action_repeat import ActionRepeat

    one = HalfCarEnv(action_mode="slowdown")
    one.reset(seed=4)
    ref = sum(one.step(np.zeros(1, dtype=np.float32))[1] for _ in range(5))
    rep = ActionRepeat(HalfCarEnv(action_mode="slowdown"), 5)
    rep.reset(seed=4)
    _, reward, _, _, _ = rep.step(np.zeros(1, dtype=np.float32))
    assert abs(reward - ref) < 1e-9
    assert rep.unwrapped._step_count == 5


def test_action_repeat_can_expose_every_inner_step():
    from gym_env.common.wrappers.action_repeat import ActionRepeat

    rep = ActionRepeat(HalfCarEnv(action_mode="slowdown"), 4, record_inner=True)
    rep.reset(seed=4)
    _, reward, _, _, info = rep.step(np.zeros(1, dtype=np.float32))
    assert len(info["inner"]) == 4
    assert abs(sum(i["step_reward"] for i in info["inner"]) - reward) < 1e-9


def test_speed_floor_ratio_keeps_slowdown_above_the_floor(monkeypatch):
    import gym_env.common.car_env as ce

    monkeypatch.setattr(ce, "SPEED_FLOOR_RATIO", 0.7)
    env = HalfCarEnv(action_mode="slowdown")
    env.reset(seed=3)
    top = env._v_episode_ref
    low = min(env.step(np.full(1, -1.0, dtype=np.float32))[4]["speed"] for _ in range(600))
    assert low >= 0.7 * top - 1e-6


def test_depth_mode_holds_at_the_hold_action_and_slows_when_pushed():
    env = HalfCarEnv(action_mode="depth")
    env.reset(seed=3)
    top = env._v_episode_ref
    held = []
    for _ in range(600):
        env.step(env.hold_action)
        held.append(env._v)
    assert max(held) <= top + 1e-9
    assert held[-1] > top - 1e-3

    env.reset(seed=3)
    pushed = []
    for _ in range(400):
        env.step(np.array([env.action_space.high[0]], dtype=np.float32))
        pushed.append(env._v)
    assert min(pushed) < top * 0.5


def test_depth_mode_slows_more_for_larger_actions():
    def final_speed(a):
        env = HalfCarEnv(action_mode="depth")
        env.reset(seed=3)
        for _ in range(300):
            env.step(np.array([a], dtype=np.float32))
        return env._v
    assert final_speed(1.0) > final_speed(2.0) > final_speed(4.0)
