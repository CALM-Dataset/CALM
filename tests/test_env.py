import gymnasium as gym
import numpy as np

from gym_env.common.reward_params import load_reward_config
from gym_env.half_car.envs import HalfCarEnv
from gym_env.common.reward import compute_reward
from gym_env.common import TerminationWrapper


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


def test_target_speed_stays_below_scenario_speed():
    env = gym.make("gym_env/HalfCar", action_mode="target_speed")
    env.reset(seed=3)
    top = env.unwrapped._v_episode_ref
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(np.ones(1, dtype=np.float32))
        assert info["speed"] <= top + 1e-9
        if terminated or truncated:
            break


def test_body_acceleration_costs_reward():
    cfg = load_reward_config()
    calm, _ = compute_reward(5.0, 0.0, 0.0, 0.0, 0.0, cfg, v_ref=5.0)
    rough, _ = compute_reward(5.0, 5.0, 5.0, 0.0, 0.0, cfg, v_ref=5.0)
    assert rough < calm


def test_same_seed_same_training_episodes():
    from pathlib import Path

    from gym_env.common.curriculum import load_curriculum_config
    from train.environment import _make_env

    cfg = load_curriculum_config(Path(__file__).resolve().parents[1] / "config" / "curriculum" / "curriculum.yaml")

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
