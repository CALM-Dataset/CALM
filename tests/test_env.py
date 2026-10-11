import gymnasium as gym
import numpy as np

from gym_env.half_car.envs import HalfCarEnv
from gym_env.half_car.reward.reward import compute_reward
from gym_env.half_car.config.reward_params import load_reward_config
from gym_env.half_car.reward.utils import reward_bounds
from gym_env.half_car.wrappers import TerminationWrapper


def _run(env, action, steps=6000):
    env.reset(seed=0)
    for _ in range(steps):
        _, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        if terminated or truncated:
            return terminated, truncated, info, reward
    raise AssertionError("episode did not end")


def test_completion_is_terminated():
    terminated, truncated, info, _ = _run(TerminationWrapper(gym.make("gym_env/HalfCar")), 0.0)
    assert terminated and not truncated
    assert info["termination_reason"] == "completed"


def test_timeout_is_truncated_with_the_penalty():
    env = TerminationWrapper(HalfCarEnv(max_episode_steps=50))
    terminated, truncated, info, reward = _run(env, 0.0)
    assert truncated and not terminated
    assert info["termination_reason"] == "timeout"
    assert reward < -200.0


def test_crash_is_terminated():
    env = TerminationWrapper(HalfCarEnv())
    env.reset(seed=0)
    env.unwrapped._state[6] = 1.0
    _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
    assert terminated and not truncated
    assert info["termination_reason"] == "crash_susp"


def test_observations_stay_inside_the_space_for_random_actions():
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


def test_speed_never_exceeds_the_scenario_speed_in_target_mode():
    env = gym.make("gym_env/HalfCar", action_mode="target_speed")
    env.reset(seed=3)
    top = env.unwrapped._v_episode_ref
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(np.ones(1, dtype=np.float32))
        assert info["speed"] <= top + 1e-9
        if terminated or truncated:
            break


def test_comfort_terms_lower_the_reward():
    cfg = load_reward_config()
    calm, _ = compute_reward(5.0, 0.0, 0.0, 0.0, 0.0, cfg, v_ref=5.0)
    rough, _ = compute_reward(5.0, 5.0, 5.0, 0.0, 0.0, cfg, v_ref=5.0)
    assert rough < calm <= 0.0 + cfg.Q_step


def test_reward_bounds_are_finite_and_ordered():
    b = reward_bounds(load_reward_config(), 4000, n_bumps=1)
    assert np.isfinite(b["episode_min"]) and b["episode_min"] < b["episode_max"]
