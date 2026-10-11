from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

__import__("gym_env.quarter_car.envs")
from gym_env.quarter_car.wrappers import (
    EpisodeLogger, PreviewWrapper, CurriculumWrapper, TerminationWrapper,
    TerminalRewardWrapper,
)

ENV_ID = "gym_env/QuarterCar"


def _vehicle_bits(vehicle: str):
    if vehicle == "half":
        __import__("gym_env.half_car.envs")
        from gym_env.half_car.wrappers import (
            EpisodeLogger as EL, PreviewWrapper as PW,
            CurriculumWrapper as CW, TerminationWrapper as TW,
            TerminalRewardWrapper as TRW,
        )
        return "gym_env/HalfCar", (EL, PW, CW, TW, TRW)
    if vehicle not in ("quarter", None):
        raise ValueError(f"unknown vehicle {vehicle!r} (expected 'quarter' or 'half')")
    return ENV_ID, (EpisodeLogger, PreviewWrapper, CurriculumWrapper,
                    TerminationWrapper, TerminalRewardWrapper)


class ActionSaturationPenalty(gym.Wrapper):
    def __init__(self, env: gym.Env, coef: float):
        super().__init__(env)
        self._coef = float(coef)
        self._low = env.action_space.low
        self._high = env.action_space.high
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=env.action_space.shape, dtype=np.float32)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        over = np.maximum(a - self._high, 0.0) + np.maximum(self._low - a, 0.0)
        obs, reward, terminated, truncated, info = self.env.step(np.clip(a, self._low, self._high))
        return obs, reward - self._coef * float(np.sum(over ** 2)), terminated, truncated, info


def _make_env(
    seed: int,
    monitor_dir: str | None,
    env_kwargs: dict,
    curriculum_cfg: dict,
    n_envs: int = 1,
    sparse_reward: bool = False,
    vehicle: str = "half",
    action_penalty: float = 0.0,
) -> Callable[[], gym.Env]:

    env_id, (EpisodeLogger, PreviewWrapper, CurriculumWrapper,
             TerminationWrapper, TerminalRewardWrapper) = _vehicle_bits(vehicle)

    def _init() -> gym.Env:
        env = gym.make(env_id, **env_kwargs)
        env = TerminationWrapper(env)
        if sparse_reward:
            env = TerminalRewardWrapper(env)
        env = PreviewWrapper(env)
        env = CurriculumWrapper(env, curriculum_cfg, n_envs=n_envs)
        if action_penalty > 0.0:
            env = ActionSaturationPenalty(env, action_penalty)
        if monitor_dir:
            Path(monitor_dir).mkdir(parents=True, exist_ok=True)
            env = Monitor(env, str(Path(monitor_dir) / f"env{seed}"))
            env = EpisodeLogger(env, log_dir=monitor_dir, tag=f"env{seed}")
        else:
            env = Monitor(env, None)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return _init


def make_vec_env(
    n_envs: int,
    base_seed: int,
    curriculum_cfg: dict,
    monitor_dir: str | None = None,
    gamma: float = 0.99,
    norm_obs: bool = True,
    norm_reward: bool = True,
    env_kwargs: dict | None = None,
    sparse_reward: bool = False,
    vehicle: str = "half",
    action_penalty: float = 0.0,
) -> VecNormalize:
    env_kwargs = env_kwargs or {}
    fns = [
        _make_env(base_seed + i, monitor_dir, env_kwargs, curriculum_cfg, n_envs, sparse_reward, vehicle, action_penalty)
        for i in range(n_envs)
    ]
    venv = DummyVecEnv(fns)
    venv = VecNormalize(
        venv,
        norm_obs=norm_obs,
        norm_reward=norm_reward,
        gamma=gamma,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    return venv


def make_eval_vec_env(
    n_envs: int,
    base_seed: int,
    train_venv: VecNormalize,
    curriculum_cfg: dict,
    monitor_dir: str | None = None,
    env_kwargs: dict | None = None,
    sparse_reward: bool = False,
    vehicle: str = "half",
) -> VecNormalize:
    env_kwargs = env_kwargs or {}
    fns = [
        _make_env(base_seed + i, monitor_dir, env_kwargs, curriculum_cfg, n_envs, sparse_reward, vehicle)
        for i in range(n_envs)
    ]
    venv = DummyVecEnv(fns)
    eval_venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=False,
        gamma=train_venv.gamma,
        clip_obs=train_venv.clip_obs,
    )
    eval_venv.obs_rms  = train_venv.obs_rms
    eval_venv.ret_rms  = train_venv.ret_rms
    eval_venv.training = False

    try:
        eval_venv.venv.env_method("set_forced_level", 0)
    except AttributeError:
        pass

    return eval_venv


def load_real_val_scenarios(real_dir: str | Path | None, n: int = 20, seed: int = 0) -> list[tuple]:
    import numpy as np
    from utils.scenario_loader import load_scenario_json

    if not real_dir or not Path(real_dir).is_dir():
        return []
    rsd = Path(real_dir)
    files = sorted(p for p in rsd.glob("*.json")
                   if p.name not in ("manifest.json", "curriculum.json"))
    if not files:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    out = []
    for i in sorted(idx):
        road, track = load_scenario_json(str(files[i]))
        out.append((files[i].stem, road, track))
    return out


def make_real_val_env(vehicle: str, seed: int = 0) -> VecNormalize:
    env_id, (_, PreviewW, _, TerminationW, _) = _vehicle_bits(vehicle)

    def _init() -> gym.Env:
        env = gym.make(env_id, random_road_on_reset=False)
        env = TerminationW(env)
        env = PreviewW(env)
        env.reset(seed=seed)
        return env

    venv = DummyVecEnv([_init])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.training = False
    return venv


def restore_vecnormalize(venv: VecNormalize, path: str) -> bool:
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        return False
    saved = VecNormalize.load(str(p), venv.venv)
    venv.obs_rms = saved.obs_rms
    venv.ret_rms = saved.ret_rms
    return True
