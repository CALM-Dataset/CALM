from __future__ import annotations

from pathlib import Path

import optuna
import yaml
from stable_baselines3.common.evaluation import evaluate_policy

from utils.paths import ROOT as _ROOT

from train.agent import build_model, load_algo_config
from train.environment import make_eval_vec_env, make_vec_env
from train.monitoring import SpeedWeightController
from tune.search_spaces import sample_from_config

_TUNE_CONFIG_PATH = _ROOT / "config" / "agent" / "tune_config.yaml"
_ALGO_CONFIG_PATH = _ROOT / "config" / "agent" / "algo_configs.yaml"
_CURRICULUM_PATH  = _ROOT / "config" / "curriculum" / "curriculum.yaml"


def _load_space(config_path: Path = _TUNE_CONFIG_PATH) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh).get("PPO", {})


class Objective:

    def __init__(
        self,
        timesteps: int,
        n_eval_episodes: int,
        seed: int,
        use_curriculum: bool = True,
        config_path: Path = _TUNE_CONFIG_PATH,
    ):
        self.timesteps       = timesteps
        self.n_eval_episodes = n_eval_episodes
        self.seed            = seed
        self.use_curriculum  = use_curriculum
        self._space          = _load_space(config_path)
        self._base           = load_algo_config(_ALGO_CONFIG_PATH)

    def __call__(self, trial: optuna.Trial) -> float:
        params     = sample_from_config(trial, self._space)
        trial_seed = self.seed + trial.number
        meta       = self._base["training"]

        from gym_env.half_car.wrappers.curriculum import load_curriculum_config
        curriculum_cfg = load_curriculum_config(_CURRICULUM_PATH)

        algo_kwargs = {**self._base["PPO"], **params}
        algo_kwargs["policy_kwargs"] = {
            **self._base["PPO"].get("policy_kwargs", {}), **params.get("policy_kwargs", {}),
        }

        train_venv = make_vec_env(
            n_envs=1,
            base_seed=trial_seed,
            norm_obs=True,
            norm_reward=True,
            gamma=algo_kwargs.get("gamma", 0.99),
            curriculum_cfg=curriculum_cfg,
            action_penalty=float(meta.get("action_saturation_penalty", 0.0)),
        )
        if not self.use_curriculum:
            train_venv.venv.env_method("set_level", len(curriculum_cfg["levels"]) - 1)

        model = build_model("PPO", train_venv, algo_kwargs, tensorboard_log=None, seed=trial_seed, verbose=0)
        ctrl = meta.get("speed_controller")
        callback = SpeedWeightController(ctrl["target"], ctrl["init"], ctrl["gain"], ctrl["max"]) if ctrl else None
        model.learn(total_timesteps=self.timesteps, callback=callback)

        eval_venv = make_eval_vec_env(
            n_envs=1,
            base_seed=trial_seed + 5_000,
            train_venv=train_venv,
            curriculum_cfg=curriculum_cfg,
        )

        mean_return, _ = evaluate_policy(
            model, eval_venv,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            warn=False,
        )

        eval_venv.close()
        train_venv.close()

        return float(mean_return)
