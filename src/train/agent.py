from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv


_REGISTRY: dict[str, type[BaseAlgorithm]] = {
    "PPO": PPO,
}


def supported_algos() -> list[str]:
    return list(_REGISTRY.keys())


def load_algo_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _parse_lr(lr_val):
    if isinstance(lr_val, str) and lr_val.startswith("lin_"):
        parts = lr_val[4:].split("_", 1)
        initial = float(parts[0])
        floor   = float(parts[1]) if len(parts) == 2 else 0.0
        return lambda p, i=initial, f=floor: f + (i - f) * p
    return lr_val


def _resolve_policy_kwargs(policy_kwargs: dict) -> dict:
    import torch.nn as nn
    import copy
    pk = copy.deepcopy(policy_kwargs)
    if "activation_fn" in pk and isinstance(pk["activation_fn"], str):
        pk["activation_fn"] = getattr(nn, pk["activation_fn"])
    return pk


def build_model(
    algo: str,
    venv: VecEnv,
    algo_kwargs: dict,
    tensorboard_log: str,
    seed: int,
    resume: str | None = None,
    verbose: int = 1,
) -> BaseAlgorithm:
    cls    = _REGISTRY[algo]
    kw     = copy.deepcopy(algo_kwargs)
    policy = kw.pop("policy", "MlpPolicy")
    if policy == "TanhMeanPolicy":
        from train.policy import TanhMeanPolicy
        policy = TanhMeanPolicy
    if "learning_rate" in kw:
        kw["learning_rate"] = _parse_lr(kw["learning_rate"])
    if "policy_kwargs" in kw:
        kw["policy_kwargs"] = _resolve_policy_kwargs(kw["policy_kwargs"])

    if resume:
        model = cls.load(
            resume,
            env=venv,
            seed=seed,
            tensorboard_log=tensorboard_log,
            **kw,
        )
    else:
        model = cls(
            policy,
            venv,
            verbose=verbose,
            seed=seed,
            tensorboard_log=tensorboard_log,
            **kw,
        )
    return model
