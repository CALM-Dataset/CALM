
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO


class DummyAgent(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pin_zero()

    def _pin_zero(self) -> None:
        policy = getattr(self, "policy", None)
        if policy is None:
            return
        import torch
        with torch.no_grad():
            if hasattr(policy, "action_net"):
                policy.action_net.weight.zero_()
                policy.action_net.bias.zero_()
            if hasattr(policy, "log_std"):
                policy.log_std.fill_(-20.0)

    def predict(self, observation, state=None, episode_start=None,
                deterministic=True):
        actions, state = super().predict(
            observation, state, episode_start, deterministic)
        return np.zeros_like(actions), state
