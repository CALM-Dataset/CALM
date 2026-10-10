from __future__ import annotations

import torch
from stable_baselines3.common.policies import ActorCriticPolicy


class TanhMeanPolicy(ActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = torch.tanh(self.action_net(latent_pi))
        return self.action_dist.proba_distribution(mean_actions, self.log_std)
