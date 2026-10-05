from __future__ import annotations

import gymnasium as gym


class TerminalRewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._accum = 0.0

    def reset(self, **kwargs):
        self._accum = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._accum += reward

        if terminated or truncated:
            total = self._accum
            self._accum = 0.0
            return observation, total, terminated, truncated, info

        return observation, 0.0, terminated, truncated, info
