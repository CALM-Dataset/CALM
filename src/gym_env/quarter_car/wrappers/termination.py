from __future__ import annotations

import gymnasium as gym

from gym_env.quarter_car.config.env_params import (
    DT, TRUNC_SUSP_TRAVEL, TRUNC_BODY_DISP, TRUNC_DAMPER_VEL,
    POST_BUMP_MARGIN, POST_BUMP_MARGIN_MATCH_FLAT_START,
)
from gym_env.quarter_car.config.reward_params import RewardConfig, load_reward_config
from gym_env.quarter_car.reward.reward import compute_terminal_reward


class TerminationWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        damper_velocity_limit: float = TRUNC_DAMPER_VEL,
        susp_travel_limit: float = TRUNC_SUSP_TRAVEL,
        body_disp_limit: float = TRUNC_BODY_DISP,
        post_bump_margin: float = POST_BUMP_MARGIN,
        match_flat_start: bool  = POST_BUMP_MARGIN_MATCH_FLAT_START,
        reward_config: RewardConfig | None = None,
    ):
        super().__init__(env)
        self._cfg              = reward_config or load_reward_config()
        self._damper_limit     = float(damper_velocity_limit)
        self._susp_travel_lim  = float(susp_travel_limit)
        self._body_disp_lim    = float(body_disp_limit)
        self._post_bump_margin = float(post_bump_margin)
        self._match_flat_start = bool(match_flat_start)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            return observation, reward, terminated, truncated, info

        env = self.env.unwrapped

        if abs(float(env._state[2])) > self._susp_travel_lim:
            truncated = True
            reward += self._cfg.terminal_penalty
            env._termination_reason = "crash_susp"

        if not truncated and abs(float(env._state[5])) > self._body_disp_lim:
            truncated = True
            reward += self._cfg.terminal_penalty
            env._termination_reason = "crash_body"

        if not truncated:
            damper_velocity = float(env._state[1] - env._state[3])
            if abs(damper_velocity) > self._damper_limit:
                truncated = True
                reward += self._cfg.terminal_penalty
                env._termination_reason = "crash_damper"

        if not truncated:
            v = float(env._v)
            if v < self._cfg.terminal_v_min or v > self._cfg.terminal_v_max:
                truncated = True
                reward += self._cfg.terminal_penalty
                env._termination_reason = "crash_speed"

        if not terminated and not truncated:
            lbe = env._last_bump_end
            if lbe is not None:
                if self._match_flat_start and env._road._bumps:
                    margin = float(env._road._bumps[0][0])
                else:
                    margin = self._post_bump_margin
                if env._s_pos >= lbe + margin:
                    terminated = True
                    reward += compute_terminal_reward(
                        success=True, t=env._t,
                        t_max=env._max_episode_steps * DT, cfg=self._cfg,
                    )
                    env._termination_reason = "completed"

        if truncated or terminated:
            info["termination_reason"] = env._termination_reason

        return observation, reward, terminated, truncated, info
