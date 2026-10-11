from __future__ import annotations

import gymnasium as gym

from gym_env.half_car.config.env_params import (
    DT, TRUNC_SUSP_TRAVEL, TRUNC_BODY_DISP, TRUNC_DAMPER_VEL,
    POST_BUMP_MARGIN, POST_BUMP_MARGIN_MATCH_FLAT_START,
)
from gym_env.half_car.config.reward_params import RewardConfig, load_reward_config
from gym_env.half_car.reward.reward import compute_terminal_reward

_REAL_TAIL_MARGIN_M = 3.0


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
        st = env._state
        lf, lr = float(env._ode._lf), float(env._ode._lr)

        susp_travel = max(abs(float(st[6])), abs(float(st[4])))
        if susp_travel > self._susp_travel_lim:
            terminated = True
            reward += self._cfg.terminal_penalty
            env._termination_reason = "crash_susp"

        if not terminated and abs(float(st[9])) > self._body_disp_lim:
            terminated = True
            reward += self._cfg.terminal_penalty
            env._termination_reason = "crash_body"

        if not terminated:
            damper_f = float(st[3]) - (float(st[5]) - lf * float(st[7]))
            damper_r = float(st[1]) - (float(st[5]) + lr * float(st[7]))
            if max(abs(damper_f), abs(damper_r)) > self._damper_limit:
                terminated = True
                reward += self._cfg.terminal_penalty
                env._termination_reason = "crash_damper"

        if not terminated:
            v = float(env._v)
            if v < self._cfg.terminal_v_min or v > self._cfg.terminal_v_max:
                terminated = True
                reward += self._cfg.terminal_penalty
                env._termination_reason = "crash_speed"

        if not terminated:
            road_end = getattr(env, "_road_end", None)
            if road_end is not None:
                complete_at = road_end + float(env._ode._L) + _REAL_TAIL_MARGIN_M
            else:
                lbe = env._last_bump_end
                if lbe is not None:
                    if self._match_flat_start and env._road._bumps:
                        margin = float(env._road._bumps[0][0])
                    else:
                        margin = self._post_bump_margin
                    complete_at = lbe + margin
                else:
                    complete_at = None
            if complete_at is not None and env._s_pos >= complete_at:
                terminated = True
                reward += compute_terminal_reward(
                    success=True, t=env._t,
                    t_max=env._max_episode_steps * DT, cfg=self._cfg,
                )
                env._termination_reason = "completed"

        if truncated or terminated:
            info["termination_reason"] = env._termination_reason

        return observation, reward, terminated, truncated, info
