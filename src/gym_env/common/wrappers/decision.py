import gymnasium as gym


def _at_decision(env: gym.Env) -> bool:
    inner = env.unwrapped
    return inner._slowdown_window(inner._s_pos) > 0.0 and not inner._choice_latched


class DecisionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, record_inner: bool = False):
        super().__init__(env)
        self._record_inner = record_inner
        self._carry = 0.0
        self._ended = None

    def _advance(self, action, until_decision: bool):
        total, inner = 0.0, []
        obs = info = None
        terminated = truncated = False
        while True:
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += reward
            if self._record_inner:
                inner.append({**info, "step_reward": float(reward)})
            if terminated or truncated or (until_decision and _at_decision(self.env)):
                break
        return obs, total, terminated, truncated, info, inner

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        hold = self.env.unwrapped.hold_action
        self._carry, self._ended = 0.0, None
        inner = []
        while not _at_decision(self.env):
            obs, reward, terminated, truncated, step_info = self.env.step(hold)
            self._carry += reward
            if self._record_inner:
                inner.append({**step_info, "step_reward": float(reward)})
            info = step_info
            if terminated or truncated:
                self._ended = (terminated, truncated)
                break
        if self._record_inner:
            info = {**info, "inner": inner}
        return obs, info

    def step(self, action):
        if self._ended is not None:
            terminated, truncated = self._ended
            obs = self.env.unwrapped._obs()
            return obs, self._take_carry(), terminated, truncated, {}
        obs, total, terminated, truncated, info, inner = self._advance(action, until_decision=True)
        total += self._take_carry()
        if self._record_inner:
            info = {**info, "inner": inner}
        return obs, total, terminated, truncated, info

    def _take_carry(self) -> float:
        carry, self._carry = self._carry, 0.0
        return carry


def with_decisions(env: gym.Env, record_inner: bool = False) -> gym.Env:
    if env.unwrapped._action_mode == "choice":
        return DecisionWrapper(env, record_inner=record_inner)
    return env
