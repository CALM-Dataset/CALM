import gymnasium as gym

from gym_env.common.env_params import ACTION_REPEAT


class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, repeat: int = ACTION_REPEAT, record_inner: bool = False):
        super().__init__(env)
        self._repeat = max(int(repeat), 1)
        self._record_inner = record_inner

    def step(self, action):
        total = 0.0
        inner = []
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += reward
            if self._record_inner:
                inner.append({**info, "step_reward": float(reward)})
            if terminated or truncated:
                break
        if self._record_inner:
            info = {**info, "inner": inner}
        return obs, total, terminated, truncated, info


def with_action_repeat(env: gym.Env, record_inner: bool = False) -> gym.Env:
    return ActionRepeat(env, record_inner=record_inner) if ACTION_REPEAT > 1 else env
