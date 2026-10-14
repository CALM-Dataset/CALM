import dataclasses

from baseline.mpc.config import load_mpc_cfg, reward_config
from gym_env.common.reward.reward_params import load_reward_config


def test_mpc_weights_are_independent_of_ppo():
    mpc_weights = load_mpc_cfg()["weights"]

    def altered_reward_config():
        base = load_reward_config()
        return dataclasses.replace(base, **{key: value + 7.0 for key, value in mpc_weights.items()
                                            if hasattr(base, key)})

    cfg = reward_config(altered_reward_config)
    for key, value in mpc_weights.items():
        assert getattr(cfg, key) == value
