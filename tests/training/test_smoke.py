import numpy as np

from gym_env.common.wrappers.curriculum import load_curriculum_config
from train.agent import build_model
from train.environment import make_vec_env
from utils.paths import ROOT


def test_short_training_run():
    venv = make_vec_env(1, 0, load_curriculum_config(ROOT / "config" / "curriculum" / "curriculum.yaml"),
                        action_penalty=0.0)
    kwargs = {"policy": "MlpPolicy", "n_steps": 256, "batch_size": 128, "n_epochs": 1,
              "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}}}
    model = build_model("PPO", venv, kwargs, tensorboard_log=None, seed=0, verbose=0)
    model.learn(total_timesteps=512)
    action, _ = model.predict(venv.reset(), deterministic=True)
    assert all(venv.action_space.contains(a) for a in action)
    venv.close()


def test_slowdown_mode_charges_positive_actions_and_clips_them_to_hold():
    from gym_env.half_car.envs import HalfCarEnv
    from train.environment import ActionSaturationPenalty

    def one_step(u):
        env = ActionSaturationPenalty(HalfCarEnv(action_mode="slowdown"), 1.0, high=0.0)
        env.reset(seed=2)
        return env.step(np.full(1, u, dtype=np.float32))[1], env.unwrapped._v

    r_hold, v_hold = one_step(0.0)
    r_pos, v_pos = one_step(0.8)
    assert r_pos < r_hold - 0.3
    assert v_pos == v_hold
