import numpy as np

from train.agent import build_model
from train.environment import make_vec_env
from gym_env.half_car.wrappers.curriculum import load_curriculum_config
from pathlib import Path


def test_a_short_training_run_with_the_tanh_policy(tmp_path):
    root = Path(__file__).resolve().parents[1]
    venv = make_vec_env(1, 0, load_curriculum_config(root / "config" / "curriculum" / "curriculum.yaml"),
                        action_penalty=0.1)
    kwargs = {"policy": "TanhMeanPolicy", "n_steps": 256, "batch_size": 128, "n_epochs": 1,
              "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}}}
    model = build_model("PPO", venv, kwargs, tensorboard_log=None, seed=0, verbose=0)
    model.learn(total_timesteps=512)
    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert np.all(np.abs(action) <= 1.0 + 1e-6)
    venv.close()
