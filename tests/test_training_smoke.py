from pathlib import Path

import numpy as np

from gym_env.common.curriculum import load_curriculum_config
from train.agent import build_model
from train.environment import make_vec_env


def test_a_short_training_run_produces_a_bounded_policy():
    root = Path(__file__).resolve().parents[1]
    venv = make_vec_env(1, 0, load_curriculum_config(root / "config" / "curriculum" / "curriculum.yaml"),
                        action_penalty=0.1)
    kwargs = {"policy": "TanhMeanPolicy", "n_steps": 256, "batch_size": 128, "n_epochs": 1,
              "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}}}
    model = build_model("PPO", venv, kwargs, tensorboard_log=None, seed=0, verbose=0)
    model.learn(total_timesteps=512)
    action, _ = model.predict(venv.reset(), deterministic=True)
    assert np.all(np.abs(action) <= 1.0 + 1e-6)
    venv.close()


def test_evaluation_does_not_see_the_controller_weight():
    from gym_env.common.reward import get_track_scale_override, set_track_scale_override
    from train.monitoring import base_speed_weight

    set_track_scale_override(2.5)
    try:
        with base_speed_weight():
            assert get_track_scale_override() is None
        assert get_track_scale_override() == 2.5
    finally:
        set_track_scale_override(None)
