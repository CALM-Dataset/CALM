import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from train.agent import _resolve_policy_kwargs
from utils.portable import export_policy, export_vecnormalize, load_policy, load_vecnormalize

KWARGS = {'net_arch': {'pi': [16, 8], 'vf': [16, 16]}, 'activation_fn': 'ReLU', 'ortho_init': True}


def _trained(tmp_path):
    venv = VecNormalize(DummyVecEnv([lambda: gym.make('CartPole-v1')]))
    model = PPO('MlpPolicy', venv, policy_kwargs=_resolve_policy_kwargs(KWARGS), n_steps=64, batch_size=32,
                seed=1, device='cpu')
    model.learn(256)
    model.save(tmp_path / 'model.zip')
    venv.save(str(tmp_path / 'vecnormalize.pkl'))
    return model, venv


def test_policy_without_pickle_gives_the_same_actions(tmp_path):
    model, _ = _trained(tmp_path)
    export_policy(tmp_path / 'model.zip', tmp_path / 'policy.safetensors')
    fresh = load_policy(tmp_path / 'policy.safetensors', DummyVecEnv([lambda: gym.make('CartPole-v1')]), KWARGS)
    obs = np.random.default_rng(0).normal(size=(200, 4)).astype(np.float32)
    assert (model.predict(obs, deterministic=True)[0] == fresh.predict(obs, deterministic=True)[0]).all()


def test_json_normalisation_matches_the_pickle(tmp_path):
    _, venv = _trained(tmp_path)
    export_vecnormalize(tmp_path / 'vecnormalize.pkl', tmp_path / 'vecnormalize.json')
    plain = DummyVecEnv([lambda: gym.make('CartPole-v1')])
    from_json = load_vecnormalize(tmp_path / 'vecnormalize.json', plain)
    from_pkl = load_vecnormalize(tmp_path / 'vecnormalize.pkl', DummyVecEnv([lambda: gym.make('CartPole-v1')]))
    obs = np.random.default_rng(1).normal(scale=5.0, size=(50, 4))
    assert np.allclose(from_json.normalize_obs(obs), from_pkl.normalize_obs(obs))
