from __future__ import annotations

import argparse
import io
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize


def export_policy(model_zip: Path, out: Path) -> None:
    with zipfile.ZipFile(model_zip) as z:
        state = torch.load(io.BytesIO(z.read('policy.pth')), map_location='cpu', weights_only=True)
    save_file({k: t.contiguous() for k, t in state.items()}, str(out))


def export_vecnormalize(pkl: Path, out: Path) -> None:
    with open(pkl, 'rb') as f:
        v = pickle.load(f)
    stats = {'obs_mean': v.obs_rms.mean.tolist(), 'obs_var': v.obs_rms.var.tolist(),
             'obs_count': float(v.obs_rms.count), 'clip_obs': v.clip_obs,
             'epsilon': v.epsilon, 'norm_obs': v.norm_obs}
    out.write_text(json.dumps(stats, indent=2))


def load_policy(path: str | Path, env: VecEnv, policy_kwargs: dict) -> PPO:
    from train.agent import _resolve_policy_kwargs

    model = PPO('MlpPolicy', env, policy_kwargs=_resolve_policy_kwargs(policy_kwargs), device='cpu')
    model.policy.load_state_dict(load_file(str(path)))
    return model


def load_vecnormalize(path: str | Path, venv: VecEnv) -> VecNormalize:
    p = Path(path)
    if p.suffix != '.json':
        return VecNormalize.load(str(p), venv)
    s = json.loads(p.read_text())
    vn = VecNormalize(venv, norm_obs=s['norm_obs'], norm_reward=False,
                      clip_obs=s['clip_obs'], epsilon=s['epsilon'])
    vn.obs_rms.mean = np.array(s['obs_mean'], dtype=np.float64)
    vn.obs_rms.var = np.array(s['obs_var'], dtype=np.float64)
    vn.obs_rms.count = s['obs_count']
    return vn


def main() -> None:
    ap = argparse.ArgumentParser(description='write policy.safetensors and vecnormalize.json next to a trained model')
    ap.add_argument('model_zip', type=Path)
    ap.add_argument('vecnormalize_pkl', type=Path)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.model_zip.parent
    out.mkdir(parents=True, exist_ok=True)
    export_policy(args.model_zip, out / 'policy.safetensors')
    export_vecnormalize(args.vecnormalize_pkl, out / 'vecnormalize.json')


if __name__ == '__main__':
    main()
