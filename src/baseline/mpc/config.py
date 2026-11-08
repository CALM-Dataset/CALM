from __future__ import annotations

import dataclasses

import yaml

from utils.paths import CONFIG

KMH_FIELDS = ('v_limit', 'v_min', 'v_hard_min', 'v_init_low')
OCP_KEYS = ('Q_speed', 'R_accel', 'S_jerk', 'Q_terminal')


def load_mpc_cfg() -> dict:
    with open(CONFIG / 'baseline' / 'mpc.yaml') as fh:
        return yaml.safe_load(fh)


def horizon() -> int:
    return int(load_mpc_cfg()['N'])


def ramp_decel() -> float:
    return float(load_mpc_cfg()['frontier']['ramp_decel'])


def ocp_weights() -> dict:
    weights = load_mpc_cfg()['ocp']
    missing = [k for k in OCP_KEYS if k not in weights]
    if missing:
        raise KeyError(f'config/baseline/mpc.yaml ocp block is missing {missing}')
    return {k: float(weights[k]) for k in OCP_KEYS}


def reward_config(load_reward_config):
    mcfg = load_mpc_cfg()
    base = load_reward_config()
    names = {f.name for f in dataclasses.fields(base)}
    over = {k: float(v) for k, v in (mcfg.get('weights') or {}).items() if k in names}
    for k, v in (mcfg.get('velocity') or {}).items():
        if k in names:
            over[k] = float(v) / 3.6 if k in KMH_FIELDS else float(v)
    return dataclasses.replace(base, **over)
