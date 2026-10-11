import numpy as np

from gym_env.common.config_manager import _load_yaml
from gym_env.common.env_params import SUSPENSION

_cfg = _load_yaml("env.yaml")

_m_B = float(_cfg["m_B"])
_k_S = float(_cfg["k_S"])
_D = float(_cfg["D"])

_z1 = 2.0 * _D / (1.0 + SUSPENSION["A"])
_d1 = SUSPENSION["A"] * _z1

PHYSICS = {
    **{key: value for key, value in SUSPENSION.items() if key != "A"},
    "m_B": _m_B,
    "m_W": float(_cfg["m_W"]),
    "k_S": _k_S,
    "dz_S_stat": _m_B * 9.81 / _k_S,
    "d1": _d1,
    "z1": _z1,
    "d2": _d1 / 0.25,
    "z2": _z1 / 0.40,
}

OBS_HIGH = np.array(_cfg["OBS_HIGH"], dtype=np.float32)
OBS_LOW  = np.array(_cfg["OBS_LOW"],  dtype=np.float32)
