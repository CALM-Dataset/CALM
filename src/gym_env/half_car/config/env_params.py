import numpy as np

from gym_env.common.config_manager import _load_yaml
from gym_env.common.env_params import SUSPENSION

_cfg = _load_yaml("env.yaml")

_MB_H, _IB_H = 803.0, 1430.0
_LF, _LR = 1.206, 1.664
_L = _LF + _LR
_HT, _HB = 0.600, 0.222
BODY = {"m_B": _MB_H, "I_B": _IB_H, "l_f": _LF, "l_r": _LR, "h_T": _HT, "h_B": _HB}

def _axle(k_S_i: float, D_i: float, load_lever: float) -> dict:
    static = _MB_H * 9.81 * (load_lever / _L)
    return {**SUSPENSION, "m_W": 50.4, "k_S": k_S_i, "D": D_i,
            "dz_S_stat": static / k_S_i}


PHYSICS_F = _axle(2.7915e4, 3530.0, _LR)
PHYSICS_R = _axle(2.0231e4, 3515.0, _LF)

_KS = float(_cfg.get("plant_k_S_scale", 1.0))
_DD = float(_cfg.get("plant_D_scale", 1.0))
PLANT_PARAMS = {
    "front": {"k_S": PHYSICS_F["k_S"] * _KS, "D": PHYSICS_F["D"] * _DD},
    "rear":  {"k_S": PHYSICS_R["k_S"] * _KS, "D": PHYSICS_R["D"] * _DD},
}

OBS_HIGH = np.array([ 10.0,  300.0,  20.0,  50.0,  5.0,  20.0,  5.0,  20.0], dtype=np.float32)
OBS_LOW  = np.array([-10.0, -300.0, -20.0, -50.0, -5.0,   0.0, -5.0, -20.0], dtype=np.float32)

