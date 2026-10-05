import numpy as np

from gym_env.half_car.config.config_manager import _load_yaml

_cfg = _load_yaml("env.yaml")

m_B = float(_cfg["m_B"])
m_W = float(_cfg["m_W"])

c_T = float(_cfg["c_T"])
k_T = float(_cfg["k_T"])
k_S = float(_cfg["k_S"])

dz_S_stat = m_B * 9.81 / k_S

_D = float(_cfg["D"])
_A = float(_cfg["A"])

z1 = 2.0 * _D / (1.0 + _A)
d1 = _A * z1

d2 = d1 / 0.25
z2 = z1 / 0.40

v_d = float(_cfg["v_d"])
v_z = float(_cfg["v_z"])

f1_cmp = float(_cfg["f1_cmp"])
f2_cmp = float(_cfg["f2_cmp"])
f1_rbd = float(_cfg["f1_rbd"])
f2_rbd = float(_cfg["f2_rbd"])

dz_cmp = float(_cfg["dz_cmp"])
dz_rbd = float(_cfg["dz_rbd"])

F_ks_nlin_max = float(_cfg["F_ks_nlin_max"])

PHYSICS = {
    "m_B": m_B,
    "m_W": m_W,
    "c_T": c_T,
    "k_T": k_T,
    "k_S": k_S,
    "dz_S_stat": dz_S_stat,
    "d1": d1,
    "z1": z1,
    "d2": d2,
    "z2": z2,
    "v_d": v_d,
    "v_z": v_z,
    "f1_cmp": f1_cmp,
    "f2_cmp": f2_cmp,
    "f1_rbd": f1_rbd,
    "f2_rbd": f2_rbd,
    "dz_cmp": dz_cmp,
    "dz_rbd": dz_rbd,
    "F_ks_nlin_max": F_ks_nlin_max,
}

_MB_H, _IB_H = 803.0, 1430.0
_LF, _LR = 1.206, 1.664
_L = _LF + _LR
_HT, _HB = 0.600, 0.222
BODY = {"m_B": _MB_H, "I_B": _IB_H, "l_f": _LF, "l_r": _LR, "h_T": _HT, "h_B": _HB}

_SHARED = {
    "c_T": c_T, "k_T": k_T, "A": _A,
    "v_d": v_d, "v_z": v_z,
    "f1_cmp": f1_cmp, "f2_cmp": f2_cmp, "f1_rbd": f1_rbd, "f2_rbd": f2_rbd,
    "dz_cmp": dz_cmp, "dz_rbd": dz_rbd, "F_ks_nlin_max": F_ks_nlin_max,
}


def _axle(k_S_i: float, D_i: float, load_lever: float) -> dict:
    static = _MB_H * 9.81 * (load_lever / _L)
    return {**_SHARED, "m_W": 50.4, "k_S": k_S_i, "D": D_i,
            "dz_S_stat": static / k_S_i}


PHYSICS_F = _axle(2.7915e4, 3530.0, _LR)
PHYSICS_R = _axle(2.0231e4, 3515.0, _LF)

_KS = float(_cfg.get("plant_k_S_scale", 1.0))
_DD = float(_cfg.get("plant_D_scale", 1.0))
PLANT_PARAMS = {
    "front": {"k_S": PHYSICS_F["k_S"] * _KS, "D": PHYSICS_F["D"] * _DD},
    "rear":  {"k_S": PHYSICS_R["k_S"] * _KS, "D": PHYSICS_R["D"] * _DD},
}

DT = float(_cfg["DT"])
DT_SIM = float(_cfg["DT_SIM"])
N_SUB = int(DT / DT_SIM)
EPISODE_STEPS = int(_cfg["EPISODE_STEPS"])

TRUNC_SUSP_TRAVEL = float(_cfg["TRUNC_SUSP_TRAVEL"])
TRUNC_BODY_DISP   = float(_cfg["TRUNC_BODY_DISP"])
TRUNC_DAMPER_VEL  = float(_cfg["TRUNC_DAMPER_VEL"])
POST_BUMP_MARGIN  = float(_cfg["POST_BUMP_MARGIN"])
POST_BUMP_MARGIN_MATCH_FLAT_START = bool(_cfg.get("POST_BUMP_MARGIN_MATCH_FLAT_START", False))

OBS_HIGH = np.array([ 10.0,  300.0,  20.0,  50.0,  5.0,  20.0,  5.0,  20.0], dtype=np.float32)
OBS_LOW  = np.array([-10.0, -300.0, -20.0, -50.0, -5.0,   0.0, -5.0, -20.0], dtype=np.float32)

