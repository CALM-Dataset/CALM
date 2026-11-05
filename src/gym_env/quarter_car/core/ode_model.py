import numpy as np
from numba import njit

from gym_env.common.env_params import DT_SIM, N_SUB

from gym_env.quarter_car.config.env_params import PHYSICS


_I_MB,  _I_MW  = 0, 1
_I_CT,  _I_KT, _I_KS, _I_DZS = 2, 3, 4, 5
_I_D1,  _I_Z1, _I_D2, _I_Z2  = 6, 7, 8, 9
_I_VD,  _I_VZ                 = 10, 11
_I_F1C, _I_F2C, _I_F1R, _I_F2R = 12, 13, 14, 15
_I_DZC, _I_DZR, _I_FMAX       = 16, 17, 18
_P_LEN = 19

_X_TYRE = 0
_X_ZW   = 1
_X_SUSP = 2
_X_ZB   = 3
_X_V    = 4
_X_POS  = 5


def _build_pvec(d: dict) -> np.ndarray:
    v = np.empty(_P_LEN, dtype=np.float64)
    v[_I_MB]  = d['m_B'];      v[_I_MW]  = d['m_W']
    v[_I_CT]  = d['c_T'];      v[_I_KT]  = d['k_T']
    v[_I_KS]  = d['k_S'];      v[_I_DZS] = d['dz_S_stat']
    v[_I_D1]  = d['d1'];       v[_I_Z1]  = d['z1']
    v[_I_D2]  = d['d2'];       v[_I_Z2]  = d['z2']
    v[_I_VD]  = d['v_d'];      v[_I_VZ]  = d['v_z']
    v[_I_F1C] = d['f1_cmp'];   v[_I_F2C] = d['f2_cmp']
    v[_I_F1R] = d['f1_rbd'];   v[_I_F2R] = d['f2_rbd']
    v[_I_DZC] = d['dz_cmp'];   v[_I_DZR] = d['dz_rbd']
    v[_I_FMAX]= d['F_ks_nlin_max']
    return v


@njit(cache=True)
def _spring_nonlin(dyn: float, p: np.ndarray) -> float:
    dz_cmp = p[_I_DZC];   dz_rbd = p[_I_DZR]
    k_S    = p[_I_KS];    dz_s   = p[_I_DZS]
    F_max  = p[_I_FMAX]

    if dyn > dz_cmp:
        dz_F    = dyn - dz_cmp
        exp_arg = dz_F * p[_I_F2C] / dz_s
        F = k_S * (dz_s * p[_I_F1C] * (np.exp(exp_arg) - 1.0) - dz_F)
    elif dyn < -dz_rbd:
        dz_F    = -dyn - dz_rbd
        exp_arg = dz_F * p[_I_F2R] / dz_s
        F = -k_S * (dz_s * p[_I_F1R] * (np.exp(exp_arg) - 1.0) - dz_F)
    else:
        return 0.0

    if F >  F_max: return  F_max
    if F < -F_max: return -F_max
    return F


@njit(cache=True)
def _damper(v_S: float, p: np.ndarray) -> float:
    d1 = p[_I_D1];  z1 = p[_I_Z1]
    d2 = p[_I_D2];  z2 = p[_I_Z2]
    v_d = p[_I_VD]; v_z = p[_I_VZ]
    k   = 50.0

    w_cmp  = 1.0 / (1.0 + np.exp(-k * v_S))
    F_base = (w_cmp * d1 + (1.0 - w_cmp) * z1) * v_S

    w_hc = 1.0 / (1.0 + np.exp(-k * (v_S - v_d)))
    F_hc = w_hc * (d2 - d1) * (v_S - v_d)

    w_hr = 1.0 / (1.0 + np.exp(-k * (-v_S - v_z)))
    F_hr = w_hr * (z2 - z1) * (-v_S - v_z)

    return F_base + F_hc - F_hr


@njit(cache=True)
def _ode(x: np.ndarray, z_q: float, p: np.ndarray) -> np.ndarray:
    F_spring = p[_I_KS] * x[_X_SUSP] + _spring_nonlin(x[_X_SUSP], p)
    F_damp   = _damper(x[_X_ZW] - x[_X_ZB], p)

    F_tire_k = p[_I_KT] * x[_X_TYRE]
    F_tire_c = p[_I_CT] * (z_q - x[_X_ZW])

    dx            = np.empty(6)
    dx[_X_TYRE]   = z_q - x[_X_ZW]
    dx[_X_ZW]     = (-F_spring - F_damp + F_tire_k + F_tire_c) / p[_I_MW]
    dx[_X_SUSP]   = x[_X_ZW] - x[_X_ZB]
    dx[_X_ZB]     = ( F_spring + F_damp) / p[_I_MB]
    dx[_X_V]      = 0.0
    dx[_X_POS]    = x[_X_ZB]
    return dx


@njit(cache=True)
def _rk4_loop(x: np.ndarray, zq_pre: np.ndarray, dt: float, p: np.ndarray) -> np.ndarray:
    xi = x.copy()
    n  = zq_pre.shape[0]
    for i in range(n):
        zq0 = zq_pre[i, 0]
        zqh = zq_pre[i, 1]
        zqe = zq_pre[i, 2]
        k1 = _ode(xi,               zq0, p)
        k2 = _ode(xi + 0.5*dt*k1,  zqh, p)
        k3 = _ode(xi + 0.5*dt*k2,  zqh, p)
        k4 = _ode(xi +     dt*k3,  zqe, p)
        xi = xi + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return xi


class QuarterCarODE:

    def __init__(self, params: dict = None):
        d = {**PHYSICS, **(params or {})}
        self._pvec = _build_pvec(d)

    def step(
        self,
        x: np.ndarray,
        road,
        s_pos: float,
        v: float,
        a: float = 0.0,
    ) -> tuple[np.ndarray, float, float, float, float]:
        dt = DT_SIM
        p  = self._pvec

        zq_pre = np.empty((N_SUB, 3), dtype=np.float64)
        for i in range(N_SUB):
            s0 = s_pos + i       * dt * v
            sh = s_pos + (i + 0.5) * dt * v
            se = s_pos + (i + 1.0) * dt * v
            zq_pre[i, 0] = road.get_height_dot_at(s0, v)
            zq_pre[i, 1] = road.get_height_dot_at(sh, v)
            zq_pre[i, 2] = road.get_height_dot_at(se, v)

        xi = _rk4_loop(x, zq_pre, dt, p)

        s_end    = s_pos + N_SUB * dt * v
        zq_end   = road.get_height_dot_at(s_end, v)
        dx_end   = _ode(xi, zq_end, p)
        z_B_ddot = float(dx_end[3])
        z_W_ddot = float(dx_end[1])

        return xi, z_B_ddot, 0.0, z_W_ddot, z_W_ddot

    v_index = _X_V
    wheelbase = 0.0

    def observe(self, x, z_W_ddot, z_B_ddot, phi_ddot) -> list:
        return [float(x[_X_ZW] - x[_X_ZB]), z_W_ddot, z_B_ddot]

    def deflections(self, x, road, s_pos) -> tuple:
        return float(x[_X_SUSP]), float(x[_X_TYRE])

    def crash_signals(self, x) -> tuple:
        return abs(float(x[_X_SUSP])), abs(float(x[_X_POS])), abs(float(x[_X_ZW] - x[_X_ZB]))

    def reset(self, v0: float = 6.0) -> np.ndarray:

        x    = np.zeros(6, dtype=np.float64)
        x[4] = v0
        return x
