import numpy as np
from numba import njit

from gym_env.common.env_params import DT_SIM, N_SUB

from gym_env.half_car.config.env_params import PHYSICS_F, PHYSICS_R, BODY


_I_MW  = 0
_I_CT, _I_KT, _I_KS, _I_DZS = 1, 2, 3, 4
_I_D1, _I_Z1, _I_D2, _I_Z2  = 5, 6, 7, 8
_I_VD, _I_VZ                = 9, 10
_I_F1C, _I_F2C, _I_F1R, _I_F2R = 11, 12, 13, 14
_I_DZC, _I_DZR, _I_FMAX     = 15, 16, 17
_P_LEN = 18

_X_TYRE_R = 0
_X_ZW_R   = 1
_X_TYRE_F = 2
_X_ZW_F   = 3
_X_SUSP_R = 4
_X_ZB     = 5
_X_SUSP_F = 6
_X_PHI    = 7
_X_V      = 8
_X_POS    = 9
_X_PITCH  = 10
_X_LEN = 11


def _axle_pvec(d: dict) -> np.ndarray:
    z1 = 2.0 * d["D"] / (1.0 + d["A"])
    d1 = d["A"] * z1
    v = np.empty(_P_LEN, dtype=np.float64)
    v[_I_MW]  = d["m_W"]
    v[_I_CT]  = d["c_T"];  v[_I_KT] = d["k_T"];  v[_I_KS] = d["k_S"]
    v[_I_DZS] = d["dz_S_stat"]
    v[_I_D1]  = d1;        v[_I_Z1] = z1
    v[_I_D2]  = d1 / 0.25; v[_I_Z2] = z1 / 0.40
    v[_I_VD]  = d["v_d"];  v[_I_VZ] = d["v_z"]
    v[_I_F1C] = d["f1_cmp"]; v[_I_F2C] = d["f2_cmp"]
    v[_I_F1R] = d["f1_rbd"]; v[_I_F2R] = d["f2_rbd"]
    v[_I_DZC] = d["dz_cmp"]; v[_I_DZR] = d["dz_rbd"]
    v[_I_FMAX] = d["F_ks_nlin_max"]
    return v


@njit(cache=True)
def _spring_nonlin(dyn: float, p: np.ndarray) -> float:
    dz_cmp = p[_I_DZC]; dz_rbd = p[_I_DZR]
    k_S = p[_I_KS];     dz_s = p[_I_DZS];   F_max = p[_I_FMAX]
    if dyn > dz_cmp:
        dz_F = dyn - dz_cmp
        F = k_S * (dz_s * p[_I_F1C] * (np.exp(dz_F * p[_I_F2C] / dz_s) - 1.0) - dz_F)
    elif dyn < -dz_rbd:
        dz_F = -dyn - dz_rbd
        F = -k_S * (dz_s * p[_I_F1R] * (np.exp(dz_F * p[_I_F2R] / dz_s) - 1.0) - dz_F)
    else:
        return 0.0
    if F > F_max:
        return F_max
    if F < -F_max:
        return -F_max
    return F


@njit(cache=True)
def _damper(v_S: float, p: np.ndarray) -> float:
    d1 = p[_I_D1]; z1 = p[_I_Z1]; d2 = p[_I_D2]; z2 = p[_I_Z2]
    v_d = p[_I_VD]; v_z = p[_I_VZ]; k = 50.0
    w_cmp = 1.0 / (1.0 + np.exp(-k * v_S))
    F_base = (w_cmp * d1 + (1.0 - w_cmp) * z1) * v_S
    w_hc = 1.0 / (1.0 + np.exp(-k * (v_S - v_d)))
    F_hc = w_hc * (d2 - d1) * (v_S - v_d)
    w_hr = 1.0 / (1.0 + np.exp(-k * (-v_S - v_z)))
    F_hr = w_hr * (z2 - z1) * (-v_S - v_z)
    return F_base + F_hc - F_hr


@njit(cache=True)
def _axle_force(susp, susp_vel, tyre, zr_dot, zw_vel, p):
    F_susp = p[_I_KS] * susp + _spring_nonlin(susp, p) + _damper(susp_vel, p)
    F_tire = p[_I_KT] * tyre + p[_I_CT] * (zr_dot - zw_vel)
    return F_susp, F_tire


@njit(cache=True)
def _ode(x, zf_dot, zr_dot, pf, pr, m_B, I_B, l_f, l_r, a, hT, hB, m_tot):
    corner_vel_f = x[_X_ZB] - l_f * x[_X_PHI]
    corner_vel_r = x[_X_ZB] + l_r * x[_X_PHI]

    Fs_f, Ft_f = _axle_force(x[_X_SUSP_F], x[_X_ZW_F] - corner_vel_f,
                             x[_X_TYRE_F], zf_dot, x[_X_ZW_F], pf)
    Fs_r, Ft_r = _axle_force(x[_X_SUSP_R], x[_X_ZW_R] - corner_vel_r,
                             x[_X_TYRE_R], zr_dot, x[_X_ZW_R], pr)

    dx = np.empty(_X_LEN)
    dx[_X_TYRE_R] = zr_dot - x[_X_ZW_R]
    dx[_X_ZW_R]   = (-Fs_r + Ft_r) / pr[_I_MW]
    dx[_X_TYRE_F] = zf_dot - x[_X_ZW_F]
    dx[_X_ZW_F]   = (-Fs_f + Ft_f) / pf[_I_MW]
    dx[_X_SUSP_R] = x[_X_ZW_R] - x[_X_ZB] - l_r * x[_X_PHI]
    dx[_X_ZB]     = (Fs_f + Fs_r) / m_B
    dx[_X_SUSP_F] = x[_X_ZW_F] - x[_X_ZB] + l_f * x[_X_PHI]
    p_h = hT if a > 0.0 else hB
    dx[_X_PHI]    = (l_r * Fs_r - l_f * Fs_f - p_h * m_tot * a) / I_B
    dx[_X_V]      = 0.0
    dx[_X_POS]    = x[_X_ZB]
    dx[_X_PITCH]  = x[_X_PHI]
    return dx


@njit(cache=True)
def _rk4_loop(x, zf_pre, zr_pre, dt, pf, pr, m_B, I_B, l_f, l_r, a, hT, hB, m_tot):
    xi = x.copy()
    for i in range(zf_pre.shape[0]):
        f0, fh, fe = zf_pre[i, 0], zf_pre[i, 1], zf_pre[i, 2]
        r0, rh, re = zr_pre[i, 0], zr_pre[i, 1], zr_pre[i, 2]
        k1 = _ode(xi,              f0, r0, pf, pr, m_B, I_B, l_f, l_r, a, hT, hB, m_tot)
        k2 = _ode(xi + 0.5*dt*k1,  fh, rh, pf, pr, m_B, I_B, l_f, l_r, a, hT, hB, m_tot)
        k3 = _ode(xi + 0.5*dt*k2,  fh, rh, pf, pr, m_B, I_B, l_f, l_r, a, hT, hB, m_tot)
        k4 = _ode(xi +     dt*k3,  fe, re, pf, pr, m_B, I_B, l_f, l_r, a, hT, hB, m_tot)
        xi = xi + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    return xi


class HalfCarODE:

    def __init__(self, params: dict = None):
        self._pf = _axle_pvec({**PHYSICS_F, **(params or {}).get("front", {})})
        self._pr = _axle_pvec({**PHYSICS_R, **(params or {}).get("rear", {})})
        b = {**BODY, **(params or {}).get("body", {})}
        self._mB, self._IB = float(b["m_B"]), float(b["I_B"])
        self._lf, self._lr = float(b["l_f"]), float(b["l_r"])
        self._L = self._lf + self._lr
        self._hT, self._hB = float(b.get("h_T", 0.0)), float(b.get("h_B", 0.0))
        self._mtot = self._mB + float(self._pf[_I_MW]) + float(self._pr[_I_MW])

    v_index = _X_V

    @property
    def wheelbase(self) -> float:
        return self._L

    def observe(self, x, z_W_ddot, z_B_ddot, phi_ddot) -> list:
        return [float(x[_X_ZW_F] - x[_X_ZB]), z_W_ddot, z_B_ddot, phi_ddot, float(x[_X_PHI])]

    def deflections(self, x, road, s_pos) -> tuple:
        return float(x[_X_SUSP_F]), float(x[_X_TYRE_F])

    def crash_signals(self, x) -> tuple:
        susp_travel = max(abs(float(x[_X_SUSP_F])), abs(float(x[_X_SUSP_R])))
        damper_f = float(x[_X_ZW_F]) - (float(x[_X_ZB]) - self._lf * float(x[_X_PHI]))
        damper_r = float(x[_X_ZW_R]) - (float(x[_X_ZB]) + self._lr * float(x[_X_PHI]))
        return susp_travel, abs(float(x[_X_POS])), max(abs(damper_f), abs(damper_r))

    def perturb(self, x, rng) -> None:
        x[0:4] += rng.normal(0.0, 0.005, size=4)
        x[_X_POS] += rng.normal(0.0, 0.001)

    def step(self, x, road, s_pos, v, a=0.0):
        dt = DT_SIM
        L = self._L
        zf = np.empty((N_SUB, 3))
        zr = np.empty((N_SUB, 3))
        for i in range(N_SUB):
            s0 = s_pos + i * dt * v
            sh = s_pos + (i + 0.5) * dt * v
            se = s_pos + (i + 1.0) * dt * v
            zf[i, 0] = road.get_height_dot_at(s0, v)
            zf[i, 1] = road.get_height_dot_at(sh, v)
            zf[i, 2] = road.get_height_dot_at(se, v)
            zr[i, 0] = road.get_height_dot_at(s0 - L, v)
            zr[i, 1] = road.get_height_dot_at(sh - L, v)
            zr[i, 2] = road.get_height_dot_at(se - L, v)

        xi = _rk4_loop(x, zf, zr, dt, self._pf, self._pr,
                       self._mB, self._IB, self._lf, self._lr,
                       a, self._hT, self._hB, self._mtot)

        s_end = s_pos + N_SUB * dt * v
        zf_e = road.get_height_dot_at(s_end, v)
        zr_e = road.get_height_dot_at(s_end - L, v)
        dx = _ode(xi, zf_e, zr_e, self._pf, self._pr,
                  self._mB, self._IB, self._lf, self._lr,
                  a, self._hT, self._hB, self._mtot)
        z_B_ddot = float(dx[_X_ZB])
        phi_ddot = float(dx[_X_PHI])
        z_Wf_ddot = float(dx[_X_ZW_F])
        z_Wr_ddot = float(dx[_X_ZW_R])
        return xi, z_B_ddot, phi_ddot, z_Wf_ddot, z_Wr_ddot

    def reset(self, v0: float = 6.0) -> np.ndarray:
        x = np.zeros(_X_LEN, dtype=np.float64)
        x[_X_V] = v0
        return x
