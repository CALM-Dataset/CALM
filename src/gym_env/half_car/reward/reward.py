from __future__ import annotations

import numpy as np

from gym_env.half_car.config.reward_params import RewardConfig


_track_scale_override: float | None = None


def set_track_scale_override(value: float | None) -> None:
    global _track_scale_override
    _track_scale_override = value


def j_heave(z_B_ddot: float, g: float, clip: float) -> float:
    return -(np.clip(z_B_ddot, -clip, clip) / g) ** 2


def j_wheel(z_W_ddot: float, g: float, clip: float) -> float:
    return -(np.clip(z_W_ddot, -clip, clip) / g) ** 2


def j_pitch(phi_ddot: float, arm: float, g: float, clip: float) -> float:
    return -((arm * np.clip(phi_ddot, -clip, clip)) / g) ** 2


def j_long_acc(a: float, g: float) -> float:
    return -(a / g) ** 2


def j_long_jerk(a_actual: float, a_prev: float, g: float) -> float:
    return -((a_actual - a_prev) / g) ** 2


def j_speed_track(v: float, v_ref: float) -> float:
    return -abs(v - v_ref) / max(v_ref, 1e-8)


def j_destination(success: bool, cfg: RewardConfig) -> float:
    return cfg.terminal_bonus if success else cfg.terminal_penalty


def j_time(t: float, t_max: float, cfg: RewardConfig) -> float:
    frac_used = float(np.clip(t / max(t_max, 1e-6), 0.0, 1.0))
    return cfg.Q_t * cfg.terminal_bonus * (1.0 - frac_used)


def compute_terminal_reward(
    success: bool, t: float, t_max: float,
    cfg: RewardConfig,
) -> float:
    Jd = j_destination(success, cfg)
    Jt = j_time(t, t_max, cfg) if success else 0.0
    return Jd + Jt


def compute_reward(
    v: float,
    z_B_ddot: float,
    z_W_ddot: float,
    a_actual: float,
    a_prev: float,
    cfg: RewardConfig,
    v_ref: float = 0.0,
    phi_ddot: float = 0.0,
) -> tuple[float, dict]:
    v_nom = max(v_ref, 1e-8)

    Jh = j_heave(z_B_ddot, cfg.g, cfg.heave_clip)
    Jw = j_wheel(z_W_ddot, cfg.g, cfg.wheel_clip)
    Jp = j_pitch(phi_ddot, cfg.pitch_arm, cfg.g, cfg.pitch_clip)
    Ja = j_long_acc(a_actual, cfg.g)
    Jj = j_long_jerk(a_actual, a_prev, cfg.g)
    Jv = j_speed_track(v, v_nom)
    Qv_eff = cfg.Q_v * (cfg.track_scale if _track_scale_override is None else _track_scale_override)
    Jas = -((a_actual - a_prev) / max(cfg.a_max, 1e-6)) ** 2

    total = (
        cfg.Q_zBddot * Jh
        + cfg.Q_zWddot * Jw
        + cfg.Q_pitch  * Jp
        + cfg.Q_a      * Ja
        + cfg.w_jerk   * Jj
        + cfg.w_action_smooth * Jas
        + Qv_eff       * Jv
        + cfg.Q_step
    )
    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))

    bd = {
        "J_heave":     float(cfg.Q_zBddot * Jh),
        "J_wheel":     float(cfg.Q_zWddot * Jw),
        "J_pitch":     float(cfg.Q_pitch  * Jp),
        "J_long_acc":  float(cfg.Q_a      * Ja),
        "J_long_jerk": float(cfg.w_jerk   * Jj),
        "J_long_vel":  float(Qv_eff       * Jv),
        "J_act_smooth": float(cfg.w_action_smooth * Jas),
        "J_total":     total,
    }
    return total, bd
