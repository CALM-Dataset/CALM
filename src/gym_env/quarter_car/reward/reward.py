import numpy as np

from gym_env.quarter_car.config.reward_params import RewardConfig


def j_heave(z_B_ddot: float, g: float, clip: float) -> float:
    return -(np.clip(z_B_ddot, -clip, clip) / g) ** 2


def j_wheel(z_W_ddot: float, g: float, clip: float) -> float:
    return -(np.clip(z_W_ddot, -clip, clip) / g) ** 2


def j_long_acc(a: float, g: float) -> float:
    return -(a / g) ** 2


def j_long_jerk(a_actual: float, a_prev: float, g: float) -> float:
    return -((a_actual - a_prev) / g) ** 2


def j_long_vel(v: float, v_ref: float, deadband: float = 0.0, clip: float = 1e9,
               over_clip: float = 1e9) -> float:
    signed = (v - v_ref) / max(v_ref, 1e-8)
    rel = max(0.0, abs(signed) - deadband)
    c = over_clip if signed > 0.0 else clip
    return -min(rel, c)


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
) -> tuple[float, dict]:
    _v_ref = max(v_ref, 1e-8)

    Jh = j_heave(z_B_ddot, cfg.g, cfg.heave_clip)
    Jw = j_wheel(z_W_ddot, cfg.g, cfg.wheel_clip)
    Ja = j_long_acc(a_actual, cfg.g)
    Jj = j_long_jerk(a_actual, a_prev, cfg.g)
    Jv = j_long_vel(v, _v_ref, cfg.v_deadband, cfg.v_err_clip, cfg.v_err_over_clip)
    Qv_eff = cfg.Q_v_over if v > _v_ref else cfg.Q_v
    Jf = -max(0.0, (cfg.v_floor - v) / max(cfg.v_floor, 1e-8)) if v < cfg.v_floor else 0.0
    Jas = -((a_actual - a_prev) / max(cfg.a_max, 1e-6)) ** 2

    total = (
        cfg.Q_zBddot * Jh
        + cfg.Q_zWddot * Jw
        + cfg.Q_a      * Ja
        + cfg.w_jerk   * Jj
        + cfg.w_action_smooth * Jas
        + Qv_eff       * Jv
        + cfg.Q_floor  * Jf
        + cfg.Q_step
    )
    total = float(np.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0))

    bd = {
        "J_heave":     float(cfg.Q_zBddot * Jh),
        "J_wheel":     float(cfg.Q_zWddot * Jw),
        "J_long_acc":  float(cfg.Q_a      * Ja),
        "J_long_jerk": float(cfg.w_jerk   * Jj),
        "J_long_vel":  float(Qv_eff       * Jv),
        "J_act_smooth": float(cfg.w_action_smooth * Jas),
        "J_floor":     float(cfg.Q_floor  * Jf),
        "J_total":     total,
    }
    return total, bd
