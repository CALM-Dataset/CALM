from gym_env.common.reward_params import RewardConfig

_ACC_WORST_G    = 1.0
_VEL_WORST      = 1.0
_JERK_WORST_G   = 0.1
_ACT_SMOOTH_WORST = 4.0
_HEAVE_WORST    = 20.0
_WHEEL_WORST    = 300.0
_PITCH_WORST    = 50.0


def reward_bounds(cfg: RewardConfig, n_steps: int, n_bumps: int = 0) -> dict:
    per_step_max = cfg.Q_step

    heave_worst_g = min(cfg.heave_clip, _HEAVE_WORST) / cfg.g
    wheel_worst_g = min(cfg.wheel_clip, _WHEEL_WORST) / cfg.g
    pitch_worst_g = cfg.pitch_arm * min(cfg.pitch_clip, _PITCH_WORST) / cfg.g

    per_step_min = (
        cfg.Q_zBddot * -(heave_worst_g ** 2)
        + cfg.Q_zWddot * -(wheel_worst_g ** 2)
        + cfg.Q_pitch  * -(pitch_worst_g ** 2)
        + cfg.Q_a      * -(_ACC_WORST_G   ** 2)
        + cfg.w_jerk   * -(_JERK_WORST_G  ** 2)
        + cfg.Q_v      * cfg.track_scale * -_VEL_WORST
        + cfg.w_action_smooth * -_ACT_SMOOTH_WORST
        + cfg.Q_step
    )

    j_time_max = cfg.Q_t * cfg.terminal_bonus

    return {
        "per_step_max": round(per_step_max, 4),
        "per_step_min": round(per_step_min, 4),
        "episode_max":  round(per_step_max * n_steps + cfg.terminal_bonus + j_time_max, 1),
        "episode_min":  round(per_step_min * n_steps + cfg.terminal_penalty, 1),
        "n_steps":  n_steps,
        "n_bumps":  n_bumps,
    }
