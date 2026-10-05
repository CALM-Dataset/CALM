from gym_env.quarter_car.config.reward_params import RewardConfig

_ACC_WORST_G    = 1.0
_VEL_WORST      = 1.0
_JERK_WORST_G   = 0.1


def reward_bounds(cfg: RewardConfig, n_steps: int, n_bumps: int = 0) -> dict:
    per_step_max = cfg.Q_step

    heave_worst_g = cfg.heave_clip / cfg.g
    wheel_worst_g = cfg.wheel_clip / cfg.g

    per_step_min = (
        cfg.Q_zBddot * -(heave_worst_g ** 2)
        + cfg.Q_zWddot * -(wheel_worst_g ** 2)
        + cfg.Q_a      * -(_ACC_WORST_G   ** 2)
        + cfg.w_jerk   * -(_JERK_WORST_G  ** 2)
        + cfg.Q_v      * -_VEL_WORST
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
