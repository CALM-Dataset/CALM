from dataclasses import dataclass
from gym_env.common.config_manager import _load_yaml


@dataclass(frozen=True)
class RewardConfig:
    Q_zBddot:  float = 50.0
    Q_zWddot:  float =  0.5
    Q_pitch:   float = 20.0
    pitch_arm: float =  1.5
    pitch_clip:float = 30.0
    Q_a:       float =  1.0
    Q_v:       float =  1.0
    track_scale: float = 1.0
    Q_step:    float =  0.5
    time_cost: float =  0.0
    wk_heave:  bool  = False
    w_jerk:          float = 1.0
    w_action_smooth: float = 0.0
    g:          float = 9.81
    heave_clip: float = 1.0e9
    wheel_clip: float = 1.0e9
    v_limit:    float = 9.722
    v_min:      float = 1.389
    v_hard_min: float = 2.222
    v_init_low: float = 1.944
    a_max:      float = 5.0
    v_max:      float = 13.889
    a_comfort:  float = 2.0
    terminal_bonus:   float =  100.0
    terminal_penalty: float = -250.0
    terminal_v_min:   float =    0.0
    terminal_v_max:   float =   18.1
    Q_t:              float =    0.0
    a_limit:          float =    5.0


def load_reward_config() -> RewardConfig:
    try:
        cfg = _load_yaml("reward.yaml")
    except FileNotFoundError:
        return RewardConfig()

    w  = cfg.get("weights",       {})
    v  = cfg.get("vertical",      {})
    s  = cfg.get("velocity",      {})
    e  = cfg.get("eval",          {})
    t  = cfg.get("terminal",      {})
    tr = cfg.get("traffic_rules", {})

    return RewardConfig(
        Q_zBddot        = float(w.get("Q_zBddot",        50.0)),
        Q_zWddot        = float(w.get("Q_zWddot",         0.5)),
        Q_pitch         = float(w.get("Q_pitch",         20.0)),
        pitch_arm       = float(w.get("pitch_arm",        1.5)),
        pitch_clip      = float(w.get("pitch_clip",      30.0)),
        Q_a             = float(w.get("Q_a",              1.0)),
        Q_v             = float(w.get("Q_v",              1.0)),
        track_scale     = float(w.get("track_scale",      1.0)),
        Q_step          = float(w.get("Q_step",           0.5)),
        time_cost       = float(w.get("time_cost",        0.0)),
        wk_heave        = bool(w.get("wk_heave",          False)),
        w_jerk          = float(w.get("w_jerk",           1.0)),
        w_action_smooth = float(w.get("w_action_smooth",  0.0)),

        g          = float(v.get("g",          9.81)),
        heave_clip = float(v.get("heave_clip", 1.0e9)),
        wheel_clip = float(v.get("wheel_clip", 1.0e9)),

        v_limit    = float(s.get("v_limit",    35.0)) / 3.6,
        v_min      = float(s.get("v_min",       5.0)) / 3.6,
        v_hard_min = float(s.get("v_hard_min",  8.0)) / 3.6,
        v_init_low = float(s.get("v_init_low",  7.0)) / 3.6,
        a_max      = float(s.get("a_max",       5.0)),
        v_max      = float(tr.get("v_max",     50.0)) / 3.6,

        a_comfort  = float(e.get("a_comfort", 2.0)),

        terminal_bonus   = float(t.get("bonus",    100.0)),
        terminal_penalty = float(t.get("penalty", -250.0)),
        terminal_v_min   = float(t.get("v_min",     0.0)),
        terminal_v_max   = float(t.get("v_max",    18.1)),
        Q_t              = float(t.get("Q_t",        0.0)),
        a_limit          = float(t.get("a_limit",    5.0)),
    )
