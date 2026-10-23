from gym_env.common.config_manager import _load_yaml

_cfg = _load_yaml("env.yaml")

DT = float(_cfg["DT"])
DT_SIM = float(_cfg["DT_SIM"])
N_SUB = int(DT / DT_SIM)
EPISODE_STEPS = int(_cfg["EPISODE_STEPS"])

TRUNC_SUSP_TRAVEL = float(_cfg["TRUNC_SUSP_TRAVEL"])
TRUNC_BODY_DISP   = float(_cfg["TRUNC_BODY_DISP"])
TRUNC_DAMPER_VEL  = float(_cfg["TRUNC_DAMPER_VEL"])
POST_BUMP_MARGIN  = float(_cfg["POST_BUMP_MARGIN"])
POST_BUMP_MARGIN_MATCH_FLAT_START = bool(_cfg.get("POST_BUMP_MARGIN_MATCH_FLAT_START", False))

ACTION_MODE = str(_cfg.get("action_mode", "acceleration"))
SPEED_GAIN = float(_cfg.get("speed_gain", 2.0))
SPEED_FLOOR_RATIO = float(_cfg.get("speed_floor_ratio", 0.0))
DEPTH_OFFSET = float(_cfg.get("depth_offset", 2.0))
DEPTH_SLOPE = float(_cfg.get("depth_slope", 2.0))
WINDOW_LEAD_FAR = float(_cfg.get("window_lead_far", 25.0))
WINDOW_LEAD_NEAR = float(_cfg.get("window_lead_near", 8.0))
WINDOW_AFTER = float(_cfg.get("window_after", 1.5))
WINDOW_RELEASE = float(_cfg.get("window_release", 4.0))
ACTION_REPEAT = int(_cfg.get("action_repeat", 1))

SUSPENSION = {key: float(_cfg[key]) for key in (
    "c_T", "k_T", "A", "v_d", "v_z",
    "f1_cmp", "f2_cmp", "f1_rbd", "f2_rbd", "dz_cmp", "dz_rbd", "F_ks_nlin_max",
)}
