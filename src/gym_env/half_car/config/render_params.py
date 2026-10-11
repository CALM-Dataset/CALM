from gym_env.common.config_manager import _load_yaml


_cfg = _load_yaml("render.yaml")




RENDER_Y_W_NOM = float(_cfg["RENDER_Y_W_NOM"])
RENDER_Y_B_NOM = float(_cfg["RENDER_Y_B_NOM"])
RENDER_H_MW = float(_cfg["RENDER_H_MW"])
RENDER_REALISTIC_WHEELS = bool(_cfg.get("RENDER_REALISTIC_WHEELS", True))
RENDER_W_MW = float(_cfg["RENDER_W_MW"])
RENDER_H_MB = float(_cfg["RENDER_H_MB"])
RENDER_W_MB = float(_cfg["RENDER_W_MB"])

RENDER_XLIM = tuple(_cfg["RENDER_XLIM"])
RENDER_YLIM = tuple(_cfg["RENDER_YLIM"])

RENDER_ROAD_HALF = float(_cfg["RENDER_ROAD_HALF"])
RENDER_ROAD_N = int(_cfg["RENDER_ROAD_N"])
RENDER_GROUND_Y = float(_cfg["RENDER_GROUND_Y"])

RENDER_C_MB = str(_cfg["RENDER_C_MB"])
RENDER_C_MW = str(_cfg["RENDER_C_MW"])
RENDER_C_SPRING = str(_cfg["RENDER_C_SPRING"])
RENDER_C_DAMPER = str(_cfg["RENDER_C_DAMPER"])
RENDER_C_ROAD = str(_cfg["RENDER_C_ROAD"])
RENDER_C_GROUND = str(_cfg["RENDER_C_GROUND"])

RENDER_SP_X = float(_cfg["RENDER_SP_X"])
RENDER_SP_W = float(_cfg["RENDER_SP_W"])
RENDER_SP_N = int(_cfg["RENDER_SP_N"])

RENDER_DA_X = float(_cfg["RENDER_DA_X"])
RENDER_DA_W = float(_cfg["RENDER_DA_W"])
RENDER_DA_PIST_H = float(_cfg["RENDER_DA_PIST_H"])
RENDER_DA_PIST_FRAC = float(_cfg["RENDER_DA_PIST_FRAC"])
RENDER_DA_LOWER_STEM = float(_cfg["RENDER_DA_LOWER_STEM"])
RENDER_DA_CYL_H_SUSP = float(_cfg["RENDER_DA_CYL_H_SUSP"])
RENDER_DA_CYL_H_TIRE = float(_cfg["RENDER_DA_CYL_H_TIRE"])

RENDER_CONTACT_STEM = float(_cfg["RENDER_CONTACT_STEM"])
Y_LINE_OFFSET = float(_cfg["Y_LINE_OFFSET"])

RENDER_ARR_X   = float(_cfg.get("RENDER_ARR_X",    -2.5))
RENDER_ARR_Y   = float(_cfg.get("RENDER_ARR_Y",     5.5))
RENDER_ARR_LEN = float(_cfg.get("RENDER_ARR_LEN",   8.0))

from gym_env.half_car.config.env_params import BODY as _BODY

RENDER_LF      = float(_BODY["l_f"])
RENDER_LR      = float(_BODY["l_r"])
RENDER_L       = RENDER_LF + RENDER_LR
RENDER_X_FRONT = 0.0
RENDER_X_REAR  = -RENDER_L

_hc = _cfg.get("half_car", {})

RENDER_W_MW          = float(_hc["W_MW"])
RENDER_W_MB          = float(_hc["W_MB"])
RENDER_MB_OVERHANG   = float(_hc["MB_OVERHANG"])
RENDER_Y_W_NOM       = float(_hc["Y_W_NOM"])
RENDER_Y_B_NOM       = float(_hc["Y_B_NOM"])
RENDER_H_MB          = float(_hc["H_MB"])
RENDER_H_MW          = float(_hc["H_MW"])
Y_LINE_OFFSET        = float(_hc["Y_LINE_OFFSET"])
RENDER_DA_CYL_H_SUSP = float(_hc["DA_CYL_H_SUSP"])
RENDER_DA_CYL_H_TIRE = float(_hc["DA_CYL_H_TIRE"])
RENDER_DA_LOWER_STEM = float(_hc["DA_LOWER_STEM"])
RENDER_DA_PIST_H     = float(_hc["DA_PIST_H"])

RENDER_OBSTACLE_FLAG_M = float(_hc["OBSTACLE_FLAG_M"])
RENDER_MAX_EP_STEPS    = int(_hc["MAX_EP_STEPS"])

RENDER_XLIM = (-(RENDER_L + RENDER_MB_OVERHANG + float(_hc["XLIM_LEFT_PAD"])),
               float(_hc["XLIM_RIGHT"]))
RENDER_YLIM = tuple(_hc["YLIM"])
RENDER_ARR_X = -(RENDER_L + RENDER_MB_OVERHANG + float(_hc["ARR_X_PAD"]))
RENDER_ARR_Y = float(_hc["ARR_Y"])
