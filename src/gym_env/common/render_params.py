from gym_env.common.config_manager import _load_yaml


_cfg = _load_yaml("render.yaml")

RENDER_Y_SCALE = int(_cfg["RENDER_Y_SCALE"])
RENDER_SHOW_TS = bool(_cfg.get("RENDER_SHOW_TS", False))
RENDER_TS_Z = bool(_cfg.get("RENDER_TS_Z", True))
RENDER_TS_Z_DDOT = bool(_cfg.get("RENDER_TS_Z_DDOT", True))
RENDER_TS_SPEED = bool(_cfg.get("RENDER_TS_SPEED", True))

RENDER_FREEZE_EPISODE = bool(_cfg.get("RENDER_FREEZE_EPISODE", False))
RENDER_BUMP_MARKERS  = bool(_cfg.get("RENDER_BUMP_MARKERS",  True))

RENDER_MUSIC         = bool(_cfg.get("RENDER_MUSIC", False))
RENDER_MUSIC_MUTED   = bool(_cfg.get("RENDER_MUSIC_MUTED", True))
RENDER_MUSIC_TRACK   = str(_cfg.get("RENDER_MUSIC_TRACK", "cruise"))
RENDER_MUSIC_VOLUME  = float(_cfg.get("RENDER_MUSIC_VOLUME", 0.6))
RENDER_MUSIC_JUKEBOX = bool(_cfg.get("RENDER_MUSIC_JUKEBOX", False))
