from dataclasses import dataclass
from gym_env.common.config_manager import _load_yaml


@dataclass(frozen=True)
class PreviewConfig:
    preview_distance:    float = 40.0
    h_clip:              float = 0.15
    n_peaks:             int   = 3
    peak_height_min:     float = 0.01
    peak_distance_min_m: float = 0.5

    noise_active:        bool  = False
    noise_height_std:    float = 0.005
    noise_distance_std:  float = 0.5
    noise_width_std:     float = 0.05

    pt1_tau:             float = 0.05

    use_dist_obs:        bool  = True
    source:              str   = "bumps"
    features:            str   = "v2"
    length_clip:         float = 8.0
    freq_clip:           float = 10.0

    dense_n:             int   = 200
    freq_max:            float = 21.74
    v_min_preview:       float = 0.5

    perception_mode:     str   = "none"
    model_path:          str   = "config/perception/perception_model.yaml"
    on_bump_m:           float = 0.5
    sentinel:            float = 0.0
    baseline_nominal_bump_h: float = 0.05
    baseline_nominal_bump_w: float = 1.0


def load_preview_config() -> PreviewConfig:
    try:
        cfg = _load_yaml("preview.yaml")
    except FileNotFoundError:
        return PreviewConfig()

    return PreviewConfig(
        preview_distance    = float(cfg.get("preview_distance",    40.0)),
        h_clip              = float(cfg.get("h_clip",              0.15)),
        n_peaks             = int(  cfg.get("n_peaks",             3)),
        peak_height_min     = float(cfg.get("peak_height_min",     0.01)),
        peak_distance_min_m = float(cfg.get("peak_distance_min_m", 0.5)),
        noise_active        = bool( cfg.get("noise_active",        False)),
        noise_height_std    = float(cfg.get("noise_height_std",    0.005)),
        noise_distance_std  = float(cfg.get("noise_distance_std",  0.5)),
        noise_width_std     = float(cfg.get("noise_width_std",     0.05)),
        pt1_tau             = float(cfg.get("pt1_tau",             0.05)),
        use_dist_obs        = bool( cfg.get("use_dist_obs",        True)),
        source              = str(  cfg.get("source",              "bumps")),
        features            = str(  cfg.get("features",            "v2")),
        length_clip         = float(cfg.get("length_clip",         8.0)),
        freq_clip           = float(cfg.get("freq_clip",           10.0)),
        dense_n             = int(  cfg.get("dense_n",             200)),
        freq_max            = float(cfg.get("freq_max",            21.74)),
        v_min_preview       = float(cfg.get("v_min_preview",       0.5)),
        perception_mode     = str(  cfg.get("perception_mode",     "none")),
        model_path          = str(  cfg.get("model_path",          "config/perception/perception_model.yaml")),
        on_bump_m           = float(cfg.get("on_bump_m",           0.5)),
        sentinel            = float(cfg.get("sentinel",            0.0)),
        baseline_nominal_bump_h = float(cfg.get("baseline_nominal_bump_h", 0.05)),
        baseline_nominal_bump_w = float(cfg.get("baseline_nominal_bump_w", 1.0)),
    )
