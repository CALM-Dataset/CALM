from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.signal import find_peaks

from gym_env.quarter_car.config.preview_params import PreviewConfig, load_preview_config
from gym_env.quarter_car.config.env_params import DT


def load_perception_model(path):
    import yaml
    from pathlib import Path
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[4] / p
    return yaml.safe_load(p.read_text())


def _p_detect(model, d):
    if d < 0 or d >= model["max_range_m"]:
        return 0.0
    return model["p_detect"][int(d // model["bin_width_m"])]


def _range_err(model, d):
    if d < 0 or d >= model["max_range_m"]:
        return 0.0, 0.0
    i = int(d // model["bin_width_m"])
    bias, std = model["range_bias"][i], model["range_std"][i]
    # a bin can be inside max_range_m but still have too few valid ranged detections
    # to fit (see MIN_BIN_SAMPLES in perception-pipeline's model.py) - None there
    # means "no data", not zero error; fall back to no bias/noise for that case
    return (bias if bias is not None else 0.0), (std if std is not None else 0.0)


def _apply_perception_model(peak3, dist_m, model, rng, on_bump_m, sentinel, horizon_m):
    if rng.random() > _p_detect(model, dist_m):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    bias, std = _range_err(model, dist_m)
    d_noisy = max(0.0, dist_m + bias + (rng.normal(0, std) if std > 0 else 0.0))
    peak3[0] = float(np.clip(d_noisy / horizon_m, 0.0, 1.0))
    if dist_m > on_bump_m:
        peak3[1] = sentinel
        peak3[2] = sentinel
    return peak3


def _apply_perception_replay(peak3, dist_m, track, on_bump_m, sentinel, horizon_m):
    samples = track["samples"]
    d = np.array([s["d_true"] for s in samples])
    s = samples[int(np.argmin(np.abs(d - dist_m)))]
    if not s["detected"]:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    seen = s["d_hat"] if s["d_hat"] is not None else s["d_true"]
    peak3[0] = float(np.clip(seen / horizon_m, 0.0, 1.0))
    if dist_m > on_bump_m:
        peak3[1] = sentinel
        peak3[2] = sentinel
    return peak3


class PreviewWrapper(gym.ObservationWrapper):

    def __init__(self, env: gym.Env, cfg: PreviewConfig | None = None):
        super().__init__(env)

        self._cfg = cfg or load_preview_config()
        n_prev    = self._cfg.n_peaks * 3

        orig_high = env.observation_space.high
        orig_low  = env.observation_space.low

        self.observation_space = spaces.Box(
            low  = np.concatenate([orig_low,  np.zeros(n_prev, dtype=np.float32)]),
            high = np.concatenate([orig_high, np.ones(n_prev,  dtype=np.float32)]),
            dtype=np.float32,
        )

        self.preview: np.ndarray = np.tile(
            [1.0, 0.0, 0.0], self._cfg.n_peaks
        ).astype(np.float32)
        self._filtered_preview = self.preview.copy()
        self._track = None
        self._model = None
        if self._cfg.perception_mode == "model":
            try:
                self._model = load_perception_model(self._cfg.model_path)
            except Exception:
                self._model = None

    def set_perception_track(self, track):
        self._track = track

    def observation(self, obs: np.ndarray) -> np.ndarray:
        peak_obs = self._compute_peaks()
        return np.concatenate([obs, peak_obs]).astype(np.float32)

    def reset(self, **kwargs):
        self._filtered_preview = np.tile(
            [1.0, 0.0, 0.0], self._cfg.n_peaks
        ).astype(np.float32)
        self.preview = self._filtered_preview.copy()
        return super().reset(**kwargs)

    def _compute_peaks(self) -> np.ndarray:
        cfg  = self._cfg
        env  = self.env.unwrapped

        v_safe = max(float(env._v), cfg.v_min_preview)
        v_min  = float(getattr(getattr(env, "_rcfg", None), "v_min", cfg.v_min_preview))
        T_MAX  = cfg.preview_distance / max(v_min, cfg.v_min_preview)

        heights = env._road.get_spatial_preview(
            s_pos=env._s_pos,
            v_current=v_safe,
            lookahead_m=cfg.preview_distance,
            n_points=cfg.dense_n,
        )

        ds               = cfg.preview_distance / cfg.dense_n
        min_dist_samples = max(1, int(cfg.peak_distance_min_m / ds))

        peaks, props = find_peaks(
            heights,
            height=cfg.peak_height_min,
            distance=min_dist_samples,
            width=0,
        )

        peak_arr = np.tile([1.0, 0.0, 0.0], cfg.n_peaks).astype(np.float32)
        peak_dists = [None] * cfg.n_peaks

        for i, pk in enumerate(peaks[: cfg.n_peaks]):
            dist_m      = float(pk * ds)
            peak_dists[i] = dist_m
            peak_h      = float(heights[pk])
            peak_w_m    = float(props["widths"][i] * ds)

            if cfg.use_dist_obs:
                peak_arr[i * 3] = float(np.clip(dist_m / cfg.preview_distance, 0.0, 1.0))
            else:
                t2r = dist_m / v_safe
                peak_arr[i * 3] = float(np.clip(t2r / T_MAX, 0.0, 1.0))

            peak_arr[i * 3 + 1] = float(np.clip(peak_h / cfg.h_clip, 0.0, 1.0))

            if cfg.use_dist_obs:
                peak_arr[i * 3 + 2] = float(np.clip(peak_w_m / cfg.preview_distance, 0.0, 1.0))
            else:
                freq = v_safe / max(peak_w_m, 0.01)
                peak_arr[i * 3 + 2] = float(np.clip(freq / cfg.freq_max, 0.0, 1.0))

        self.preview = peak_arr.copy()

        if self._track is not None and cfg.perception_mode in ("replay", "model"):
            for i, d in enumerate(peak_dists):
                if d is not None:
                    peak_arr[i * 3:i * 3 + 3] = _apply_perception_replay(
                        peak_arr[i * 3:i * 3 + 3], d, self._track,
                        cfg.on_bump_m, cfg.sentinel, cfg.preview_distance)
            peak_arr = np.clip(peak_arr, 0.0, 1.0)

        elif cfg.perception_mode == "model" and self._model is not None \
                and self.np_random is not None:
            for i, d in enumerate(peak_dists):
                if d is not None:
                    peak_arr[i * 3:i * 3 + 3] = _apply_perception_model(
                        peak_arr[i * 3:i * 3 + 3], d, self._model, self.np_random,
                        cfg.on_bump_m, cfg.sentinel, cfg.preview_distance)
            peak_arr = np.clip(peak_arr, 0.0, 1.0)

        elif cfg.noise_active and self.np_random is not None:
            rng = self.np_random
            for i in range(cfg.n_peaks):
                if peak_arr[i * 3] < 1.0:
                    scale = peak_arr[i * 3]
                    peak_arr[i * 3]     += float(rng.normal(0, cfg.noise_distance_std)) * scale
                    peak_arr[i * 3 + 1] += float(rng.normal(0, cfg.noise_height_std))   * scale
                    peak_arr[i * 3 + 2] += float(rng.normal(0, cfg.noise_width_std))    * scale
            peak_arr = np.clip(peak_arr, 0.0, 1.0)

        alpha = DT / (cfg.pt1_tau + DT)
        self._filtered_preview = (
            self._filtered_preview + alpha * (peak_arr - self._filtered_preview)
        )

        return np.clip(self._filtered_preview, 0.0, 1.0)
