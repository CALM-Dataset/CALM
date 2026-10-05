from __future__ import annotations

import numpy as np


class PerceivedRoad:
    def __init__(self, true_road, track, on_bump_m: float,
                 nominal_h: float, nominal_w: float):
        self._road      = true_road
        self._track     = track
        self._on_bump_m = float(on_bump_m)
        self._nom_h     = float(nominal_h)
        self._nom_w     = float(nominal_w)
        self._s_pos     = 0.0
        self._true_bumps = list(getattr(true_road, "_bumps", []) or [])

    def set_s_pos(self, s_pos: float) -> None:
        self._s_pos = float(s_pos)

    def _perceived_dist(self, dist: float):
        if self._track is None:
            return dist
        samples = self._track["samples"]
        d = np.array([s["d_true"] for s in samples])
        s = samples[int(np.argmin(np.abs(d - dist)))]
        if not s["detected"]:
            return None
        return s["d_hat"] if s["d_hat"] is not None else s["d_true"]

    @property
    def _bumps(self):
        out = []
        for x0, A, L in self._true_bumps:
            dist = x0 - self._s_pos
            if dist <= self._on_bump_m:
                out.append((x0, A, L))
                continue
            d_hat = self._perceived_dist(dist)
            if d_hat is None:
                continue
            x0_hat = self._s_pos + float(d_hat)
            h = self._nom_h if A >= 0 else -self._nom_h
            out.append((x0_hat, h, self._nom_w))
        return out

    def __getattr__(self, name):
        return getattr(self._road, name)


def make_perceived_road(true_road, track, cfg,
                        nominal_h: float = 0.05, nominal_w: float = 1.0):
    on_bump_m = float(getattr(cfg, "on_bump_m", 0.5))
    nh = float(getattr(cfg, "baseline_nominal_bump_h", nominal_h))
    nw = float(getattr(cfg, "baseline_nominal_bump_w", nominal_w))
    return PerceivedRoad(true_road, track, on_bump_m, nh, nw)
