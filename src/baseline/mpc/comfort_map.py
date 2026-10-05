from __future__ import annotations

import numpy as np

from pathlib import Path

import yaml

from gym_env.quarter_car.config.env_params import DT
from gym_env.quarter_car.config.reward_params import RewardConfig
from baseline.mpc.vehicle import make_vehicle_spec


def _load_comfort_cfg() -> dict:
    p = Path(__file__).resolve().parents[3] / "config" / "baseline" / "mpc.yaml"
    try:
        return (yaml.safe_load(p.read_text()) or {}).get("comfort_map", {})
    except Exception:
        return {}


_CM = _load_comfort_cfg()
_X0 = float(_CM.get("x0", 20.0))
_PRE_M = float(_CM.get("pre_m", 2.0))
_SETTLE_S = float(_CM.get("settle_s", 2.0))
_MAX_STEPS = int(_CM.get("max_steps", 10000))
_DV = float(_CM.get("dv", 0.25))
_TRACK_SCALE = None
_ISO = bool(_CM.get("iso_weighted", False))
_VEH = _CM.get("vehicle") or {}


class ComfortMap:
    def __init__(self, cfg: RewardConfig, physics: dict | None = None,
                 dv: float = _DV, vehicle: str = "half"):
        self._cfg = cfg
        self._vs  = make_vehicle_spec(vehicle, physics, _VEH)
        self._ode = self._vs.ode
        self._dv = float(dv)
        self._grid = np.arange(cfg.v_min, cfg.v_limit + 1e-9, self._dv)
        self._cache: dict[tuple[float, float, float], float] = {}


    def optimal_speed(self, A: float, L: float, v_nom: float) -> float:
        key = (round(float(A), 4), round(float(L), 3), round(float(v_nom), 2))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        best_v, best_R = float(v_nom), -np.inf
        for v in self._grid:
            R = self._window_reward(A, L, float(v), v_nom)
            if R > best_R:
                best_R, best_v = R, float(v)
        self._cache[key] = best_v
        return best_v


    def _window_reward(self, A: float, L: float, v: float, v_nom: float) -> float:
        from road.road_generator import RoadGenerator

        cfg = self._cfg
        road = RoadGenerator(vehicle_speed=v)
        road._bumps = [(_X0, A, L)]

        x = self._ode.reset(v0=v)
        s = _X0 - _PRE_M
        s_end = _X0 + L + v * _SETTLE_S
        n = 0
        j_comfort = 0.0
        z_b = []
        while s < s_end and n < _MAX_STEPS:
            x[self._vs.v_index] = v
            step_out = self._ode.step(x, road, s, v)
            x = step_out[0]
            s += v * DT
            n += 1
            j_comfort += self._vs.comfort(step_out, cfg)
            z_b.append(step_out[1])

        if _ISO and z_b:
            from eval.iso2631 import weight_wk

            z = np.asarray(z_b, dtype=float)
            w = weight_wk(z, DT)
            j_comfort += cfg.Q_zBddot * (np.sum((z / cfg.g) ** 2) - np.sum((w / cfg.g) ** 2))

        scale = getattr(cfg, 'track_scale', 1.0) if _TRACK_SCALE is None else _TRACK_SCALE
        track = scale * n * cfg.Q_v * (-abs(v - v_nom) / max(v_nom, 1e-6))
        step = n * cfg.Q_step
        return j_comfort + track + step
