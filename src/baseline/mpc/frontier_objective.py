from __future__ import annotations

import copy

import numpy as np

from gym_env.common.env_params import DT
from gym_env.common.reward.reward_params import RewardConfig, load_reward_config
from baseline.mpc.config import load_mpc_cfg, ramp_decel
from baseline.mpc.vehicle import make_vehicle_spec

_FRONTIER_POINTS = 8
_PASSES = 2
_VEH = (load_mpc_cfg().get("comfort_map") or {}).get("vehicle") or {}
_DV = 0.4


class FrontierPlanner:
    def __init__(self, mpc_cfg: RewardConfig, vehicle: str = "half", physics: dict | None = None,
                 reward_cfg: RewardConfig | None = None):
        self._m = mpc_cfg
        self._r = reward_cfg or load_reward_config()
        vs = make_vehicle_spec(vehicle, physics, _VEH)
        self._ode = vs.ode
        self._vi = vs.v_index
        self._wb = float(self._ode.wheelbase)
        self._cache: dict[tuple, list[float]] = {}
        self.ramp_decel = ramp_decel()

    def _profile(self, s: float, bumps, speeds, v_nom: float) -> float:
        v = v_nom
        a_max = self.ramp_decel
        for (x0, _, length), vc in zip(bumps, speeds):
            vc = min(vc, v_nom)
            ramp = max(0.0, v_nom ** 2 - vc ** 2) / (2.0 * max(a_max, 1e-6))
            x_ramp = x0 - ramp
            x_end = x0 + length + self._wb
            x_done = x_end + ramp
            if s < x_ramp or s >= x_done:
                continue
            if s < x0:
                v = min(v, v_nom - (v_nom - vc) * (s - x_ramp) / max(ramp, 1e-6))
            elif s <= x_end:
                v = min(v, vc)
            else:
                v = min(v, vc + (v_nom - vc) * (s - x_end) / max(ramp, 1e-6))
        return float(np.clip(v, self._m.v_min, self._m.v_limit))

    def _drive(self, road, speed_at, end: float):
        from eval.iso2631 import weight_wk

        road = copy.deepcopy(road)
        v = float(speed_at(0.0))
        x = self._ode.reset(v0=v)
        s, n_max = 0.0, int(600.0 / DT)
        z_b, speeds = [], []
        v_prev = v
        while s < end and len(z_b) < n_max:
            v = float(speed_at(s))
            a = (v - v_prev) / DT
            v_prev = v
            x[self._vi] = v
            road.set_speed(v)
            s_start = s
            s += v * DT
            x, zb, *_ = self._ode.step(x, road, s_start, road.speed, a)
            z_b.append(zb)
            speeds.append(v)
        return np.asarray(z_b), np.asarray(speeds), len(z_b) * DT, weight_wk(np.asarray(z_b), DT)

    def _heave_cost(self, wk) -> float:
        return float(self._r.Q_zBddot * np.sum((wk / self._r.g) ** 2))

    def _long_cost(self, speeds) -> float:
        r = self._r
        a = np.diff(np.concatenate([[speeds[0]], speeds])) / DT
        da = np.diff(np.concatenate([[0.0], a]))
        return float(r.Q_a * np.sum((a / r.g) ** 2) + r.w_jerk * np.sum((da / r.g) ** 2)
                     + r.w_action_smooth * np.sum((da / r.a_max) ** 2))

    def _frontier(self, road, v_nom: float, end: float):
        low = min(float(self._r.v_hard_min), v_nom)
        pts = []
        for v in np.linspace(low, v_nom, _FRONTIER_POINTS):
            _, _, t, wk = self._drive(road, lambda s, v=v: v, end)
            pts.append((t, self._heave_cost(wk)))
        pts.sort()
        return pts

    def _return(self, road, bumps, speeds, v_nom, end, frontier) -> float:
        _, sp, t, wk = self._drive(road, lambda s: self._profile(s, bumps, speeds, v_nom), end)
        c = self._heave_cost(wk)
        ts, cs = zip(*frontier)
        c_eq = max(float(np.interp(t, ts, cs)), float(self._r.rel_min_cost))
        t_ref = max(ts[0], DT)
        s = np.cumsum(sp) * DT
        lag = np.maximum(np.arange(1, len(sp) + 1) * DT - s / v_nom, 0.0)
        price = self._r.rel_time_quad * self._r.rel_scale * float(np.sum((lag / t_ref) ** 2)) * DT / t_ref
        return -self._r.rel_scale * c / c_eq - price - self._long_cost(sp)

    def plan(self, road, v_nom: float, end: float) -> list[float]:
        bumps = [(float(x0), float(a), float(l)) for x0, a, l in (getattr(road, "_bumps", None) or [])]
        if not bumps:
            return []
        key = (round(v_nom, 3), round(end, 2), tuple(tuple(round(x, 4) for x in b) for b in bumps))
        if key in self._cache:
            return self._cache[key]
        frontier = self._frontier(road, v_nom, end)
        grid = np.unique(np.concatenate([np.arange(float(self._m.v_min), v_nom, _DV), [v_nom]]))
        speeds = [v_nom] * len(bumps)
        best = self._return(road, bumps, speeds, v_nom, end, frontier)
        for _ in range(_PASSES):
            changed = False
            for i in range(len(bumps)):
                for vc in grid:
                    trial = list(speeds)
                    trial[i] = float(vc)
                    r = self._return(road, bumps, trial, v_nom, end, frontier)
                    if r > best + 1e-9:
                        best, speeds, changed = r, trial, True
            if not changed:
                break
        self._cache[key] = speeds
        return speeds
