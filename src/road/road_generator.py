from __future__ import annotations
from typing import Optional
import json
from pathlib import Path
import numpy as np


def _get_catalog() -> list:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        p = parent / 'config' / 'catalog' / 'speed_bumps.json'
        if p.is_file():
            return json.loads(p.read_text())['bumps']
    raise FileNotFoundError('speed_bumps.json not found in config/catalog/')


_DEFAULT_BUMP_CONFIG = {
    "num_bumps":     1,
    "bump_sequence": [0],
    "constant_dis":  20.0,
    "bump_x_start":  60.0,
}


class RoadGenerator:
    def __init__(self, vehicle_speed: float = 6.0,
                 params: Optional[dict] = None):
        self.speed = vehicle_speed
        self._bump_labels: Optional[list] = None
        self._profile: Optional[tuple] = None
        self._s_bump: Optional[float] = None

        if params and all(k in params for k in ('bump_height', 'bump_length', 'bump_x_start')):
            self._bumps = [(
                float(params['bump_x_start']),
                float(params['bump_height']),
                float(params['bump_length']),
            )]
        else:
            mc = (params or {}).get('bump_config', _DEFAULT_BUMP_CONFIG)
            self._bumps = self._build_bumps(mc)

    @staticmethod
    def _build_bumps(config: dict) -> list:
        catalog = _get_catalog()
        bumps   = []
        x       = float(config["bump_x_start"])
        seq     = config["bump_sequence"]
        n       = min(int(config["num_bumps"]), len(seq))
        gap     = float(config.get("constant_dis", 5.0))

        for i in range(n):
            ci = int(seq[i])
            A  = float(catalog[ci]['height_m'])
            L  = float(catalog[ci]['width_m'])
            bumps.append((x, A, L))
            if i < n - 1:
                x += L + gap

        return bumps

    def get_height_at(self, s: float) -> float:
        if self._profile is not None:
            s_grid, z, _ = self._profile
            return float(np.interp(s, s_grid, z, left=0.0, right=0.0))
        for x0, A, L in self._bumps:
            dx = s - x0
            if 0.0 <= dx <= L:
                return (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L))
        return 0.0

    def get_height_dot_at(self, s: float, v: float) -> float:
        if self._profile is not None:
            s_grid, _, dz = self._profile
            return float(np.interp(s, s_grid, dz, left=0.0, right=0.0)) * v
        for x0, A, L in self._bumps:
            dx = s - x0
            if 0.0 < dx < L:
                dzdx = (A / 2.0) * (2.0 * np.pi / L) * np.sin(2.0 * np.pi * dx / L)
                return dzdx * v
        return 0.0

    def get_height_array_pos(self, s_array: np.ndarray) -> np.ndarray:
        s      = np.asarray(s_array, dtype=np.float64)
        if self._profile is not None:
            s_grid, z, _ = self._profile
            return np.interp(s, s_grid, z, left=0.0, right=0.0)
        result = np.zeros(len(s))
        for x0, A, L in self._bumps:
            dx   = s - x0
            mask = (dx >= 0.0) & (dx <= L)
            result = np.where(mask, result + (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L)), result)
        return result

    def get_spatial_preview(
        self,
        s_pos: float,
        v_current: float,
        lookahead_m: float,
        n_points: int,
    ) -> np.ndarray:
        s_offsets = np.linspace(lookahead_m / n_points, lookahead_m, n_points)
        x_ahead   = s_pos + s_offsets
        if self._profile is not None:
            s_grid, z, _ = self._profile
            return np.interp(x_ahead, s_grid, z, left=0.0, right=0.0).astype(np.float32)
        return self._bump_heights(x_ahead)

    def get_bump_preview(
        self,
        s_pos: float,
        lookahead_m: float,
        n_points: int,
    ) -> np.ndarray:
        s_offsets = np.linspace(lookahead_m / n_points, lookahead_m, n_points)
        return self._bump_heights(s_pos + s_offsets)

    def _bump_heights(self, x_ahead: np.ndarray) -> np.ndarray:
        heights = np.zeros(len(x_ahead))
        for x0, A, L in self._bumps:
            dx   = x_ahead - x0
            mask = (dx >= 0.0) & (dx <= L)
            heights = np.where(mask, heights + (A / 2.0) * (1.0 - np.cos(2.0 * np.pi * dx / L)), heights)
        return heights.astype(np.float32)

    def _clamp_speed_to_geometry(self) -> None:
        ZETA_DOT_LIMIT = 7.0
        if self._profile is not None:
            _, _, dz = self._profile
            steepest = float(np.max(np.abs(dz))) if len(dz) else 0.0
        elif self._bumps:
            steepest = max(np.pi * abs(A) / L for _, A, L in self._bumps)
        else:
            return
        if steepest > 0:
            v_lim = ZETA_DOT_LIMIT / steepest
            self.speed = float(np.clip(self.speed, 0.0, v_lim))

    def set_speed(self, v: float) -> None:
        self.speed = float(v)
        self._clamp_speed_to_geometry()

    def reset(self, seed=None):
        pass

    @classmethod
    def from_random(
        cls,
        rng: np.random.Generator,
        vehicle_speed: float,
        params: Optional[dict] = None,
        *,
        catalog_ids: Optional[list] = None,
        explicit_obstacles: Optional[list] = None,
        flat_start: float = 8.0,
        hole_fraction: float = 0.0,
    ) -> 'RoadGenerator':
        if explicit_obstacles:
            pool = [(float(h), float(w)) for h, w in explicit_obstacles]
        else:
            catalog  = _get_catalog()
            eligible = [catalog[i] for i in catalog_ids] if catalog_ids else catalog
            pool = [(float(e['height_m']), float(e['width_m'])) for e in eligible]

        _a_max = 5.0
        flat_start_safe = max(float(flat_start), vehicle_speed ** 2 / (2.0 * _a_max))
        A, L = pool[int(rng.integers(0, len(pool)))]
        if hole_fraction > 0.0 and rng.uniform() < hole_fraction:
            A = -abs(A)
        bumps = [(flat_start_safe, A, L)]

        gen = cls(vehicle_speed=vehicle_speed, params=params)
        gen._bumps = bumps
        gen._clamp_speed_to_geometry()
        return gen

    @classmethod
    def from_profile(
        cls,
        z,
        ds_m: float,
        s_bump: Optional[float] = None,
        vehicle_speed: float = 6.0,
    ) -> 'RoadGenerator':
        z = np.asarray(z, dtype=np.float64)
        s_grid = np.arange(len(z)) * float(ds_m)
        dz = np.gradient(z, float(ds_m)) if len(z) > 1 else np.zeros_like(z)
        gen = cls(vehicle_speed=vehicle_speed)
        gen._bumps = []
        gen._profile = (s_grid, z, dz)
        gen._s_bump = None if s_bump is None else float(s_bump)
        gen._clamp_speed_to_geometry()
        return gen

    @classmethod
    def from_scenario(
        cls,
        vehicle_speed: float,
        flat_start: float,
        obstacles: list,
        gaps: Optional[list] = None,
    ) -> 'RoadGenerator':
        gaps = gaps or []
        bumps: list = []
        x = float(flat_start)
        for i, (h, w) in enumerate(obstacles):
            bumps.append((x, float(h), float(w)))
            if i < len(obstacles) - 1:
                x += float(w) + float(gaps[i])

        gen = cls(vehicle_speed=vehicle_speed)
        gen._bumps = bumps
        gen._clamp_speed_to_geometry()
        return gen
