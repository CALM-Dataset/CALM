from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import yaml

if TYPE_CHECKING:
    from road.road_generator import RoadGenerator


def _load_params() -> dict:
    here = Path(__file__).resolve()
    for parent in here.parents:
        p = parent / 'config' / 'baseline' / 'reactive_driver.yaml'
        if p.is_file():
            with open(p) as fh:
                return yaml.safe_load(fh)
    return {}


class ReactiveController:
    def __init__(
        self,
        v_max:          Optional[float] = None,
        v_min:          Optional[float] = None,
        a_max:          Optional[float] = None,
        preview_m:      Optional[float] = None,
        steep_full_slow: Optional[float] = None,
        hole_severity:  Optional[float] = None,
        brake_margin:   Optional[float] = None,
        a_brake:        Optional[float] = None,
        a_accel:        Optional[float] = None,
        clear_m:        Optional[float] = None,
        v_index:        int = 4,
    ):
        p = _load_params()
        self.v_index         = int(v_index)
        self.v_max           = v_max
        self.v_min           = v_min
        self.a_max           = a_max
        self.preview_m       = preview_m       if preview_m       is not None else float(p.get('preview_m',       40.0))
        self.steep_full_slow = steep_full_slow if steep_full_slow is not None else float(p.get('steep_full_slow', 0.10))
        self.hole_severity   = hole_severity   if hole_severity   is not None else float(p.get('hole_severity',   0.60))
        self.brake_margin    = brake_margin    if brake_margin    is not None else float(p.get('brake_margin',    3.0))
        self.a_brake         = a_brake         if a_brake         is not None else float(p.get('a_brake',         2.5))
        self.a_accel         = a_accel         if a_accel         is not None else float(p.get('a_accel',         2.0))
        self.clear_m         = clear_m         if clear_m         is not None else float(p.get('clear_m',         0.0))

    def _crossing_speed(self, H: float, W: float) -> float:
        steepness = abs(H) / max(W, 1e-6)
        severity  = steepness / max(self.steep_full_slow, 1e-6)
        if H < 0.0:
            severity = max(severity, self.hole_severity)
        severity = float(np.clip(severity, 0.0, 1.0))
        return float(self.v_max - severity * (self.v_max - self.v_min))

    def act(self, x: np.ndarray, s_pos: float, road: 'RoadGenerator') -> float:
        v = float(x[self.v_index])
        self.v_max = self.v_max if self.v_max is not None else v
        self.v_min = self.v_min if self.v_min is not None else 0.0
        a_max      = self.a_max if self.a_max is not None else 5.0

        need_brake = False
        on_bump    = False
        for x0, A, L in getattr(road, '_bumps', []):
            if s_pos >= x0 + L + self.clear_m:
                continue
            v_cross = self._crossing_speed(A, L)
            if s_pos >= x0:
                on_bump = True
                if v > v_cross:
                    need_brake = True
                continue
            d = x0 - s_pos
            if d > self.preview_m or v <= v_cross:
                continue
            d_brake = (v * v - v_cross * v_cross) / (2.0 * self.a_brake)
            if d <= d_brake + self.brake_margin:
                need_brake = True

        if need_brake:
            a = -self.a_brake
        elif on_bump:
            a = 0.0
        elif v < self.v_max:
            a = self.a_accel
        else:
            a = 0.0

        a = float(np.clip(a, -self.a_brake, self.a_accel))
        return float(np.clip(a / a_max, -1.0, 1.0))
