from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class VehicleSpec:
    name:    str
    ode:     Any
    v_index: int
    comfort: Callable[[tuple, Any], float]


def _quarter_comfort(step_out: tuple, cfg) -> float:
    _, z_B_ddot, z_W_ddot = step_out
    g = cfg.g
    return (cfg.Q_zBddot * -(z_B_ddot / g) ** 2
            + cfg.Q_zWddot * -(z_W_ddot / g) ** 2)


def _half_comfort(step_out: tuple, cfg) -> float:
    import numpy as np
    _, z_B_ddot, phi_ddot, z_Wf_ddot, _ = step_out
    g = cfg.g
    pitch = (cfg.pitch_arm * np.clip(phi_ddot, -cfg.pitch_clip, cfg.pitch_clip)) / g
    return (cfg.Q_zBddot * -(z_B_ddot / g) ** 2
            + cfg.Q_zWddot * -(z_Wf_ddot / g) ** 2
            + cfg.Q_pitch  * -(pitch) ** 2)


def _scaled_half_params(scales: dict | None) -> dict | None:
    if not scales:
        return None
    from gym_env.half_car.core import ode_model as om

    ks, dd = float(scales.get("k_S_scale", 1.0)), float(scales.get("D_scale", 1.0))
    return {axle: {"k_S": base["k_S"] * ks, "D": base["D"] * dd}
            for axle, base in (("front", om.PHYSICS_F), ("rear", om.PHYSICS_R))}


def make_vehicle_spec(vehicle: str, physics: dict | None = None,
                      scales: dict | None = None) -> VehicleSpec:
    if vehicle == "half":
        from gym_env.half_car.core.ode_model import HalfCarODE
        return VehicleSpec("half", HalfCarODE(_scaled_half_params(scales)), v_index=8,
                           comfort=_half_comfort)
    if vehicle not in ("quarter", None):
        raise ValueError(f"unknown vehicle {vehicle!r} (expected 'quarter' or 'half')")
    from gym_env.quarter_car.core.ode_model import QuarterCarODE
    return VehicleSpec("quarter", QuarterCarODE(physics), v_index=4,
                       comfort=_quarter_comfort)
