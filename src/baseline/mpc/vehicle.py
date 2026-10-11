from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from gym_env.vehicles import get_vehicle


@dataclass(frozen=True)
class VehicleSpec:
    name:    str
    ode:     Any
    v_index: int
    comfort: Callable[[tuple, Any], float]


def _comfort(step_out: tuple, cfg) -> float:
    _, z_B_ddot, phi_ddot, z_Wf_ddot, _ = step_out
    g = cfg.g
    pitch = (cfg.pitch_arm * np.clip(phi_ddot, -cfg.pitch_clip, cfg.pitch_clip)) / g
    return (cfg.Q_zBddot * -(z_B_ddot / g) ** 2
            + cfg.Q_zWddot * -(z_Wf_ddot / g) ** 2
            + cfg.Q_pitch  * -(pitch) ** 2)


def _scaled_half_params(scales: dict | None) -> dict | None:
    if not scales:
        return None
    from gym_env.half_car.config.env_params import PHYSICS_F, PHYSICS_R

    ks, dd = float(scales.get("k_S_scale", 1.0)), float(scales.get("D_scale", 1.0))
    return {axle: {"k_S": base["k_S"] * ks, "D": base["D"] * dd}
            for axle, base in (("front", PHYSICS_F), ("rear", PHYSICS_R))}


def make_vehicle_spec(vehicle: str, physics: dict | None = None,
                      scales: dict | None = None) -> VehicleSpec:
    veh = get_vehicle(vehicle)
    params = _scaled_half_params(scales) if veh.name == "half" else physics
    return VehicleSpec(veh.name, veh.make_ode(params), veh.v_index, _comfort)
