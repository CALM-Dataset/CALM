from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Vehicle:
    name: str
    env_id: str
    env_module: str
    params_module: str
    ode_module: str
    ode_class: str
    render_module: str

    def ensure_registered(self) -> None:
        import_module(self.env_module)

    def params(self):
        return import_module(self.params_module)

    def make_ode(self, params: dict | None = None):
        return getattr(import_module(self.ode_module), self.ode_class)(params)

    def default_plant_params(self) -> dict | None:
        return getattr(self.params(), "PLANT_PARAMS", None)

    def obs_bounds(self):
        p = self.params()
        return p.OBS_LOW, p.OBS_HIGH

    def physics(self) -> dict:
        return dict(getattr(self.params(), "PHYSICS", {}))

    def render(self):
        return import_module(self.render_module)

    @property
    def v_index(self) -> int:
        return import_module(self.ode_module).__dict__[self.ode_class].v_index


VEHICLES = {
    "half": Vehicle(
        name="half", env_id="gym_env/HalfCar",
        env_module="gym_env.half_car.envs",
        params_module="gym_env.half_car.config.env_params",
        ode_module="gym_env.half_car.core.ode_model", ode_class="HalfCarODE",
        render_module="gym_env.half_car.envs.render",
    ),
    "quarter": Vehicle(
        name="quarter", env_id="gym_env/QuarterCar",
        env_module="gym_env.quarter_car.envs",
        params_module="gym_env.quarter_car.config.env_params",
        ode_module="gym_env.quarter_car.core.ode_model", ode_class="QuarterCarODE",
        render_module="gym_env.quarter_car.envs.render",
    ),
}


def get_vehicle(name: str | None) -> Vehicle:
    name = "quarter" if name is None else name
    if name not in VEHICLES:
        raise ValueError(f"unknown vehicle {name!r} (expected one of {sorted(VEHICLES)})")
    vehicle = VEHICLES[name]
    vehicle.ensure_registered()
    return vehicle
