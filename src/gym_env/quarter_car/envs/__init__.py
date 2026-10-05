
from gymnasium.envs.registration import register
from gym_env.quarter_car.envs.quarter_car_env import QuarterCarEnv

register(
    id="gym_env/QuarterCar",
    entry_point="gym_env.quarter_car.envs:QuarterCarEnv",
)

__all__ = [
    "QuarterCarEnv",
]
