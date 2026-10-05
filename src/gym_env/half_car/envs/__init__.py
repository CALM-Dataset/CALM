
from gymnasium.envs.registration import register
from gym_env.half_car.envs.half_car_env import HalfCarEnv

register(
    id="gym_env/HalfCar",
    entry_point="gym_env.half_car.envs:HalfCarEnv",
)

__all__ = [
    "HalfCarEnv",
]
