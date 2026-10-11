from gymnasium.envs.registration import register

from gym_env.common.car_env import CarEnv


class HalfCarEnv(CarEnv):
    VEHICLE = "half"


register(
    id="gym_env/HalfCar",
    entry_point="gym_env.half_car.envs:HalfCarEnv",
)

__all__ = ["HalfCarEnv"]
