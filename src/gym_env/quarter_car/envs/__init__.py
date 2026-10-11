from gymnasium.envs.registration import register

from gym_env.common.car_env import CarEnv


class QuarterCarEnv(CarEnv):
    VEHICLE = "quarter"


register(
    id="gym_env/QuarterCar",
    entry_point="gym_env.quarter_car.envs:QuarterCarEnv",
)

__all__ = ["QuarterCarEnv"]
