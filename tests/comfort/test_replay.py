import numpy as np

from eval.comfort_score import _metrics, end_point, replay
from gym_env.half_car.core.ode_model import HalfCarODE
from road.road_generator import RoadGenerator


def _score(road, kmh):
    v = kmh / 3.6
    end = end_point(road, HalfCarODE().wheelbase)
    speeds = [v] * int(np.ceil(end / (v * 0.02)) + 5)
    z_b, phi, s_pos = replay(road, speeds, v)
    return _metrics(z_b, phi, s_pos, speeds, end)


def test_flat_road_is_comfortable():
    road = RoadGenerator.from_scenario(vehicle_speed=8.0, flat_start=10.0, obstacles=[])
    road._bumps = [(10.0, 0.0, 1.0)]
    assert _score(road, 28.8)["a_w"] < 1e-3


def test_slower_is_more_comfortable_over_a_bump():
    road = lambda: RoadGenerator.from_scenario(vehicle_speed=8.0, flat_start=20.0, obstacles=[(0.1, 5.0)])
    assert _score(road(), 12.0)["a_w"] < _score(road(), 30.0)["a_w"]
