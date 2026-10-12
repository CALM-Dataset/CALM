import numpy as np
import pytest

from gym_env.common.env_params import DT
from gym_env.vehicles import get_vehicle
from road.road_generator import RoadGenerator

VEHICLES = ["quarter", "half"]


def drive(vehicle, bump_height, speed):
    ode = get_vehicle(vehicle).make_ode()
    road = RoadGenerator.from_scenario(vehicle_speed=speed, flat_start=15.0, obstacles=[(bump_height, 1.5)] if bump_height else [])
    road.set_speed(speed)
    x = ode.reset(speed)
    s, accel = 0.0, []
    while s < 40.0:
        x, z_b_ddot, *_ = ode.step(x, road, s, speed, 0.0)
        accel.append(z_b_ddot)
        s += speed * DT
    return np.array(accel)


@pytest.mark.parametrize("vehicle", VEHICLES)
def test_a_flat_road_leaves_the_body_alone(vehicle):
    assert np.abs(drive(vehicle, 0.0, 8.0)).max() < 0.05


@pytest.mark.parametrize("vehicle", VEHICLES)
def test_a_bump_shakes_the_body_and_it_settles_again(vehicle):
    accel = drive(vehicle, 0.06, 8.0)
    third = len(accel) // 3
    assert np.abs(accel).max() > 0.5
    assert np.abs(accel[-third:]).mean() < 0.2 * np.abs(accel[:2 * third]).max()


@pytest.mark.parametrize("vehicle", VEHICLES)
def test_a_taller_bump_is_rougher(vehicle):
    assert np.abs(drive(vehicle, 0.08, 8.0)).max() > np.abs(drive(vehicle, 0.03, 8.0)).max()


@pytest.mark.parametrize("vehicle", VEHICLES)
def test_slowing_down_over_a_bump_is_gentler(vehicle):
    slow, fast = drive(vehicle, 0.06, 3.0), drive(vehicle, 0.06, 9.0)
    assert np.sqrt(np.mean(slow ** 2)) < np.sqrt(np.mean(fast ** 2))
