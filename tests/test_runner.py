import gymnasium as gym

from baseline import runner
from baseline.reactive_driver.driver import ReactiveController
from gym_env.common import TerminationWrapper
from gym_env.vehicles import get_vehicle
from road.road_generator import RoadGenerator


def _controller():
    return ReactiveController(v_max=8.0, v_min=2.5, a_max=5.0, v_index=get_vehicle("half").v_index)


def test_an_episode_interrupted_before_its_first_step_still_reports():
    env = TerminationWrapper(gym.make(get_vehicle("half").env_id, render_mode="human", action_mode="acceleration"))
    road = RoadGenerator.from_scenario(vehicle_speed=8.0, flat_start=15.0, obstacles=[(0.05, 2.0)])
    for request in ("next", "prev", "replay"):
        env.unwrapped._nav_request = request
        result = runner.run_episode(env, _controller(), 0, lambda r: 1200, scenario_road=road, render_live=True)
        env.unwrapped._nav_request = None
        assert result["n_steps"] == 0
        assert result["act_ms_mean"] == 0.0 and result["act_ms_max"] == 0.0
    env.close()
