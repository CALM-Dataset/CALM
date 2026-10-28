from baseline.mpc.config import reward_config
from baseline.mpc.frontier_objective import FrontierPlanner
from eval.comfort_score import end_point
from gym_env.common.reward.reward_params import load_reward_config
from utils.scenario_loader import load_scenario_any


def _plan(name):
    cfg = reward_config(load_reward_config)
    planner = FrontierPlanner(cfg)
    road, *_ = load_scenario_any(name)
    return road, planner.plan(road, float(road.speed), end_point(road, planner._wb))


def test_a_road_without_bumps_needs_no_crossing_speed():
    assert _plan('flat30')[1] == []


def test_a_severe_bump_is_crossed_slower_than_the_scenario_speed():
    road, speeds = _plan('c')
    assert len(speeds) == 1 and speeds[0] < float(road.speed)
