import pytest

from road.road_generator import RoadGenerator
from utils.paths import ROOT
from utils.scenario_loader import load_scenario_any

SUITES = ROOT / "config" / "eval" / "scenarios"
NAMES = sorted(p.stem for p in SUITES.glob("0*/*.yaml") if p.stem != "manifest")


@pytest.mark.parametrize("name", NAMES)
def test_every_bundled_scenario_loads_with_a_usable_road(name):
    road = load_scenario_any(name)[0]
    assert road._bumps
    for start, height, width in road._bumps:
        assert start > 0 and width > 0 and height != 0


def test_a_seed_gives_the_same_random_road():
    roads = [RoadGenerator.from_random(__import__("numpy").random.default_rng(5), vehicle_speed=8.0) for _ in range(2)]
    assert roads[0]._bumps == roads[1]._bumps


def test_a_flat_road_has_no_relief_and_the_requested_speed():
    road, track, speed, _ = load_scenario_any('flat50')
    z = road._profile[1]
    assert track is None
    assert float(abs(z).max()) == 0.0
    assert abs(speed - 50 / 3.6) < 1e-9


def test_a_bump_road_runs_on_after_its_last_bump():
    from eval.comfort_score import end_point

    road, *_ = load_scenario_any('01_single_bump/c')
    x0, _, length = road._bumps[-1]
    assert end_point(road, 2.5) - (x0 + length + 2.5) == road._runout_m > 0
