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
