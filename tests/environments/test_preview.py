import numpy as np

from road.road_generator import RoadGenerator


def _undulating_road_with_one_bump():
    ds = 0.05
    s = np.arange(0, 60, ds)
    z = 0.04 * np.sin(2 * np.pi * s / 25.0)
    road = RoadGenerator.from_profile(z, ds)
    road._bumps = [(30.0, 0.05, 2.0)]
    return road


def test_bump_preview_ignores_slow_undulation():
    road = _undulating_road_with_one_bump()
    heights = road.get_bump_preview(s_pos=0.0, lookahead_m=40.0, n_points=200)
    assert heights.max() > 0.04
    away = np.linspace(0.2, 40.0, 200) < 29.0
    assert np.all(heights[away] == 0.0)


def test_profile_preview_still_contains_the_undulation():
    road = _undulating_road_with_one_bump()
    heights = road.get_spatial_preview(s_pos=0.0, v_current=6.0, lookahead_m=40.0, n_points=200)
    assert heights[:100].max() > 0.03
