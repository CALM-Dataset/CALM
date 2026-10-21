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


def _env_with(road, features="v2"):
    from dataclasses import replace
    from gym_env.half_car.envs import HalfCarEnv
    from gym_env.common.preview.preview import PreviewWrapper
    from gym_env.common.preview.preview_params import load_preview_config

    env = HalfCarEnv(action_mode="slowdown", random_road_on_reset=False)
    wrapped = PreviewWrapper(env, replace(load_preview_config(), features=features, noise_active=False, perception_mode="none"))
    wrapped.reset(seed=0, options={"road": road})
    return wrapped


def _bump_road(x0=20.0, A=0.06, L=3.0):
    ds = 0.05
    s = np.arange(0, 80, ds)
    road = RoadGenerator.from_profile(np.zeros_like(s), ds)
    road._bumps = [(x0, A, L)]
    return road


def test_v2_bump_length_feature_does_not_depend_on_speed():
    env = _env_with(_bump_road())
    n = env.observation_space.shape[0]
    base = env.env.unwrapped
    a = env.observation(base._obs())[n - 15 + 1]
    base._v = base._v * 0.6
    b = env.observation(base._obs())[n - 15 + 1]
    assert a == b and 0.0 < a < 1.0


def test_v2_distance_shrinks_and_recovery_flags_follow_the_bump():
    env = _env_with(_bump_road())
    base = env.env.unwrapped
    n = env.observation_space.shape[0]
    first = lambda: env.observation(base._obs())
    d0 = first()[n - 15]
    base._s_pos = 10.0
    d1 = first()[n - 15]
    assert d1 < d0
    base._s_pos = 21.0
    assert first()[n - 1] == 1.0
    base._s_pos = 30.0
    obs = first()
    assert obs[n - 1] == 0.0 and obs[n - 2] < 1.0 and obs[n - 15] == 1.0
