import numpy as np

from utils.scenario_loader import edge_trim_indices

DS = 0.05
CFG = {"trim_edges": True, "edge_tol_m": 0.01, "bump_margin_m": 0.5}


def _profile(lead, tail):
    n = 600
    z = np.zeros(n)
    z[:lead] = -0.05
    z[lead:lead + 40] = np.linspace(-0.05, 0.0, 40)
    z[n - tail:] = 0.05
    pos = np.arange(n) * DS
    s0, s1 = 15.0, 17.0
    z[(pos >= s0) & (pos <= s1)] = 0.04
    return z, [{"s_start_m": s0, "s_end_m": s1}]


def test_edges_that_do_not_sit_at_road_level_are_trimmed():
    z, bumps = _profile(lead=30, tail=60)
    start, end = edge_trim_indices(z, DS, bumps, CFG)
    assert start > 0 and end < len(z)
    assert abs(z[start]) <= CFG["edge_tol_m"] and abs(z[end - 1]) <= CFG["edge_tol_m"]


def test_the_bump_is_never_trimmed():
    z, bumps = _profile(lead=30, tail=60)
    start, end = edge_trim_indices(z, DS, bumps, CFG)
    assert start * DS < bumps[0]["s_start_m"] and end * DS > bumps[0]["s_end_m"]


def test_a_profile_that_already_starts_and_ends_level_is_untouched():
    z, bumps = _profile(lead=0, tail=0)
    z[:40] = 0.0
    assert edge_trim_indices(z, DS, bumps, CFG) == (0, len(z))


def test_trimming_can_be_disabled():
    z, bumps = _profile(lead=30, tail=60)
    assert edge_trim_indices(z, DS, bumps, {**CFG, "trim_edges": False}) == (0, len(z))
