import numpy as np
import pytest

from eval.iso2631 import iso_band, rms, vdv, weight_wk, wk_magnitude

ISO_WK = {0.5: 0.418, 1: 0.482, 2: 0.531, 4: 0.967, 5: 1.039, 6.3: 1.054, 8: 1.036,
          10: 0.988, 12.5: 0.902, 16: 0.768, 20: 0.636, 31.5: 0.405, 50: 0.246}


@pytest.mark.parametrize("freq,expected", list(ISO_WK.items()))
def test_wk_matches_the_standard_table(freq, expected):
    assert wk_magnitude([freq])[0] == pytest.approx(expected, abs=0.005)


def test_weighting_ignores_a_constant_offset():
    t = np.arange(0.0, 10.0, 0.02)
    x = np.sin(2 * np.pi * 5 * t)
    assert rms(weight_wk(x + 3.0, 0.02)) == pytest.approx(rms(weight_wk(x, 0.02)), rel=0.02)


def test_weighted_rms_of_a_five_hertz_sine():
    t = np.arange(0.0, 10.0, 0.02)
    x = np.sin(2 * np.pi * 5 * t)
    assert rms(weight_wk(x, 0.02)) == pytest.approx(1.039 / np.sqrt(2), rel=0.02)


def test_vdv_and_rms_of_known_signals():
    assert rms([3.0, 4.0]) == pytest.approx(np.sqrt(12.5))
    assert vdv(np.ones(50), 0.02) == pytest.approx(1.0)
    assert rms([]) == 0.0


def test_iso_bands():
    assert iso_band(0.1) == "not uncomfortable"
    assert iso_band(0.5) == "a little uncomfortable"
    assert iso_band(3.0) == "extremely uncomfortable"
