from __future__ import annotations

import numpy as np
from scipy import signal

_F1, _F2 = 0.4, 100.0
_F3, _F4, _Q4 = 12.5, 12.5, 0.63
_F5, _F6, _Q5, _Q6 = 2.37, 3.35, 0.91, 0.91
_K = 0.5
_OVERSAMPLE = 10


def _w(f: float) -> float:
    return 2.0 * np.pi * f


def wk_transfer_function() -> tuple[np.ndarray, np.ndarray]:
    w1, w2, w3, w4, w5, w6 = (_w(f) for f in (_F1, _F2, _F3, _F4, _F5, _F6))
    r2 = np.sqrt(2.0)
    num = np.polymul(
        np.polymul([1.0, 0.0, 0.0], [w2 ** 2]),
        np.polymul([w4 ** 2 / w3, w4 ** 2],
                   _K * np.array([1.0 / w5 ** 2, 1.0 / (_Q5 * w5), 1.0])),
    )
    den = np.polymul(
        np.polymul([1.0, r2 * w1, w1 ** 2], [1.0, r2 * w2, w2 ** 2]),
        np.polymul([1.0, w4 / _Q4, w4 ** 2],
                   [1.0 / w6 ** 2, 1.0 / (_Q6 * w6), 1.0]),
    )
    return num, den


def wk_magnitude(freq_hz) -> np.ndarray:
    num, den = wk_transfer_function()
    _, h = signal.freqs(num, den, worN=_w(np.asarray(freq_hz, float)))
    return np.abs(h)


def _wk_sos(fs: float) -> np.ndarray:
    num, den = wk_transfer_function()
    z, p, k = signal.tf2zpk(num, den)
    zd, pd, kd = signal.bilinear_zpk(z, p, k, fs)
    return signal.zpk2sos(zd, pd, kd)


def weight_wk(x, dt: float) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size < 2:
        return np.zeros_like(x)
    fs = _OVERSAMPLE / dt
    up = signal.resample_poly(x, _OVERSAMPLE, 1)
    sos = _wk_sos(fs)
    y, _ = signal.sosfilt(sos, up, zi=signal.sosfilt_zi(sos) * up[0])
    return y[::_OVERSAMPLE][: x.size]


def rms(x) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def vdv(x, dt: float) -> float:
    x = np.asarray(x, float)
    return float((np.sum(x ** 4) * dt) ** 0.25) if x.size else 0.0


def iso_band(a_w: float) -> str:
    for limit, label in ((0.315, 'not uncomfortable'),
                         (0.63, 'a little uncomfortable'),
                         (1.0, 'fairly uncomfortable'),
                         (1.6, 'uncomfortable'),
                         (2.5, 'very uncomfortable')):
        if a_w < limit:
            return label
    return 'extremely uncomfortable'
