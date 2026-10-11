from __future__ import annotations

from pathlib import Path

import numpy as np

from gym_env.common.env_params import DT

STRIDE = 2
SHRINK = 2


def keep(step: int) -> bool:
    return step % STRIDE == 0


def shrink(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame[::SHRINK, ::SHRINK])


def save_gif(frames: list, path: Path, dt: float = DT) -> None:
    try:
        from PIL import Image
    except ImportError:
        print('  [Pillow not installed - cannot save GIF; run: pip install Pillow]')
        return
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=int(dt * STRIDE * 1000), loop=0)
    print(f'  GIF saved → {path}  ({len(images)} frames)')
