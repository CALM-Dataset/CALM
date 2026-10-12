
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from utils.paths import ROOT as _ROOT
_MON_ROOT   = _ROOT / "logs" / "monitor"
_TB_ROOT    = _ROOT / "logs" / "tensorboard"
_OUT        = _ROOT / "data" / "training" /"train_curve.mat"
_N_BINS     = 200

_TB_TAGS = {
    "eval":    "eval/mean_reward",
    "expvar":  "train/explained_variance",
    "entropy": "train/entropy_loss",
    "vloss":   "train/value_loss",
    "kl":      "train/approx_kl",
}


def _monitor_files(run_dir: Path) -> list[Path]:
    files = sorted(run_dir.glob("*.monitor.csv"))
    legacy = run_dir.parent / f"{run_dir.name}.monitor.csv"
    return files or ([legacy] if legacy.is_file() else [])


def _read_monitor(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    episodes: list[tuple[float, float, float]] = []
    for path in _monitor_files(run_dir):
        rows = [r for r in path.read_text().splitlines() if r and not r.startswith("#")]
        if len(rows) < 2:
            continue
        header = rows[0].split(",")
        try:
            ri, li, ti = header.index("r"), header.index("l"), header.index("t")
        except ValueError:
            continue
        for line in rows[1:]:
            parts = line.split(",")
            try:
                episodes.append((float(parts[ti]), float(parts[li]), float(parts[ri])))
            except (ValueError, IndexError):
                continue
    if not episodes:
        return np.empty(0), np.empty(0)
    episodes.sort()
    arr = np.asarray(episodes)
    return np.cumsum(arr[:, 1]), arr[:, 2]


def _read_tb_scalar(event_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(event_dir.glob("**/events.out.tfevents*"))
    steps: list[float] = []
    vals:  list[float] = []
    for f in files:
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0})
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars(tag):
            steps.append(float(e.step))
            vals.append(float(e.value))
    if not steps:
        return np.empty(0), np.empty(0)
    order = np.argsort(steps)
    return np.asarray(steps)[order], np.asarray(vals)[order]


def _aggregate(curves: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curves = [(s, v) for s, v in curves if len(s) > 1]
    if not curves:
        return np.full_like(grid, np.nan), np.full_like(grid, np.nan)
    stack = np.vstack([np.interp(grid, s, v, right=np.nan) for s, v in curves])
    return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def _norm_exp(tag: str) -> str:
    return f"exp_{tag}" if tag.isdigit() else tag


def main(tags: list[str], out_path: Path = _OUT) -> None:
    tags = [_norm_exp(t) for t in tags]
    if tags:
        files = [_MON_ROOT / t / "train" for t in tags]
    else:
        files = sorted(d.parent / "train" for d in _MON_ROOT.glob("exp_*"))
    files = [f for f in files if _monitor_files(f)]
    if not files:
        print("No monitor files found under", _MON_ROOT)
        return

    curves = [_read_monitor(f) for f in files]
    curves = [(s, r) for s, r in curves if len(s) > 1]
    if not curves:
        print("No usable monitor rows.")
        return

    x_full = max(float(s[-1]) for s, _ in curves)
    curves = [(s, r) for s, r in curves if s[-1] >= 0.2 * x_full]
    grid   = np.linspace(0, x_full, _N_BINS)
    stack  = np.vstack([np.interp(grid, s, r, right=np.nan) for s, r in curves])

    out = {
        "steps":       grid,
        "reward_mean": np.nanmean(stack, axis=0),
        "reward_std":  np.nanstd(stack, axis=0),
        "n_seeds":     float(stack.shape[0]),
    }

    if tags:
        tb_dirs = [_TB_ROOT / t for t in tags if (_TB_ROOT / t).is_dir()]
    else:
        tb_dirs = sorted(d for d in _TB_ROOT.glob("exp_*") if d.is_dir())
    if tb_dirs:
        tb_grid = np.linspace(0, x_full, _N_BINS)
        out["tb_steps"] = tb_grid
        for key, tag in _TB_TAGS.items():
            curves = [_read_tb_scalar(d, tag) for d in tb_dirs]
            mu, sd = _aggregate(curves, tb_grid)
            out[f"{key}_mean"] = mu
            out[f"{key}_std"]  = sd
        print(f"  tensorboard: {len(tb_dirs)} run(s), tags {list(_TB_TAGS)}")

    try:
        from scipy.io import savemat
        out_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(out_path), out)
        print(f"wrote {out_path}  ({stack.shape[0]} run(s))")
    except ImportError:
        print("scipy required for .mat export: pip install scipy")


def _parse_args(argv: list[str]) -> tuple[list[str], Path]:
    tags: list[str] = []
    out_path = _OUT
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out_path = Path(argv[i + 1])
            i += 2
        else:
            tags.append(argv[i])
            i += 1
    return tags, out_path


if __name__ == "__main__":
    try:
        _tags, _out_path = _parse_args(sys.argv[1:])
        main(_tags, _out_path)
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
