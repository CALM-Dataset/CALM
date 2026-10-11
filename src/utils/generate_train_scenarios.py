#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT          = Path(__file__).resolve().parents[2]
_CATALOG_PATH  = _ROOT / "config" / "catalog" / "speed_bumps.json"
_POTHOLE_PATH  = _ROOT / "config" / "catalog" / "potholes.json"
_GEN_CFG_PATH  = _ROOT / "config" / "train" / "scenario_gen.yaml"
_ALGO_CFG_PATH = _ROOT / "config" / "agent" / "algo_configs.yaml"
_DEFAULT_OUT   = _ROOT / "config" / "train" / "scenarios"

def _load_gen_config() -> dict:
    with open(_GEN_CFG_PATH) as f:
        return yaml.safe_load(f)

def _load_training_seed() -> int:
    with open(_ALGO_CFG_PATH) as f:
        return int(yaml.safe_load(f)["training"]["seed"])

_gcfg = _load_gen_config()

_BUMP_RANK: dict[int, float] = {int(k): float(v) for k, v in _gcfg["bump_rank"].items()}
_POTHOLE_RANK: dict[int, float] = {int(k): float(v) for k, v in _gcfg.get("pothole_rank", {}).items()}
_V_MIN_KMH = float(_gcfg["v_min_kmh"])
_V_MAX_KMH = float(_gcfg["v_max_kmh"])
_THRESHOLDS = [
    (name, float(lo), float(hi))
    for name, (lo, hi) in _gcfg["difficulty_thresholds"].items()
]
_DIFFICULTIES = ("easy", "medium", "hard", "expert")
_MODES = ("bumps", "potholes", "both")
_MODE_DEFAULT = str(_gcfg.get("obstacle_mode", "both")).lower()


def _sample_real_range(rng: np.random.Generator) -> tuple[list[dict], dict[int, float]]:
    rr = _gcfg["real_range"]
    n = int(rr["samples"])
    h_lo, h_hi = (float(x) / 1000.0 for x in rr["height_mm"])
    w_lo, w_hi = (float(x) for x in rr["width_m"])
    heights = np.exp(rng.uniform(np.log(h_lo), np.log(h_hi), n))
    widths = np.exp(rng.uniform(np.log(w_lo), np.log(w_hi), n))
    order = np.argsort(np.argsort(heights / widths))
    bumps = [{"id": 1000 + i, "name": "sampled", "height_m": round(float(heights[i]), 4),
              "width_m": round(float(widths[i]), 3)} for i in range(n)]
    return bumps, {1000 + i: float(order[i]) / max(n - 1, 1) for i in range(n)}


def _load_catalog(mode: str, rng: np.random.Generator) -> tuple[list[dict], dict[int, float]]:
    if mode == "bumps" and "real_range" in _gcfg:
        return _sample_real_range(rng)
    bumps = json.loads(_CATALOG_PATH.read_text())["bumps"]
    pots  = json.loads(_POTHOLE_PATH.read_text())["potholes"]
    if mode == "bumps":
        return bumps, dict(_BUMP_RANK)
    if mode == "potholes":
        return pots, dict(_POTHOLE_RANK)
    return bumps + pots, {**_BUMP_RANK, **_POTHOLE_RANK}


def _difficulty_score(rank: float, speed_kmh: float) -> float:
    speed_n = float(np.clip((speed_kmh - _V_MIN_KMH) / (_V_MAX_KMH - _V_MIN_KMH), 0.0, 1.0))
    return 0.6 * rank + 0.4 * speed_n


def _assign_difficulty(score: float) -> str:
    for name, lo, hi in _THRESHOLDS:
        if lo <= score < hi:
            return name
    return "expert"


def _gen_single(catalog: list[dict], rank: dict[int, float], per_level: int,
                 fs_min: float, fs_max: float, rng: np.random.Generator) -> dict[str, list[dict]]:
    n_speeds = max(100, per_level * 2)
    speeds   = np.linspace(_V_MIN_KMH, _V_MAX_KMH, n_speeds)

    by_diff: dict[str, list[dict]] = {d: [] for d in _DIFFICULTIES}
    for bump in catalog:
        cid = bump["id"]
        if cid not in rank:
            continue
        for v_raw in speeds:
            v     = round(float(v_raw), 2)
            score = _difficulty_score(rank[cid], v)
            diff  = _assign_difficulty(score)
            by_diff[diff].append({
                "catalog_id":        cid,
                "catalog_name":      bump["name"],
                "speed_kmh":         v,
                "difficulty":        diff,
                "difficulty_score":  round(score, 4),
                "height_m":          bump["height_m"],
                "width_m":           bump["width_m"],
            })

    selected: dict[str, list[dict]] = {d: [] for d in _DIFFICULTIES}
    for diff in _DIFFICULTIES:
        items = by_diff[diff]
        if not items:
            raise ValueError(f"no {diff} scenarios: difficulty_thresholds do not cover any bump and speed in scenario_gen.yaml")
        if len(items) < per_level:
            print(f"  WARNING: only {len(items)} unique {diff} single scenarios; upsampling to {per_level}")
            idxs = rng.choice(len(items), per_level, replace=True)
        else:
            idxs = rng.choice(len(items), per_level, replace=False)
        flat_starts = rng.uniform(fs_min, fs_max, size=per_level)
        for idx, fs in zip(idxs, flat_starts):
            sc = dict(items[int(idx)])
            sc["flat_start_m"] = round(float(fs), 2)
            sc["type"] = "hole" if sc["height_m"] < 0 else "bump"
            selected[diff].append(sc)
        selected[diff].sort(key=lambda c: c["difficulty_score"])

    return selected


def _write_pool(out_dir: Path, pool: list[dict]) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sc in enumerate(pool, start=1):
        (out_dir / f"scenario_{i:03d}.yaml").write_text(
            yaml.dump(sc, default_flow_style=False, sort_keys=False)
        )
    return len(pool)


def generate(
    per_level: int | None = None,
    flat_start_range: list[float] | None = None,
    out_dir: Path = _DEFAULT_OUT,
    seed: int | None = None,
    mode: str | None = None,
) -> dict[str, int]:
    if per_level is None:
        per_level = int(_gcfg["per_level"])
    if flat_start_range is None:
        flat_start_range = [float(x) for x in _gcfg["flat_start_range"]]
    if seed is None:
        seed = _load_training_seed()
    if mode is None:
        mode = _MODE_DEFAULT
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    fs_min, fs_max = float(flat_start_range[0]), float(flat_start_range[1])
    rng = np.random.default_rng(seed)
    catalog, rank = _load_catalog(mode, rng)

    single_by_diff = _gen_single(catalog, rank, per_level, fs_min, fs_max, rng)

    curriculum_dir = out_dir / "curriculum"

    counts: dict[str, int] = {}
    for diff in _DIFFICULTIES:
        counts[diff] = _write_pool(curriculum_dir / "single" / diff, single_by_diff[diff])

    manifest = {
        "obstacle_mode":         mode,
        "v_range_kmh":           [_V_MIN_KMH, _V_MAX_KMH],
        "flat_start_m_range":    [fs_min, fs_max],
        "per_level":             per_level,
        "difficulty_thresholds": {
            name: {"score_min": lo, "score_max": hi}
            for name, lo, hi in _THRESHOLDS
        },
        "obstacle_rank": rank,
        "counts":        counts,
    }
    (out_dir / "manifest.yaml").write_text(
        yaml.dump(manifest, default_flow_style=False, sort_keys=False)
    )
    return counts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate curriculum training scenarios for CALM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT,
                   help="Output directory")
    p.add_argument("--seed", type=int, default=None,
                   help=f"RNG seed (default from algo_configs.yaml training.seed: {_load_training_seed()})")
    p.add_argument("--mode", choices=_MODES, default=None,
                   help=f"Obstacle mix (default from scenario_gen.yaml obstacle_mode: {_MODE_DEFAULT})")
    args = p.parse_args()

    print(f"Generating scenarios -> {args.out_dir}")
    counts = generate(out_dir=args.out_dir, seed=args.seed, mode=args.mode)

    for diff in _DIFFICULTIES:
        print(f"  curriculum/{diff:8s}: {counts[diff]:3d} scenarios")
    print(f"Manifest -> {args.out_dir / 'manifest.yaml'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
