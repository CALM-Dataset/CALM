from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gym_env.common.env_params import DT
from eval.runners import mpc_runner, ppo_runner
from utils.dataset import build_index, index_path, load_index, real_files, region_of
from utils.scenario_loader import load_scenario_any


def _write_window_profile(win_dir: Path, kind: str, vehicle: str,
                          speeds: list[float], comfort_score: float, rms_accel: float) -> None:
    profile = {'vehicle': vehicle, 'samples': _samples(speeds),
              'metrics': {'rms_a': round(rms_accel, 4), 'comfort_score': round(comfort_score, 4)}}
    out = win_dir / 'comfort_profiles'
    out.mkdir(exist_ok=True)
    (out / f'{kind}.json').write_text(json.dumps(profile, indent=2))


def _samples(speeds: list[float]) -> list[dict]:
    v = np.asarray(speeds, dtype=float)
    s = np.concatenate([[0.0], np.cumsum(v[:-1] * DT)]) if len(v) else v
    return [{'t_s': round(k * DT, 4), 's_m': round(float(s[k]), 4),
             'v_kmh': round(float(v[k]) * 3.6, 3)} for k in range(len(v))]


def _min_kmh(speeds: list[float]) -> float:
    return round(float(min(speeds)) * 3.6, 3) if speeds else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ppo-model', default=None)
    ap.add_argument('--mpc', action='store_true')
    ap.add_argument('--vehicle', default='half')
    ap.add_argument('--horizon', type=int, default=None, help='MPC horizon in steps (default: N in config/baseline/mpc.yaml)')
    ap.add_argument('--perception', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--push', action='store_true')
    ap.add_argument('--calm-root', action='append', default=[],
                    help="CALM-Italy/CALM-Germany local sequences root - write "
                         "comfort_profiles/ppo.json and mpc.json into the matching "
                         "window folder (repeatable)")
    args = ap.parse_args()

    if not args.ppo_model and not args.mpc:
        raise SystemExit("give --ppo-model and/or --mpc")

    ppo = ppo_runner(args.ppo_model, args.seed) if args.ppo_model else None
    if args.mpc:
        from baseline.mpc.config import horizon

        mpc = mpc_runner(args.vehicle, args.horizon or horizon(), args.perception, args.seed)
    else:
        mpc = None

    from eval.comfort_score import find_window

    roots = [Path(r) for r in args.calm_root]
    files = {f.stem: f for f, _ in real_files()}

    out_dir = index_path().parent / 'labeled' / 'real'
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {r['id']: r for r in load_index()}

    real_ids = [r['id'] for r in load_index() if r['group'] == 'real']
    for i, sid in enumerate(real_ids, 1):
        src = files.get(sid)
        if src is None:
            continue
        scenario = json.loads(src.read_text())
        road, track, _, _ = load_scenario_any(sid)

        win_dir = find_window(sid, roots) if roots else None
        if roots and win_dir is None:
            print(f"  WARNING: no window match for {sid}")

        comfort: dict = {'vehicle': args.vehicle, 'recommended_speed': {}, 'metrics': {}}
        row = index.get(sid, {})
        if ppo is not None:
            speeds, cs, rms = ppo(road, track)
            comfort['recommended_speed']['ppo'] = {'samples': _samples(speeds)}
            comfort['metrics']['ppo'] = {'rms_a': round(rms, 4), 'comfort_score': round(cs, 4)}
            row['rec_speed_ppo_kmh'] = _min_kmh(speeds)
            if win_dir is not None:
                _write_window_profile(win_dir, 'ppo', args.vehicle, speeds, cs, rms)
        if mpc is not None:
            speeds, cs, rms = mpc(road, track)
            comfort['recommended_speed']['mpc'] = {'samples': _samples(speeds)}
            comfort['metrics']['mpc'] = {'rms_a': round(rms, 4), 'comfort_score': round(cs, 4)}
            row['rec_speed_mpc_kmh'] = _min_kmh(speeds)
            if win_dir is not None:
                _write_window_profile(win_dir, 'mpc', args.vehicle, speeds, cs, rms)

        scenario['comfort'] = comfort
        (out_dir / f'{sid}.json').write_text(json.dumps(scenario, indent=2))
        index[sid] = row
        print(f"[{i}/{len(real_ids)}] {sid}")

    build_index(write=True)
    merged = load_index()
    by_id = {r['id']: r for r in merged}
    for sid, row in index.items():
        if sid in by_id:
            by_id[sid].update({k: row[k] for k in
                               ('rec_speed_ppo_kmh', 'rec_speed_mpc_kmh') if k in row})
    index_path().write_text(json.dumps(list(by_id.values()), indent=2))
    print(f"labels written to {out_dir}")

    if args.push:
        from huggingface_hub import upload_file

        for f in out_dir.glob('*.json'):
            sc = json.loads(f.read_text())
            reg = region_of(sc.get('name', f.stem))
            upload_file(path_or_fileobj=str(f), path_in_repo=f'{reg}/labeled/{f.name}',
                       repo_id='CALM-Dataset/CALM-Scenarios', repo_type='dataset',
                       commit_message='Add comfort labels')
        print("pushed labels to CALM-Dataset/CALM-Scenarios")


if __name__ == '__main__':
    main()
