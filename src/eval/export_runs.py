from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from utils.paths import ROOT

SUMMARY = ['run', 'group', 'seed', 'target', 'subset', 'n', 'a_w', 'speed_kmh', 'time_s',
           'vs_human_pct', 'lower_than_human', 'vs_const_pct']
PER_SCENARIO = ['run', 'scenario', 'source', 'split', 'a_w', 'vdv', 'mean_speed_kmh', 'travel_time_s', 'a_w_vs_const_pct', 'heldout']


def _rows(path: Path) -> dict:
    by = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            if r['suite'] == 'real_all':
                by[r['scenario']][r['source']] = r
    return by


def _summary(run, meta, source, by, names, subset):
    names = [n for n in names if source in by[n] and 'human' in by[n]]
    if not names:
        return None
    g = lambda s, k: np.array([float(by[n][s][k]) for n in names])
    h, v = g('human', 'a_w'), g(source, 'a_w')
    return {'run': run, 'group': meta.get('group', ''), 'seed': meta.get('seed', ''), 'target': meta.get('target', ''),
            'subset': subset, 'n': len(names), 'a_w': round(float(np.median(v)), 4),
            'speed_kmh': round(float(np.median(g(source, 'mean_speed_kmh'))), 2),
            'time_s': round(float(np.median(g(source, 'travel_time_s'))), 2),
            'vs_human_pct': round(float(np.median(100 * (v - h) / h)), 1), 'lower_than_human': int((v < h).sum()),
            'vs_const_pct': round(float(np.median(g(source, 'a_w_vs_const_pct'))), 2)}


def _geometry(names):
    rows = []
    for n in names:
        for d in ('real', 'real_train'):
            f = ROOT / 'config' / 'eval' / 'scenarios' / d / f'{n}.json'
            if f.is_file():
                b = json.loads(f.read_text())['bumps'][0]
                rows.append({'scenario': n, 'location': n.rsplit('_sb', 1)[0], 'width_m': b['width_m'],
                             'height_mm': round(1000.0 * b['peak_z_m'], 1), 'split': 'unseen' if d == 'real' else 'seen'})
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description='collect the per-run comfort scores for the MATLAB figures')
    ap.add_argument('--runs', default=str(ROOT / 'config' / 'eval' / 'runs.yaml'))
    ap.add_argument('--runs-dir', default=str(ROOT / 'logs' / 'comfort' / 'runs'))
    ap.add_argument('--reference', default='s69', help='run whose human, constant and MPC rows are used')
    ap.add_argument('--out-dir', default=str(ROOT / 'logs' / 'comfort'))
    args = ap.parse_args()

    runs = yaml.safe_load(Path(args.runs).read_text())['runs']
    rdir, out = Path(args.runs_dir), Path(args.out_dir)
    ref = _rows(rdir / f'{runs[args.reference]["csv"]}.csv')
    names = sorted(ref)
    split = {n: ref[n]['human']['split'] for n in names}
    summary, per = [], []
    for src in ('human', 'const', 'mpc'):
        for subset, sel in (('all', names), ('unseen', [n for n in names if split[n] == 'unseen'])):
            s = _summary(src, {'group': 'reference'}, src, ref, sel, subset)
            if s:
                summary.append(s)
        per += [{'run': src, 'scenario': n, 'source': src, 'split': split[n],
                 **{k: ref[n][src][k] for k in ('a_w', 'vdv', 'mean_speed_kmh', 'travel_time_s', 'a_w_vs_const_pct')}}
                for n in names if src in ref[n]]
    for row in per:
        row['heldout'] = ''
    for run, meta in runs.items():
        path = rdir / f'{meta["csv"]}.csv'
        if not path.is_file():
            continue
        by = _rows(path)
        for n in names:
            if 'human' not in by[n]:
                by[n]['human'] = ref[n]['human']
        subsets = [('all', names), ('unseen', [n for n in names if split[n] == 'unseen'])]
        if meta.get('heldout'):
            subsets.append(('heldout', [n for n in names if n.rsplit('_sb', 1)[0] in meta['heldout']]))
        for subset, sel in subsets:
            s = _summary(run, meta, 'ppo', by, sel, subset)
            if s:
                summary.append(s)
        per += [{'run': run, 'scenario': n, 'source': 'ppo', 'split': split[n],
                 **{k: by[n]['ppo'][k] for k in ('a_w', 'vdv', 'mean_speed_kmh', 'travel_time_s', 'a_w_vs_const_pct')},
                 'heldout': int(n.rsplit('_sb', 1)[0] in meta['heldout']) if meta.get('heldout') else ''}
                for n in names if 'ppo' in by[n]]
    out.mkdir(parents=True, exist_ok=True)
    for fname, fields, rows in (('ppo_runs.csv', SUMMARY, summary), ('ppo_runs_scenarios.csv', PER_SCENARIO, per),
                                ('scenarios.csv', ['scenario', 'location', 'width_m', 'height_mm', 'split'], _geometry(names))):
        with (out / fname).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f'{len(rows)} rows -> {out / fname}')


if __name__ == '__main__':
    main()
