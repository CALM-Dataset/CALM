from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from baseline.mpc.config import horizon
from eval.comfort_score import _metrics, end_point, find_window, human_speeds, replay
from gym_env.common.env_params import DT
from gym_env.half_car.core.ode_model import HalfCarODE
from utils.scenario_loader import enabled_eval_suites, load_scenario_any

FIELDS = ['scenario', 'track_scale', 'cruise_kmh', 'travel_time_s', 'a_w']


def _span(name: str, roots: list[Path], wheelbase: float) -> float:
    road, _, _, _ = load_scenario_any(name)
    end = end_point(road, wheelbase)
    win = find_window(name, roots)
    if win is None:
        return end
    h, n_rec = human_speeds(win, end)
    return min(end, float(np.sum(h[:n_rec]) * DT))


def build(args) -> None:
    import baseline.mpc.comfort_map as comfort_map
    from eval.runners import mpc_runner

    roots = [Path(r) for r in [*args.real_root, *[p for p in os.environ.get('CALM_SEQUENCES', '').split(':') if p]]]
    names = [n for _, ns in enabled_eval_suites(args.suites) for n in ns]
    wb = HalfCarODE().wheelbase
    span = {n: _span(n, roots, wb) for n in names}
    cruise = [float(v) for v in args.cruise_kmh.split(',')] if args.cruise_kmh else []
    runs = [(float(s), 0.0) for s in args.scales.split(',')] + [(args.cruise_scale, k) for k in cruise]
    rows = []
    for scale, kmh in runs:
        comfort_map._TRACK_SCALE = scale
        run = mpc_runner('half', args.horizon or horizon(), False, 0)
        for name in names:
            road, track, _, _ = load_scenario_any(name)
            if kmh:
                road.set_speed(kmh / 3.6)
            speeds = run(road, track)[0]
            r, _, _, _ = load_scenario_any(name)
            if kmh:
                r.set_speed(kmh / 3.6)
            z_b, phi, s_pos = replay(r, speeds, float(road.speed))
            m = _metrics(z_b, phi, s_pos, speeds, span[name])
            rows.append({'scenario': name, 'track_scale': scale, 'cruise_kmh': kmh,
                         'travel_time_s': m['travel_time_s'], 'a_w': m['a_w']})
        print(f'track_scale {scale} cruise {kmh or "scenario"}: {len(names)} scenarios', flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} rows -> {out}')


def _envelope(points: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    t_env, a_env, best = [], [], np.inf
    for t, a in sorted(points):
        if a < best - 1e-9:
            t_env.append(t)
            a_env.append(a)
            best = a
    return np.array(t_env), np.array(a_env)


def excess(args) -> None:
    frontier = defaultdict(list)
    with open(args.frontier) as f:
        for r in csv.DictReader(f):
            frontier[r['scenario']].append((float(r['travel_time_s']), float(r['a_w'])))
    env = {n: _envelope(p) for n, p in frontier.items()}
    by_source = defaultdict(list)
    outside = defaultdict(int)
    with open(args.scores) as f:
        for r in csv.DictReader(f):
            if r['scenario'] not in env:
                continue
            t_env, a_env = env[r['scenario']]
            t, a = float(r['travel_time_s']), float(r['a_w'])
            if t < t_env.min() or t > t_env.max():
                outside[r['source']] += 1
                continue
            by_source[r['source']].append(100.0 * (a / float(np.interp(t, t_env, a_env)) - 1.0))
    print('a_w above the MPC frontier at equal travel time (0 = on the frontier, positive = worse)')
    print('points outside the frontier time range are left out of the statistics')
    print(f"{'source':<8}{'n':>5}{'median':>9}{'p25':>8}{'p75':>8}{'on or below':>13}{'outside time range':>20}")
    for src in ('human', 'const', 'mpc', 'ppo'):
        v = np.array(by_source.get(src, []))
        if v.size == 0 and outside[src]:
            print(f"{src:<8}{0:>5}   all {outside[src]} points lie outside the frontier time range")
        if v.size:
            print(f"{src:<8}{v.size:>5}{np.median(v):>+8.1f}%{np.percentile(v, 25):>+7.1f}%{np.percentile(v, 75):>+7.1f}%"
                  f"{int((v <= 1.0).sum()):>9}/{v.size:<3}{outside[src]:>14}")


def main() -> None:
    ap = argparse.ArgumentParser(description='MPC comfort-versus-time frontier and the distance of each profile to it')
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build', help='run the MPC at several comfort weights and cruise speeds')
    b.add_argument('--scales', default='0.02,0.1,0.5,1.0,3.0', help='comfort_map track_scale values')
    b.add_argument('--cruise-kmh', default='8,10,13,16,19', help='extra scenario speeds that give slower operating points')
    b.add_argument('--cruise-scale', type=float, default=0.1, help='track_scale used for the cruise-speed runs')
    b.add_argument('--suites', nargs='+', default=['real_all'])
    b.add_argument('--horizon', type=int, default=None, help='MPC horizon in steps (default: N in config/baseline/mpc.yaml)')
    b.add_argument('--real-root', action='append', default=[])
    b.add_argument('--out', default='logs/comfort/mpc_frontier.csv')
    b.set_defaults(fn=build)
    e = sub.add_parser('excess', help='compare comfort_scores.csv against a built frontier')
    e.add_argument('--scores', default='logs/comfort/comfort_scores.csv')
    e.add_argument('--frontier', default='logs/comfort/mpc_frontier.csv')
    e.set_defaults(fn=excess)
    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
