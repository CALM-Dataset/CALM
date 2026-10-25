from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import savgol_filter

from eval.iso2631 import iso_band, rms, vdv, weight_wk
from gym_env.common.env_params import DT
from gym_env.half_car.core.ode_model import HalfCarODE
from utils.paths import CONFIG
from utils.scenario_loader import enabled_eval_suites, list_real_scenarios, load_scenario_any

FIELDS = ['suite', 'scenario', 'source', 'a_w', 'vdv', 'pitch_rms', 'peak_a_w',
          'mean_speed_kmh', 'min_speed_kmh', 'travel_time_s', 'completed',
          'band', 'a_w_imu', 'padded_frac', 'a_w_vs_const_pct', 'split']
SMOOTH_S = 1.0
FRONTIER_KMH = tuple(float(v) for v in range(8, 31, 2))
_CALIBRATION = {'enabled': True}


def scoring_params() -> dict | None:
    if not _CALIBRATION['enabled']:
        return None
    from gym_env.half_car.core import ode_model as om

    cfg = CONFIG / 'eval' / 'comfort.yaml'
    sv = (yaml.safe_load(cfg.read_text()) or {}).get('scoring_vehicle', {}) if cfg.is_file() else {}
    ks, dd = float(sv.get('k_S_scale', 1.0)), float(sv.get('D_scale', 1.0))
    return {axle: {'k_S': base['k_S'] * ks, 'D': base['D'] * dd}
            for axle, base in (('front', om.PHYSICS_F), ('rear', om.PHYSICS_R))}


def end_point(road, wheelbase: float) -> float:
    prof = getattr(road, '_profile', None)
    if prof is not None and len(prof[0]):
        return float(prof[0][-1]) + wheelbase + 3.0
    bumps = getattr(road, '_bumps', None)
    if bumps:
        return float(bumps[-1][0] + bumps[-1][2]) + wheelbase
    return 0.0


def replay(road, speeds, v_start: float):
    ode = HalfCarODE(scoring_params())
    x = ode.reset(float(v_start))
    s = 0.0
    v_prev = float(v_start)
    n = len(speeds)
    z_b = np.empty(n)
    phi = np.empty(n)
    s_pos = np.empty(n)
    for k, v in enumerate(speeds):
        v = float(v)
        a = (v - v_prev) / DT
        v_prev = v
        x[8] = v
        road.set_speed(v)
        s_start = s
        s += v * DT
        x, z_b[k], phi[k], _, _ = ode.step(x, road, s_start, road.speed, a)
        s_pos[k] = s
    return z_b, phi, s_pos


def _metrics(z_b, phi, s_pos, speeds, end: float) -> dict:
    reached = np.nonzero(s_pos >= end)[0]
    completed = bool(reached.size)
    k = int(reached[0]) + 1 if completed else len(s_pos)
    a_w = weight_wk(z_b[:k], DT)
    v = np.asarray(speeds[:k], float)
    t = k * DT
    return {
        'a_w': round(rms(a_w), 4),
        'vdv': round(vdv(a_w, DT), 4),
        'pitch_rms': round(rms(phi[:k]), 4),
        'peak_a_w': round(float(np.max(np.abs(a_w))), 4) if k else 0.0,
        'mean_speed_kmh': round(float(s_pos[k - 1]) / t * 3.6, 3) if k else 0.0,
        'min_speed_kmh': round(float(v.min()) * 3.6, 3) if k else 0.0,
        'travel_time_s': round(t, 3),
        'completed': completed,
        'band': iso_band(rms(a_w)),
    }


def const_speeds(road, end: float) -> list[float]:
    v = float(road.speed)
    return [v] * (int(np.ceil(end / (v * DT))) + 1)


def _load_window(win_dir: Path):
    wheel = json.loads((win_dir / 'export' / 'wheel_excitement.json').read_text())
    rec = wheel['wheel_results']['cariad_wheel_FL']
    arr = np.asarray(rec['wheel_excitement'], float)
    ts = np.array([p[0] for p in rec['contact_point_trajectory']['poses']], float)
    return ts, np.interp(ts, arr[:, 0], arr[:, 2])


def human_replay(win_dir: Path, end: float, offset: float = 0.0) -> tuple[list[float], int, float]:
    ts, s_ref = _load_window(win_dir)
    t = (ts - ts[0]) / 1e6
    s = s_ref - s_ref[0]
    grid = np.arange(0.0, t[-1], DT)
    s_grid = np.interp(grid, t, s)
    win = int(round(SMOOTH_S / DT)) | 1
    v_rec = np.clip(savgol_filter(s_grid, win, 3, deriv=1, delta=DT), 0.0, None)
    t_start = 0.0
    if offset > 0.0:
        monotone = np.maximum.accumulate(s_grid)
        if offset >= monotone[-1]:
            raise ValueError(f'{win_dir.name}: the recorded drive ends {monotone[-1]:.1f} m in, before the road starts at {offset:.1f} m')
        t_start = float(np.interp(offset, monotone, grid))
    n_rec = int(np.floor((grid[-1] - t_start) / DT)) + 1
    v = list(np.interp(t_start + np.arange(n_rec) * DT, grid, v_rec))
    tail = max(v[-1], 0.5)
    travelled = float(np.sum(v) * DT)
    while travelled < end:
        v.append(tail)
        travelled += tail * DT
    return v, n_rec, t_start


def human_speeds(win_dir: Path, end: float) -> tuple[list[float], int]:
    v, n_rec, _ = human_replay(win_dir, end)
    return v, n_rec


def imu_a_w(win_dir: Path, t_end_s: float, t_start_s: float = 0.0) -> float | None:
    f = win_dir / 'export' / 'output.laz.sensor_export.json'
    if not f.is_file():
        return None
    imu = json.loads(f.read_text()).get('imu') or []
    if len(imu) < 20:
        return None
    ts = np.array([r['timestamp'] for r in imu], float)
    az = np.array([r['accel_z'] for r in imu], float)
    t0, _ = _load_window(win_dir)
    keep = (ts >= t0[0] + t_start_s * 1e6) & (ts <= t0[0] + t_end_s * 1e6)
    if keep.sum() < 20:
        return None
    dt_imu = float(np.median(np.diff(ts[keep]))) / 1e6
    return round(rms(weight_wk(az[keep] - np.mean(az[keep]), dt_imu)), 4)


def find_window(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        for p in root.glob(f'*/{name}'):
            if (p / 'export' / 'wheel_excitement.json').is_file():
                return p
    return None


def write_profile(out_dir: Path, suite: str, name: str, source: str, speeds, z_b, s_pos, cut: float) -> None:
    reached = np.nonzero(s_pos >= cut)[0]
    k = int(reached[0]) + 1 if reached.size else len(s_pos)
    a_w = weight_wk(z_b[:k], DT)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f'{suite}__{name}__{source}.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_s', 's_m', 'v_kmh', 'a_body', 'a_w'])
        for i in range(k):
            w.writerow([round(i * DT, 3), round(float(s_pos[i]), 4), round(float(speeds[i]) * 3.6, 3),
                        round(float(z_b[i]), 4), round(float(a_w[i]), 4)])


def const_frontier(road_name: str, end: float, cut: float):
    pts = []
    for kmh in FRONTIER_KMH:
        r, _, _, _ = load_scenario_any(road_name)
        v = kmh / 3.6
        speeds = [v] * (int(np.ceil((end + 5.0) / (v * DT))) + 1)
        z_b, phi, s_pos = replay(r, speeds, v)
        m = _metrics(z_b, phi, s_pos, speeds, cut)
        pts.append((m['travel_time_s'], m['a_w']))
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def _SEEN() -> set:
    return set(list_real_scenarios('real_train'))


def _UNSEEN() -> set:
    return set(list_real_scenarios('real'))


def write_road(out_dir: Path, suite: str, name: str, road) -> None:
    prof = getattr(road, '_profile', None)
    if prof is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f'road__{suite}__{name}.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['s_m', 'z_mm'])
        for x, z in zip(prof[0], prof[1]):
            w.writerow([round(float(x), 3), round(float(z) * 1000.0, 2)])


def score_scenario(suite: str, name: str, runners: dict, roots: list[Path],
                   prof_dir: Path | None = None) -> list[dict]:
    road, track, _, _ = load_scenario_any(name)
    end = end_point(road, HalfCarODE().wheelbase)
    v_road = float(road.speed)
    rows: list[dict] = []

    if prof_dir is not None:
        write_road(prof_dir, suite, name, road)
    win = find_window(name, roots) if suite.startswith('real') else None
    offset = float(getattr(road, '_trim_offset_m', 0.0))
    human = human_replay(win, end, offset) if win is not None else None
    cut = end
    if human is not None:
        cut = min(end, float(np.sum(human[0][:human[1]]) * DT))

    frontier = const_frontier(name, end, cut)
    split = 'seen' if name in _SEEN() else ('unseen' if name in _UNSEEN() else '')

    def add(source: str, speeds, v_start: float, extra: dict | None = None):
        r, _, _, _ = load_scenario_any(name)
        z_b, phi, s_pos = replay(r, speeds, v_start)
        if prof_dir is not None:
            write_profile(prof_dir, suite, name, source, speeds, z_b, s_pos, cut)
        row = {'suite': suite, 'scenario': name, 'source': source,
               **_metrics(z_b, phi, s_pos, speeds, cut), 'a_w_imu': '', 'padded_frac': ''}
        ref = float(np.interp(row['travel_time_s'], *frontier))
        row['split'] = split
        row['a_w_vs_const_pct'] = round(100.0 * (row['a_w'] / ref - 1.0), 1) if ref > 0 else ''
        row.update(extra or {})
        rows.append(row)

    add('const', const_speeds(road, end), v_road)

    if human is not None:
        h, n_rec, t_start = human
        add('human', h, h[0], {'a_w_imu': imu_a_w(win, t_start + n_rec * DT, t_start) or '',
                               'padded_frac': round(1.0 - n_rec / len(h), 3)})

    for source in ('ppo', 'mpc'):
        run = runners.get(source)
        if run is None:
            continue
        r, tr, _, _ = load_scenario_any(name)
        speeds = run(r, tr)[0]
        add(source, speeds, v_road)
    return rows


def _runners(args) -> dict:
    from eval.runners import mpc_runner, ppo_runner

    out = {}
    if args.ppo_model:
        out['ppo'] = ppo_runner(args.ppo_model, args.seed)
    if args.mpc:
        from baseline.mpc.config import horizon

        out['mpc'] = mpc_runner('half', args.horizon or horizon(), False, args.seed)
    return out


def _all_suites() -> list[str]:
    root = CONFIG / 'eval' / 'scenarios'
    return sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('real') and 'pothole' not in d.name) + ['real']


def _med(v):
    v = [x for x in v if x != '']
    return float(np.median(v)) if v else float('nan')


def print_summary(rows: list[dict]) -> None:
    rows = [{**r, 'suite': f"{r['suite']}[{r['split']}]"} if r['suite'] == 'real_all' and r.get('split') else r for r in rows]
    order = ('human', 'const', 'mpc', 'ppo')
    suites = list(dict.fromkeys(r['suite'] for r in rows))
    print(f"\n{'suite':<24}{'source':<7}{'n':>4}{'a_w':>8}{'VDV':>8}{'km/h':>7}{'time_s':>8}{'vs_const%':>10}{'done':>6}")
    for suite in suites:
        for src in order:
            sel = [r for r in rows if r['suite'] == suite and r['source'] == src]
            if not sel:
                continue
            print(f"{suite:<24}{src:<7}{len(sel):>4}{_med([r['a_w'] for r in sel]):>8.3f}"
                  f"{_med([r['vdv'] for r in sel]):>8.3f}{_med([r['mean_speed_kmh'] for r in sel]):>7.1f}"
                  f"{_med([r['travel_time_s'] for r in sel]):>8.1f}{_med([r['a_w_vs_const_pct'] for r in sel]):>10.0f}{sum(r['completed'] for r in sel):>6}")
    by: dict = {}
    for r in rows:
        if r['suite'].startswith('real'):
            by.setdefault(r['scenario'], {})[r['source']] = r
    pairs = [d for d in by.values() if 'human' in d]
    if not pairs:
        return
    print(f"\nagainst human on {len(pairs)} real scenarios (lower a_w wins)")
    for src in ('mpc', 'ppo', 'const'):
        both = [d for d in pairs if src in d]
        wins = sum(d[src]['a_w'] < d['human']['a_w'] for d in both)
        faster = sum(d[src]['travel_time_s'] <= d['human']['travel_time_s'] for d in both)
        print(f"  {src:<6} lower a_w {wins}/{len(both)}   not slower {faster}/{len(both)}")
    ref = [d for d in by.values() if 'mpc' in d]
    if ref:
        print(f"\nagainst the MPC reference on {len(ref)} real scenarios (median per-scenario change)")
        for src in ('ppo', 'human', 'const'):
            both = [d for d in ref if src in d]
            if both:
                da = [100 * (d[src]['a_w'] / d['mpc']['a_w'] - 1) for d in both]
                dt = [100 * (d[src]['travel_time_s'] / d['mpc']['travel_time_s'] - 1) for d in both]
                print(f"  {src:<6} a_w {_med(da):+5.0f}%   travel time {_med(dt):+5.0f}%")
    imu = [d['human'] for d in pairs if d['human']['a_w_imu'] != '']
    if imu:
        err = [abs(r['a_w'] - r['a_w_imu']) / r['a_w_imu'] for r in imu]
        pad = [d['human']['padded_frac'] for d in pairs]
        print(f"human replay vs IMU a_w: median error {100 * _med(err):.0f}%, "
              f"{sum(e <= 0.3 for e in err)}/{len(err)} within 30%; median padded share {100 * _med(pad):.0f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description='ISO 2631-1 comfort score per scenario and speed profile')
    ap.add_argument('--ppo-model', default=None)
    ap.add_argument('--mpc', action='store_true')
    ap.add_argument('--horizon', type=int, default=None, help='MPC horizon in steps (default: N in config/baseline/mpc.yaml)')
    ap.add_argument('--suites', nargs='+', default=None,
                    help='suite folders, "real" (held-out), "real_train" (seen in training) or "real_all" (default: every synthetic suite and real)')
    ap.add_argument('--real-root', action='append', default=[],
                    help='sequences root holding the recorded windows (repeatable; also CALM_SEQUENCES, colon-separated)')
    ap.add_argument('--uncalibrated', action='store_true',
                    help='score on the nominal vehicle instead of the IMU-calibrated one')
    ap.add_argument('--allow-missing-human', action='store_true',
                    help='score real scenarios without the recorded drive when the recordings are not found')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--profiles-dir', default=None,
                    help='write per-scenario speed and acceleration time series for plotting')
    ap.add_argument('--out', default='logs/comfort/comfort_scores.csv')
    args = ap.parse_args()

    _CALIBRATION['enabled'] = not args.uncalibrated
    runners = _runners(args)
    env_roots = [r for r in os.environ.get('CALM_SEQUENCES', '').split(':') if r]
    roots = [Path(r) for r in [*args.real_root, *env_roots]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    prof_dir = Path(args.profiles_dir) if args.profiles_dir else None
    rows: list[dict] = []
    for suite, names in enabled_eval_suites(args.suites or _all_suites()):
        for i, name in enumerate(names, 1):
            rows += score_scenario(suite, name, runners, roots, prof_dir)
            print(f'{suite} {i}/{len(names)} {name}', flush=True)

    scored = {(r['suite'], r['scenario']) for r in rows}
    with_human = {(r['suite'], r['scenario']) for r in rows if r['source'] == 'human'}
    missing = sorted(n for suite, n in scored - with_human if suite.startswith('real'))
    if missing and not args.allow_missing_human:
        raise SystemExit(f'recorded drive not found for {len(missing)} real scenarios '
                         f'(set CALM_SEQUENCES or --real-root, or pass --allow-missing-human): '
                         + ', '.join(missing[:4]) + (' ...' if len(missing) > 4 else ''))

    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} rows -> {out}')
    print_summary(rows)


if __name__ == '__main__':
    main()
