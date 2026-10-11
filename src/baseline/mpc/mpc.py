from __future__ import annotations

import os

from pathlib import Path as _Path

_repo_acados = _Path(__file__).resolve().parents[3] / 'acados'
_acados = _repo_acados if (_repo_acados / 'lib').is_dir() else _Path(os.environ.get('ACADOS_SOURCE_DIR') or _repo_acados)
os.environ['ACADOS_SOURCE_DIR'] = str(_acados)
_lib = str(_acados / 'lib')
if _lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import argparse
import copy
import gc
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_ROOT = Path(__file__).resolve().parents[3]
_acados_py = str(_ROOT / 'acados' / 'interfaces' / 'acados_template')
if _acados_py not in sys.path:
    sys.path.insert(0, _acados_py)

__import__("gym_env.quarter_car.envs")
from gym_env.quarter_car.config.env_params import PHYSICS, DT, EPISODE_STEPS
from gym_env.quarter_car.config.reward_params import load_reward_config
from gym_env.quarter_car.wrappers import TerminationWrapper
from gym_env.quarter_car.reward.utils import reward_bounds
from baseline.mpc.controller import MPCController
from utils.scenario_loader import load_scenario_any, enabled_eval_suites
from eval.data_logger import RunLogger, EpisodeData


def _load_mpc_cfg() -> dict:
    p = _ROOT / 'config' / 'baseline' / 'mpc.yaml'
    with open(p) as fh:
        return yaml.safe_load(fh)


_KMH_FIELDS = ('v_limit', 'v_min', 'v_hard_min', 'v_init_low')


def mpc_reward_config(load_reward_config):
    import dataclasses

    mcfg = _load_mpc_cfg()
    base = load_reward_config()
    names = {f.name for f in dataclasses.fields(base)}
    over = {k: float(v) for k, v in (mcfg.get('weights') or {}).items() if k in names}
    for k, v in (mcfg.get('velocity') or {}).items():
        if k in names:
            over[k] = float(v) / 3.6 if k in _KMH_FIELDS else float(v)
    return dataclasses.replace(base, **over)


def make_env(bits: dict, cfg, render_mode: str = 'none'):
    return bits['TerminationWrapper'](gym.make(bits['env_id'], render_mode=render_mode,
                                               reward_config=cfg, **bits.get('env_kwargs', {})))


def _vehicle_bits(vehicle: str) -> dict:
    if vehicle == 'half':
        __import__("gym_env.half_car.envs")
        from gym_env.half_car.config.env_params import PHYSICS as PH, EPISODE_STEPS as ES
        from gym_env.half_car.config.reward_params import load_reward_config as lrc
        from gym_env.half_car.wrappers import TerminationWrapper as TW
        from gym_env.half_car.reward.utils import reward_bounds as rb
        from gym_env.half_car.envs.render import wait_for_episode_nav as nav
        from gym_env.half_car.config.preview_params import load_preview_config as lpc
        return dict(env_id='gym_env/HalfCar', env_kwargs={'action_mode': 'acceleration'}, PHYSICS=PH, EPISODE_STEPS=ES,
                    load_reward_config=lambda: mpc_reward_config(lrc), TerminationWrapper=TW,
                    reward_bounds=rb, wait_for_episode_nav=nav, load_preview_config=lpc)
    if vehicle not in ('quarter', None):
        raise ValueError(f"unknown vehicle {vehicle!r} (expected 'quarter' or 'half')")
    from gym_env.quarter_car.envs.render import wait_for_episode_nav as nav
    from gym_env.quarter_car.config.preview_params import load_preview_config as lpc
    return dict(env_id='gym_env/QuarterCar', PHYSICS=PHYSICS,
                EPISODE_STEPS=EPISODE_STEPS, load_reward_config=lambda: mpc_reward_config(load_reward_config),
                TerminationWrapper=TerminationWrapper, reward_bounds=reward_bounds,
                wait_for_episode_nav=nav, load_preview_config=lpc)


def _steps_for_road(road, dt: float = 0.02, margin: float = 1.5) -> int:
    if road is None or not hasattr(road, '_bumps') or not road._bumps:
        return 4000
    x0, _, L = road._bumps[-1]
    road_length = x0 + L + 20.0
    cfg = load_reward_config()
    v = max(float(cfg.v_min), 1.0)
    return max(int(road_length / v / dt * margin), 1200)


def run_episode(env, ctrl: MPCController, seed: int,
                scenario_road=None, collect_frames: bool = False,
                render_live: bool = False,
                perception_track=None, preview_cfg=None) -> dict:
    road_for_ep = copy.deepcopy(scenario_road) if scenario_road is not None else None

    raw = env.unwrapped
    raw._max_episode_steps = _steps_for_road(road_for_ep)

    opts     = {"road": road_for_ep} if road_for_ep is not None else {}
    obs, _   = env.reset(seed=seed, options=opts or None)

    if perception_track is not None and preview_cfg is not None:
        from baseline.perceived_road import make_perceived_road
        plan_road = make_perceived_road(raw._road, perception_track, preview_cfg)
    else:
        plan_road = raw._road
    ctrl.reset(plan_road)

    ep_return  = 0.0
    accel_sq   = 0.0
    pitch_sq   = 0.0
    n_steps    = 0
    solve_times: list[float] = []
    cpu_times:   list[float] = []
    speeds:      list[float] = []
    v_inits:     list[float] = []
    accels:      list[float] = []
    actions:     list[float] = []
    rewards:     list[float] = []
    susp_defl:   list[float] = []
    tire_defl:   list[float] = []
    frames:      list        = []
    done       = False

    while not done:
        if render_live:
            env.render()
            if getattr(raw, '_nav_request', None) is not None:
                break
        if collect_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if plan_road is not raw._road:
            plan_road.set_s_pos(raw._s_pos)
        t0, c0 = time.perf_counter(), time.process_time()
        u  = ctrl.act(raw._state.copy(), raw._s_pos, plan_road)
        solve_times.append(time.perf_counter() - t0)
        cpu_times.append(time.process_time() - c0)

        _, reward, terminated, truncated, info = env.step(
            np.array([u], dtype=np.float32)
        )
        ep_return += reward
        accel_sq  += info.get('z_B_ddot', 0.0) ** 2
        pitch_sq  += info.get('phi_ddot', 0.0) ** 2
        n_steps   += 1
        speeds.append(info.get('speed',    0.0))
        v_inits.append(info.get('v_init',  0.0))
        accels.append(info.get('z_B_ddot', 0.0))
        actions.append(float(u))
        rewards.append(float(reward))
        susp_defl.append(info.get('susp_defl', 0.0))
        tire_defl.append(info.get('tire_defl', 0.0))
        done = terminated or truncated

    rms_accel = float(np.sqrt(accel_sq / max(n_steps, 1)))
    rms_pitch = float(np.sqrt(pitch_sq / max(n_steps, 1)))
    cfg       = load_reward_config()
    return {
        'episode_return': round(float(ep_return),   2),
        'rms_accel':      round(rms_accel,           4),
        'rms_pitch_accel': round(rms_pitch,          4),
        'comfort_score':  round(max(0.0, 1.0 - rms_accel / cfg.a_limit), 4),
        'n_steps':        n_steps,
        'bumps_passed':   int(raw._bumps_passed),
        'bumps_total':    int(raw._bumps_total),
        'holes_passed':   int(raw._holes_passed),
        'holes_total':    int(raw._holes_total),
        'solve_ms_mean':  round(float(np.mean(solve_times)) * 1e3, 2),
        'solve_ms_max':   round(float(np.max(solve_times))  * 1e3, 2),
        'cpu_ms_mean':    round(float(np.mean(cpu_times))   * 1e3, 2),
        'n_params':       0.0,
        '_speeds':   speeds,
        '_v_inits':  v_inits,
        '_accels':   accels,
        '_actions':  actions,
        '_rewards':  rewards,
        '_susp':     susp_defl,
        '_tire':     tire_defl,
        '_frames':   frames,
    }


def _obst_str(r: dict) -> str:
    s = f'{r["bumps_passed"]}/{r["bumps_total"]}' if r.get('bumps_total') else '-'
    if r.get('holes_total'):
        s += f' H{r["holes_passed"]}/{r["holes_total"]}'
    return s


def _save_gif(frames: list, path: Path) -> None:
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     optimize=False, duration=20, loop=0)
        print(f'  GIF saved → {path}')
    except ImportError:
        print('  [PIL not installed - cannot save GIF; run: pip install Pillow]')


def _save_plot(result: dict, ep_i: int, save_dir: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        from gym_env.quarter_car.config.env_params import DT
    except ImportError:
        DT = 0.02

    t = np.arange(result['n_steps']) * DT

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, [v * 3.6 for v in result['_speeds']], label='speed km/h')
    axes[0].set_ylabel('speed [km/h]')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, result['_accels'], color='tab:orange', label='body accel')
    axes[1].axhline(0, color='gray', lw=0.7)
    axes[1].set_ylabel('z̈_B [m/s²]')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, result['_actions'], color='tab:green', label='action u')
    axes[2].axhline(0, color='gray', lw=0.7)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_ylabel('action [-1,1]')
    axes[2].set_xlabel('time [s]')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f'MPC - ep {ep_i+1}  return={result["episode_return"]:.1f}  '
        f'solve={result["solve_ms_mean"]:.1f}ms',
        fontsize=11,
    )
    fig.tight_layout()
    out = save_dir / f'mpc_ep{ep_i+1}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  plot saved → {out}')


def _run_suite_mpc(env, ctrl: MPCController, args, save_dir: Path,
                   log_root, cfg, terminal_out=None, preview_cfg=None) -> None:
    from datetime import datetime

    suites = enabled_eval_suites(args.suites)
    if not suites:
        print('No scenarios. Enable a suite in config/eval/eval_suites.yaml '
              '(incl. `real`), or pass --suites <folder|real>.')
        return

    half = getattr(args, 'vehicle', 'quarter') == 'half'
    p_hdr = f'  {"Pitch r/s²":>10}' if half else ''
    p_sep = f'  {"-"*10}' if half else ''
    total = sum(len(s) for _, s in suites)
    print('\n  MPC - suite run')
    print(f'  suites : {[n for n, _ in suites]}')
    print(f'  {total} scenarios\n')
    print(f'  {"Scenario":<24}  {"Return":>9}  {"RMS m/s²":>8}{p_hdr}  {"Comfort":>7}  {"Solve ms":>9}  Bumps')
    print(f'  {"-"*24}  {"-"*9}  {"-"*8}{p_sep}  {"-"*7}  {"-"*9}  {"-"*5}')

    all_returns: list[float] = []
    export_suites = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    bar = (tqdm(total=total, desc='MPC scenarios', unit='sc',
                file=terminal_out, leave=False)
           if tqdm is not None and terminal_out is not None else None)

    for suite_name, scenarios in suites:
        suite_rows = []
        print(f'\n  ── {suite_name} ──')

        for sc_name in scenarios:
            road, _track, _v, _desc = load_scenario_any(sc_name)
            r = run_episode(env, ctrl, seed=args.seed, scenario_road=road,
                            collect_frames=False, render_live=False,
                            perception_track=_track, preview_cfg=preview_cfg)
            bumps_str = _obst_str(r)
            p_col = f'  {r.get("rms_pitch_accel", 0.0):>10.3f}' if half else ''
            print(f'  {sc_name:<24}  {r["episode_return"]:>+9.1f}'
                  f'  {r["rms_accel"]:>8.3f}{p_col}  {r["comfort_score"]:>7.3f}'
                  f'  {r["solve_ms_mean"]:>9.2f}  {bumps_str}')
            all_returns.append(r['episode_return'])
            if bar is not None:
                bar.set_postfix({'scenario': sc_name,
                                 'return': f'{r["episode_return"]:+.0f}'},
                                refresh=False)
                bar.update(1)

            if log_root is not None:
                logger = RunLogger(
                    method='MPC', road=sc_name, out_root=log_root,
                    dt=DT, v_max_kmh=cfg.v_limit * 3.6,
                    a_comfort=cfg.a_comfort, a_limit=cfg.a_limit,
                    run_subdir=False,
                    bumps=getattr(road, '_bumps', None),
                )
                logger.add(EpisodeData(
                    v=r['_speeds'], v_init=r['_v_inits'],
                    z_B_ddot=r['_accels'], action=r['_actions'],
                    reward=r['_rewards'],
                    susp_defl=r.get('_susp'), tire_defl=r.get('_tire'),
                    episode_return=r['episode_return'],
                    rms_accel=r['rms_accel'],
                    comfort_score=r['comfort_score'],
                    solve_ms_mean=r['solve_ms_mean'],
                    cpu_ms_mean=r['cpu_ms_mean'],
                    n_params=r['n_params'],
                ))
                logger.save()

            suite_rows.append({'scenario': sc_name, **{k: v for k, v in r.items()
                                                        if not k.startswith('_')}})

        suite_mean = float(np.mean([row['episode_return'] for row in suite_rows])) if suite_rows else 0.0
        print(f'  {"Mean":<24}  {suite_mean:>+9.1f}')
        export_suites.append({'suite': suite_name, 'scenarios': suite_rows})

    if bar is not None:
        bar.close()

    if all_returns:
        print(f'\n  OVERALL  {len(all_returns)} scenarios'
              f'  mean_return={float(np.mean(all_returns)):+.1f}')

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f'suite_mpc_{ts}.json'
    out.write_text(json.dumps({'timestamp': ts, 'method': 'MPC',
                               'suites': export_suites}, indent=2))
    print(f'\n  JSON saved → {out}')
    if log_root:
        print(f'  data logged → {log_root}/MPC/')


def _render_scenarios(env, ctrl: MPCController, args, terminal_out=None,
                      nav_fn=None, preview_cfg=None) -> None:
    if nav_fn is None:
        from gym_env.quarter_car.envs.render import wait_for_episode_nav as nav_fn

    scenarios = [(name, sc) for name, scs in enabled_eval_suites(args.suites)
                 for sc in scs]
    if not scenarios:
        print('No scenarios. Enable a suite in config/eval/eval_suites.yaml '
              '(incl. `real`), or pass --suites <folder|real>.')
        return

    raw = env.unwrapped
    n   = len(scenarios)
    idx = 0
    while True:
        suite_name, sc_name = scenarios[idx]
        road, _track, _v, sc_desc = load_scenario_any(sc_name)
        raw.set_render_label(f'{sc_name} - {sc_desc}   [{idx+1}/{n} · {suite_name}]')
        r = run_episode(env, ctrl, seed=args.seed, scenario_road=road,
                        collect_frames=False, render_live=True,
                        perception_track=_track, preview_cfg=preview_cfg)
        raw = env.unwrapped
        nav = getattr(raw, '_nav_request', None)
        raw._nav_request = None
        if nav is None:
            if terminal_out is not None:
                terminal_out.write(
                    f'  [{idx+1}/{n}] {sc_name:<24}  return={r["episode_return"]:+.1f}  '
                    f'RMS={r["rms_accel"]:.3f}  obst={_obst_str(r)}\n'
                )
            nav = nav_fn(env, allow_prev=True)
        if nav == 'quit':
            break
        elif nav == 'next':
            idx = min(idx + 1, n - 1)
        elif nav == 'prev':
            idx = max(idx - 1, 0)


def main() -> None:
    output = io.StringIO()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    terminal_out = os.fdopen(os.dup(saved_stdout), 'w', buffering=1)

    try:
        with redirect_stdout(output):
            mcfg = _load_mpc_cfg()

            ap = argparse.ArgumentParser('MPC baseline (quarter- or half-car)')
            ap.add_argument('--no-perception', dest='perception', action='store_false',
                            default=True,
                            help='plan on the true road (oracle) instead of the perceived detections')
            ap.add_argument('--vehicle', default='half', choices=('quarter', 'half'),
                            help='vehicle dynamics model for the comfort-map plant')
            ap.add_argument('--horizon', type=int, default=mcfg.get('N', 50),
                            help='prediction horizon (steps)')
            ap.add_argument('--seed', type=int, default=42)
            ap.add_argument('--scenario', default=None,
                            help='fixed single scenario name or path')
            ap.add_argument('--suites', nargs='+', metavar='FOLDER', default=None,
                            help='suite folder names to run (default: enabled in eval_suites.yaml)')
            ap.add_argument('--render', action='store_true',
                            help='show live window (render_mode=human, like eval.py)')
            ap.add_argument('--save-gif', action='store_true',
                            help='save per-episode GIF (headless-safe offscreen rendering)')
            ap.add_argument('--save-plots', action='store_true',
                            help='save per-episode time-series PNG')
            ap.add_argument('--log-data', metavar='DIR', default='data',
                            help='Save run data (.mat/.npz) to DIR/MPC/<road>/<run_N>/. '
                                 'Omit DIR to use <repo>/data/.',
                            nargs='?', const='__default__')
            args = ap.parse_args()

            if args.save_gif and not args.render:
                os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

            bits = _vehicle_bits(args.vehicle)
            cfg  = bits['load_reward_config']()
            preview_cfg = bits['load_preview_config']() if args.perception else None
            ctrl = MPCController(
                cfg=cfg, physics=dict(bits['PHYSICS']), N=args.horizon,
                nlp_solver_max_iter=mcfg.get('nlp_solver_max_iter', 10),
                vehicle=args.vehicle,
            )

            if args.render:
                render_mode = 'human'
            elif args.save_gif:
                render_mode = 'rgb_array'
            else:
                render_mode = 'none'
            env  = make_env(bits, cfg, render_mode)

            scenario_label = args.scenario or 'eval_scenarios'
            save_dir = _ROOT / 'eval' / 'results' / 'mpc' / scenario_label

            log_root = (
                _ROOT / 'data'
                if args.log_data == '__default__'
                else Path(args.log_data) if args.log_data
                else None
            )

            if args.render and not args.scenario:
                _render_scenarios(env, ctrl, args, terminal_out,
                                  nav_fn=bits['wait_for_episode_nav'],
                                  preview_cfg=preview_cfg)
                env.close()
                return

            if not args.scenario:
                _run_suite_mpc(env, ctrl, args, save_dir, log_root, cfg, terminal_out,
                               preview_cfg=preview_cfg)
                env.close()
                return

            sc_road, _track, speed, sc_desc = load_scenario_any(args.scenario)
            sc_name = args.scenario
            print(f'\n  Scenario : {sc_name}')
            print(f'           : {sc_desc}')
            print(f'           : v={speed*3.6:.1f} km/h, {len(sc_road._bumps)} bump(s)')

            print('\n  MPC Baseline')
            print(f'  horizon  : {args.horizon} steps  ({args.horizon * 0.02:.2f}s)  '
                  '(config/baseline/mpc.yaml)')
            print(f'  scenarios: {scenario_label}')
            print('  solver   : acados SQP + HPIPM\n')
            w = max(len(sc_name), 8)
            half = getattr(args, 'vehicle', 'quarter') == 'half'
            p_hdr = f'  {"Pitch r/s²":>10}' if half else ''
            p_sep = f'  {"-"*10}' if half else ''
            print(f'  {"Scenario":<{w}}  {"Return":>9}  {"RMS-a m/s²":>10}{p_hdr}  '
                  f'{"Comfort":>8}  {"Bumps":>7}  {"Solve ms":>9}')
            print(f'  {"-"*w}  {"-"*9}  {"-"*10}{p_sep}  {"-"*8}  {"-"*7}  {"-"*9}')

            run_logger: RunLogger | None = (
                RunLogger(
                    method='MPC',
                    road=scenario_label,
                    out_root=log_root,
                    dt=DT,
                    v_max_kmh=cfg.v_limit * 3.6,
                    a_comfort=cfg.a_comfort,
                    a_limit=cfg.a_limit,
                    run_subdir=False,
                    bumps=getattr(sc_road, '_bumps', None),
                )
                if log_root is not None else None
            )

            if args.render:
                wait_for_episode_nav = bits['wait_for_episode_nav']
                env.unwrapped.set_render_label(f'{sc_name} - {sc_desc}')

            while True:
                r = run_episode(env, ctrl, seed=args.seed,
                                scenario_road=sc_road,
                                collect_frames=args.save_gif,
                                render_live=args.render)
                bumps_str = _obst_str(r)
                p_col = f'  {r.get("rms_pitch_accel", 0.0):>10.3f}' if half else ''
                print(f'  {sc_name:<{w}}  {r["episode_return"]:>+9.1f}  '
                      f'{r["rms_accel"]:>10.3f}{p_col}  {r["comfort_score"]:>8.3f}  '
                      f'{bumps_str:>7}  {r["solve_ms_mean"]:>9.2f}')
                nr = getattr(env.unwrapped, '_nav_request', None)
                env.unwrapped._nav_request = None
                if args.render and (nr == 'replay' or (nr is None and wait_for_episode_nav(env) == 'replay')):
                    continue
                break

            if run_logger is not None:
                run_logger.add(EpisodeData(
                    v=r['_speeds'],
                    v_init=r['_v_inits'],
                    z_B_ddot=r['_accels'],
                    action=r['_actions'],
                    susp_defl=r.get('_susp'), tire_defl=r.get('_tire'),
                    episode_return=r['episode_return'],
                    rms_accel=r['rms_accel'],
                    comfort_score=r['comfort_score'],
                    solve_ms_mean=r['solve_ms_mean'],
                    cpu_ms_mean=r['cpu_ms_mean'],
                    n_params=r['n_params'],
                ))
            if args.save_plots or args.save_gif:
                save_dir.mkdir(parents=True, exist_ok=True)
            if args.save_plots:
                _save_plot(r, 0, save_dir)
            if args.save_gif and r['_frames']:
                _save_gif(r['_frames'], save_dir / 'mpc_ep1.gif')

            n_bumps = r['bumps_total']
            bounds = bits['reward_bounds'](cfg, bits['EPISODE_STEPS'], n_bumps=n_bumps)
            print(f'\n  reward range  episode  [{bounds["episode_min"]:+.0f}, {bounds["episode_max"]:+.0f}]'
                  f'  (theoretical, {n_bumps} bumps, {bits["EPISODE_STEPS"]} steps)')
            print(f'                per-step [{bounds["per_step_min"]:+.2f}, {bounds["per_step_max"]:+.2f}]')

            summary = {
                'method': 'MPC-acados',
                'horizon_steps': args.horizon,
                'horizon_secs': round(args.horizon * 0.02, 3),
                'scenario': scenario_label,
                'return': r['episode_return'],
                'rms_accel': r['rms_accel'],
                'rms_pitch_accel': r.get('rms_pitch_accel', 0.0),
                'comfort': r['comfort_score'],
                'bumps': bumps_str,
                'mean_solve_ms': r['solve_ms_mean'],
                'episode': {k: v for k, v in r.items() if not k.startswith('_')},
            }

            out = save_dir / 'summary.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2))
            print(f'\n  Results → {out}')

            if run_logger is not None:
                saved = run_logger.save()
                print(f'  data logged → {run_logger.save_dir}')
                if 'mat' not in saved:
                    print('  (install scipy for .mat output: pip install scipy)')

            del ctrl
            del env
            gc.collect()
    finally:
        try:
            terminal_out.close()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.write(1, output.getvalue().encode('utf-8'))
        finally:
            os.close(saved_stdout)
            os.close(saved_stderr)
            os.close(devnull_fd)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n  interrupted')
        raise SystemExit(130)
