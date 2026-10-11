from __future__ import annotations

import os

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]

__import__("gym_env.quarter_car.envs")
from gym_env.quarter_car.config.reward_params import load_reward_config
from gym_env.quarter_car.config.preview_params import load_preview_config
from gym_env.quarter_car.config.env_params import DT, EPISODE_STEPS
from gym_env.quarter_car.wrappers import TerminationWrapper
from gym_env.quarter_car.reward.utils import reward_bounds
from baseline.reactive_driver.driver import ReactiveController
from utils.scenario_loader import load_scenario_any, enabled_eval_suites
from eval.data_logger import RunLogger, EpisodeData


def _vehicle_bits(vehicle: str) -> dict:
    if vehicle == 'half':
        __import__("gym_env.half_car.envs")
        from gym_env.half_car.config.env_params import EPISODE_STEPS as ES
        from gym_env.half_car.wrappers import TerminationWrapper as TW
        from gym_env.half_car.reward.utils import reward_bounds as rb
        from gym_env.half_car.envs.render import wait_for_episode_nav as nav
        from gym_env.half_car.config.preview_params import load_preview_config as lpc
        return dict(env_id='gym_env/HalfCar', EPISODE_STEPS=ES, v_index=8,
                    TerminationWrapper=TW, reward_bounds=rb, wait_for_episode_nav=nav,
                    load_preview_config=lpc)
    if vehicle not in ('quarter', None):
        raise ValueError(f"unknown vehicle {vehicle!r} (expected 'quarter' or 'half')")
    from gym_env.quarter_car.envs.render import wait_for_episode_nav as nav
    from gym_env.quarter_car.config.preview_params import load_preview_config as lpc
    return dict(env_id='gym_env/QuarterCar', EPISODE_STEPS=EPISODE_STEPS, v_index=4,
                TerminationWrapper=TerminationWrapper, reward_bounds=reward_bounds,
                wait_for_episode_nav=nav, load_preview_config=lpc)


def _steps_for_road(road, dt: float = 0.02, margin: float = 1.5) -> int:
    if road is None or not hasattr(road, '_bumps') or not road._bumps:
        return 1200
    x0, _, L = road._bumps[-1]
    road_length = x0 + L + 20.0
    v = max(float(road.speed), 1.0)
    return max(int(road_length / v / dt * margin), 600)


def run_episode(env, ctrl: ReactiveController, seed: int,
                scenario_road=None, collect_frames: bool = False,
                render_live: bool = False,
                perception_track=None, preview_cfg=None) -> dict:
    road_for_ep = copy.deepcopy(scenario_road) if scenario_road is not None else None

    raw = env.unwrapped
    raw._max_episode_steps = _steps_for_road(road_for_ep)

    opts   = {"road": road_for_ep} if road_for_ep is not None else {}
    obs, _ = env.reset(seed=seed, options=opts or None)

    if perception_track is not None and preview_cfg is not None:
        from baseline.perceived_road import make_perceived_road
        plan_road = make_perceived_road(raw._road, perception_track, preview_cfg)
    else:
        plan_road = raw._road

    saved_v_max = ctrl.v_max
    if road_for_ep is not None and hasattr(road_for_ep, 'speed'):
        ctrl.v_max = min(ctrl.v_max, float(road_for_ep.speed))

    ep_return  = 0.0
    accel_sq   = 0.0
    pitch_sq   = 0.0
    n_steps    = 0
    act_times: list[float] = []
    cpu_times: list[float] = []
    speeds:    list[float] = []
    v_inits:   list[float] = []
    accels:    list[float] = []
    actions:   list[float] = []
    rewards:   list[float] = []
    susp_defl: list[float] = []
    tire_defl: list[float] = []
    frames:    list        = []
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
        act_times.append(time.perf_counter() - t0)
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

    ctrl.v_max = saved_v_max

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
        'act_us_mean':    round(float(np.mean(act_times)) * 1e6, 1),
        'solve_ms_mean':  round(float(np.mean(act_times)) * 1e3, 4),
        'cpu_ms_mean':    round(float(np.mean(cpu_times)) * 1e3, 4),
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
        f'Reactive Driver - ep {ep_i+1}  return={result["episode_return"]:.1f}',
        fontsize=11,
    )
    fig.tight_layout()
    out = save_dir / f'reactive_driver_ep{ep_i+1}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  plot saved → {out}')


def _run_suite_reactive(env, ctrl: ReactiveController, args,
                        save_dir: Path, log_root: Path | None, cfg,
                        preview_cfg=None) -> None:
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
    print('\n  Reactive Driver - suite run')
    print(f'  suites : {[n for n, _ in suites]}')
    print(f'  {total} scenarios\n')
    print(f'  {"Scenario":<24}  {"Return":>9}  {"RMS m/s²":>8}{p_hdr}  {"Comfort":>7}  Bumps')
    print(f'  {"-"*24}  {"-"*9}  {"-"*8}{p_sep}  {"-"*7}  {"-"*5}')

    all_returns: list[float] = []
    export_suites = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

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
                  f'  {r["rms_accel"]:>8.3f}{p_col}  {r["comfort_score"]:>7.3f}  {bumps_str}')
            all_returns.append(r['episode_return'])

            if log_root is not None:
                cfg_r = load_reward_config()
                logger = RunLogger(
                    method='Reactive', road=sc_name, out_root=log_root,
                    dt=DT, v_max_kmh=cfg_r.v_limit * 3.6,
                    a_comfort=cfg_r.a_comfort, a_limit=cfg_r.a_limit,
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

    if all_returns:
        print(f'\n  OVERALL  {len(all_returns)} scenarios'
              f'  mean_return={float(np.mean(all_returns)):+.1f}')

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f'suite_reactive_{ts}.json'
    out.write_text(json.dumps({'timestamp': ts, 'method': 'Reactive',
                               'suites': export_suites}, indent=2))
    print(f'\n  JSON saved → {out}')
    if log_root:
        print(f'  data logged → {log_root}/Reactive/')


def _render_scenarios(env, ctrl: ReactiveController, args, nav_fn=None,
                      preview_cfg=None) -> None:
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
        raw.set_render_label(f'Reactive · {sc_name} - {sc_desc}   [{idx+1}/{n} · {suite_name}]')
        r = run_episode(env, ctrl, seed=args.seed, scenario_road=road,
                        collect_frames=False, render_live=True,
                        perception_track=_track, preview_cfg=preview_cfg)
        nav = getattr(raw, '_nav_request', None)
        raw._nav_request = None
        if nav is None:
            print(f'  [{idx+1}/{n}] {sc_name:<24}  return={r["episode_return"]:+.1f}  '
                  f'RMS={r["rms_accel"]:.3f}  obst={_obst_str(r)}')
            nav = nav_fn(env, allow_prev=True)
        if nav == 'quit':
            break
        elif nav == 'next':
            idx = min(idx + 1, n - 1)
        elif nav == 'prev':
            idx = max(idx - 1, 0)


def main() -> None:
    cfg   = load_reward_config()
    pcfg  = load_preview_config()

    ap = argparse.ArgumentParser('Reactive driver baseline (intuition-based)')
    ap.add_argument('--vehicle',         default='half', choices=('quarter', 'half'),
                    help='vehicle dynamics model')
    ap.add_argument('--no-perception',   dest='perception', action='store_false',
                    default=True,
                    help='plan on the true road (oracle) instead of perceived detections')
    ap.add_argument('--seed',            type=int,   default=42)
    ap.add_argument('--scenario',        default=None,
                    help='fixed single scenario name or path')
    ap.add_argument('--suites',          nargs='+',  metavar='FOLDER', default=None,
                    help='suite folder names to run (default: enabled in eval_suites.yaml)')
    ap.add_argument('--preview',         type=float,
                    default=pcfg.preview_distance)
    ap.add_argument('--render',           action='store_true',
                    help='live window; with no --scenario, step through eval scenarios (←/→)')
    ap.add_argument('--save-gif',        action='store_true',
                    help='save per-episode GIF (headless-safe offscreen rendering)')
    ap.add_argument('--save-plots',      action='store_true',
                    help='save per-episode time-series PNG')
    ap.add_argument('--log-data',        metavar='DIR', default='data',
                    help='Save run data (.mat/.npz) to DIR/Reactive/<road>/<run_N>/. '
                         'Omit DIR to use <repo>/data/.',
                    nargs='?', const='__default__')
    args = ap.parse_args()

    if args.save_gif and not args.render:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    bits = _vehicle_bits(args.vehicle)
    preview_cfg = bits['load_preview_config']() if args.perception else None
    ctrl = ReactiveController(
        v_max          = float(cfg.v_limit),
        v_min          = float(cfg.v_min),
        a_max          = float(cfg.a_max),
        preview_m      = args.preview,
        a_brake        = float(cfg.a_comfort),
        a_accel        = float(cfg.a_comfort),
        v_index        = bits['v_index'],
    )

    if args.render:
        render_mode = 'human'
    elif args.save_gif:
        render_mode = 'rgb_array'
    else:
        render_mode = 'none'
    env = bits['TerminationWrapper'](gym.make(bits['env_id'], render_mode=render_mode))

    log_root = (
        _ROOT / 'data'
        if args.log_data == '__default__'
        else Path(args.log_data) if args.log_data
        else None
    )

    if args.render and not args.scenario:
        _render_scenarios(env, ctrl, args, nav_fn=bits['wait_for_episode_nav'],
                          preview_cfg=preview_cfg)
        env.close()
        return

    scenario_road  = None
    scenario_label = 'speed_bump'
    if args.scenario:
        scenario_road, _track, _v, sc_desc = load_scenario_any(args.scenario)
        sc_name = args.scenario
        scenario_label = args.scenario
        print(f'\n  Scenario : {sc_name}')
        print(f'           : {sc_desc}')

    save_dir = _ROOT / 'logs' / 'eval' / 'results' / 'reactive_driver' / f'{scenario_label}'

    w = max(len(sc_name), 8) if args.scenario else 8
    half = getattr(args, 'vehicle', 'quarter') == 'half'
    p_hdr = f'  {"Pitch r/s²":>10}' if half else ''
    p_sep = f'  {"-"*10}' if half else ''
    print('\n  Reactive Driver Baseline  (intuition-based)')
    print(f'  preview      : {args.preview} m        (config/preview/preview.yaml)')
    print(f'  steep→v_min  : {ctrl.steep_full_slow}  (rise/run)')
    print(f'  hole severity: {ctrl.hole_severity}')
    print(f'  a_brake      : {ctrl.a_brake} m/s²')
    print(f'  road         : {scenario_label}\n')
    print(f'  {"Scenario":<{w}}  {"Return":>9}  {"RMS-a m/s²":>10}{p_hdr}  {"Comfort":>8}  {"Bumps":>7}  {"Act µs":>8}')
    print(f'  {"-"*w}  {"-"*9}  {"-"*10}{p_sep}  {"-"*8}  {"-"*7}  {"-"*8}')

    if not args.scenario:
        _run_suite_reactive(env, ctrl, args, save_dir, log_root, cfg, preview_cfg)
        env.close()
        return

    run_logger: RunLogger | None = (
        RunLogger(
            method='Reactive',
            road=scenario_label,
            out_root=log_root,
            dt=DT,
            v_max_kmh=cfg.v_limit * 3.6,
            a_comfort=cfg.a_comfort,
            a_limit=cfg.a_limit,
            run_subdir=False,
            bumps=getattr(scenario_road, '_bumps', None),
        )
        if log_root is not None else None
    )

    if args.render:
        wait_for_episode_nav = bits['wait_for_episode_nav']
        env.unwrapped.set_render_label(f'Reactive · {sc_name} - {sc_desc}')

    while True:
        r = run_episode(env, ctrl, seed=args.seed,
                        scenario_road=scenario_road,
                        collect_frames=args.save_gif,
                        render_live=args.render)
        bumps_str = _obst_str(r)
        p_col = f'  {r.get("rms_pitch_accel", 0.0):>10.3f}' if half else ''
        print(f'  {sc_name:<{w}}  {r["episode_return"]:>+9.1f}  '
              f'{r["rms_accel"]:>10.3f}{p_col}  {r["comfort_score"]:>8.3f}  '
              f'{bumps_str:>7}  {r["act_us_mean"]:>8.1f}')
        nr = getattr(env.unwrapped, '_nav_request', None)
        env.unwrapped._nav_request = None
        if args.render and (nr == 'replay' or (nr is None and wait_for_episode_nav(env) == 'replay')):
            continue
        break

    env.close()

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
        _save_gif(r['_frames'], save_dir / 'reactive_driver_ep1.gif')

    n_bumps = r['bumps_total']
    bounds = bits['reward_bounds'](cfg, bits['EPISODE_STEPS'], n_bumps=n_bumps)
    print(f'\n  reward range  episode  [{bounds["episode_min"]:+.0f}, {bounds["episode_max"]:+.0f}]'
          f'  (theoretical, {n_bumps} bumps, {bits["EPISODE_STEPS"]} steps)')
    print(f'                per-step [{bounds["per_step_min"]:+.2f}, {bounds["per_step_max"]:+.2f}]')

    summary = {
        'method':           'Reactive',
        'preview_m':        args.preview,
        'steep_full_slow':  ctrl.steep_full_slow,
        'hole_severity':    ctrl.hole_severity,
        'a_brake':          ctrl.a_brake,
        'scenario':         scenario_label,
        'return':           r['episode_return'],
        'rms_accel':        r['rms_accel'],
        'rms_pitch_accel':  r.get('rms_pitch_accel', 0.0),
        'comfort':          r['comfort_score'],
        'bumps':            bumps_str,
        'episode':          {k: v for k, v in r.items() if not k.startswith('_')},
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


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n  interrupted: no data saved')
        sys.exit(0)
