from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym

from utils.paths import ROOT as _ROOT

from gym_env.common import TerminationWrapper
from gym_env.common.env_params import DT, EPISODE_STEPS
from gym_env.common.preview_params import load_preview_config
from gym_env.common.reward_params import load_reward_config
from gym_env.common.reward_utils import reward_bounds
from gym_env.vehicles import get_vehicle
from baseline import runner
from utils.gif import save_gif
from baseline.reactive_driver.driver import ReactiveController
from utils.scenario_loader import load_scenario_any
from eval.data_logger import RunLogger, EpisodeData


def _vehicle_bits(vehicle: str) -> dict:
    v = get_vehicle(vehicle)
    return dict(env_id=v.env_id, EPISODE_STEPS=EPISODE_STEPS, v_index=v.v_index,
                TerminationWrapper=TerminationWrapper, reward_bounds=reward_bounds,
                wait_for_episode_nav=v.render().wait_for_episode_nav,
                load_preview_config=load_preview_config, load_reward_config=load_reward_config)


def run_episode(env, ctrl: ReactiveController, seed: int, scenario_road=None,
                collect_frames: bool = False, render_live: bool = False,
                perception_track=None, preview_cfg=None) -> dict:
    def steps(road):
        speed = max(float(road.speed), 1.0) if road is not None else 1.0
        return runner.steps_for_road(road, speed, no_bumps=1200, floor=600)

    def on_start(plan_road, road):
        saved = ctrl.v_max
        if road is not None and hasattr(road, 'speed'):
            ctrl.v_max = min(ctrl.v_max, float(road.speed))
        return lambda: setattr(ctrl, 'v_max', saved)

    r = runner.run_episode(env, ctrl, seed, steps, scenario_road, collect_frames, render_live,
                           perception_track, preview_cfg, on_start=on_start)
    r['act_us_mean'] = round(r['act_ms_mean'] * 1e3, 2)
    return r


def _run_suite_reactive(env, ctrl: ReactiveController, args,
                        save_dir: Path, log_root: Path | None, cfg,
                        preview_cfg=None) -> None:
    runner.run_suite(env, lambda e, seed, **kw: run_episode(e, ctrl, seed, **kw), args, save_dir,
                     log_root, cfg, 'Reactive', 'reactive', preview_cfg)


def _render_scenarios(env, ctrl: ReactiveController, args, nav_fn=None,
                      preview_cfg=None) -> None:
    runner.render_scenarios(env, lambda e, seed, **kw: run_episode(e, ctrl, seed, **kw), args,
                            nav_fn, preview_cfg, label='Reactive · ')


_obst_str = runner.obst_str


def _save_plot(result: dict, ep_i: int, save_dir: Path) -> None:
    runner.save_plot(result, ep_i, save_dir, 'Reactive Driver', 'reactive_driver')


def main() -> None:
    ap = argparse.ArgumentParser('Reactive driver baseline (intuition-based)')
    runner.add_common_args(ap, 'Reactive')
    ap.add_argument('--preview', type=float, default=None, help='preview distance in m (default: preview.yaml)')
    args = ap.parse_args()

    if args.save_gif and not args.render:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

    bits = _vehicle_bits(args.vehicle)
    cfg = bits['load_reward_config']()
    pcfg = bits['load_preview_config']()
    if args.preview is None:
        args.preview = pcfg.preview_distance
    preview_cfg = pcfg if args.perception else None
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
    env = bits['TerminationWrapper'](gym.make(bits['env_id'], render_mode=render_mode,
                                               action_mode='acceleration'))

    log_root = runner.log_root(args)

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
            solve_ms_mean=r['act_ms_mean'],
            cpu_ms_mean=r['cpu_ms_mean'],
            n_params=r['n_params'],
        ))
    if args.save_plots or args.save_gif:
        save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_plots:
        _save_plot(r, 0, save_dir)
    if args.save_gif and r['_frames']:
        save_gif(r['_frames'], save_dir / 'reactive_driver_ep1.gif')

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
