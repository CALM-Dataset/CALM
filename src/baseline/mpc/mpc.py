from __future__ import annotations

import os

from pathlib import Path as _Path

from utils.paths import ROOT as _ROOT

_repo_acados = _ROOT / 'acados'
_acados = _repo_acados if (_repo_acados / 'lib').is_dir() else _Path(os.environ.get('ACADOS_SOURCE_DIR') or _repo_acados)
os.environ['ACADOS_SOURCE_DIR'] = str(_acados)
_lib = str(_acados / 'lib')
if _lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import argparse
import gc
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import gymnasium as gym

_acados_py = str(_ROOT / 'acados' / 'interfaces' / 'acados_template')
if _acados_py not in sys.path:
    sys.path.insert(0, _acados_py)

from gym_env.common import TerminationWrapper
from gym_env.common.env_params import DT, EPISODE_STEPS
from gym_env.common.preview.preview_params import load_preview_config
from gym_env.common.reward.reward_params import load_reward_config
from gym_env.common.utils.reward_utils import reward_bounds
from gym_env.vehicles import get_vehicle
from baseline import runner
from gym_env.common.render.render import RenderQuit
from utils.gif import save_gif
from baseline.mpc.config import load_mpc_cfg, reward_config as mpc_reward_config
from baseline.mpc.controller import MPCController
from utils.scenario_loader import load_scenario_any, scenario_title
from eval.data_logger import RunLogger, EpisodeData


def make_env(bits: dict, cfg, render_mode: str = 'none'):
    return bits['TerminationWrapper'](gym.make(bits['env_id'], render_mode=render_mode,
                                               reward_config=cfg, **bits.get('env_kwargs', {})))


def _vehicle_bits(vehicle: str) -> dict:
    v = get_vehicle(vehicle)
    return dict(env_id=v.env_id, env_kwargs={'action_mode': 'acceleration'}, PHYSICS=v.physics(),
                EPISODE_STEPS=EPISODE_STEPS, load_reward_config=lambda: mpc_reward_config(load_reward_config),
                TerminationWrapper=TerminationWrapper, reward_bounds=reward_bounds,
                wait_for_episode_nav=v.render().wait_for_episode_nav, load_preview_config=load_preview_config)


def run_episode(env, ctrl: MPCController, seed: int, scenario_road=None,
                collect_frames: bool = False, render_live: bool = False,
                perception_track=None, preview_cfg=None) -> dict:
    v = max(float(ctrl._cfg.v_min), 1.0)
    r = runner.run_episode(
        env, ctrl, seed, lambda road: runner.steps_for_road(road, v, no_bumps=4000, floor=1200),
        scenario_road, collect_frames, render_live, perception_track, preview_cfg,
        on_start=lambda plan_road, road: ctrl.reset(plan_road))
    r['solve_ms_mean'], r['solve_ms_max'] = r['act_ms_mean'], r['act_ms_max']
    return r


def _run_suite_mpc(env, ctrl: MPCController, args, save_dir: Path,
                   log_root, cfg, terminal_out=None, preview_cfg=None) -> None:
    runner.run_suite(env, lambda e, seed, **kw: run_episode(e, ctrl, seed, **kw), args, save_dir,
                     log_root, cfg, 'MPC', 'mpc', preview_cfg, terminal_out, time_header='Solve ms')


def _render_scenarios(env, ctrl: MPCController, args, terminal_out=None,
                      nav_fn=None, preview_cfg=None) -> None:
    if nav_fn is None:
        nav_fn = get_vehicle(args.vehicle).render().wait_for_episode_nav
    report = (lambda line: terminal_out.write(line + '\n')) if terminal_out is not None else (lambda line: None)
    runner.render_scenarios(env, lambda e, seed, **kw: run_episode(e, ctrl, seed, **kw), args,
                            nav_fn, preview_cfg, label='MPC', report=report)


_obst_str = runner.obst_str


def _save_plot(result: dict, ep_i: int, save_dir: Path) -> None:
    runner.save_plot(result, ep_i, save_dir, 'MPC', 'mpc')


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
            mcfg = load_mpc_cfg()

            ap = argparse.ArgumentParser('MPC baseline (quarter- or half-car)')
            runner.add_common_args(ap, 'MPC')
            ap.add_argument('--horizon', type=int, default=mcfg.get('N', 50), help='prediction horizon (steps)')
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
            save_dir = _ROOT / 'logs' / 'eval' / 'results' / 'mpc' / scenario_label

            log_root = runner.log_root(args)

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
                env.unwrapped.set_render_label(scenario_title(sc_name, 'MPC'))

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
                save_gif(r['_frames'], save_dir / 'mpc_ep1.gif')

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
    except RenderQuit:
        print('\n  closed')
    except KeyboardInterrupt:
        print('\n  interrupted')
        raise SystemExit(130)
