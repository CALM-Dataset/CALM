from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from eval.data_logger import EpisodeData, RunLogger
from gym_env.common.env_params import DT
from utils.paths import ROOT
from utils.scenario_loader import enabled_eval_suites, load_scenario_any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

DATA_ROOT = ROOT / 'data'

NO_SCENARIOS = ('No scenarios. Enable a suite in config/eval/eval_suites.yaml '
                '(incl. `real`), or pass --suites <folder|real>.')


def add_common_args(ap, method: str) -> None:
    ap.add_argument('--vehicle', default='half', choices=('quarter', 'half'),
                    help='vehicle dynamics model')
    ap.add_argument('--no-perception', dest='perception', action='store_false', default=True,
                    help='plan on the true road (oracle) instead of the perceived detections')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--scenario', default=None, help='fixed single scenario name or path')
    ap.add_argument('--suites', nargs='+', metavar='FOLDER', default=None,
                    help='suite folder names to run (default: enabled in eval_suites.yaml)')
    ap.add_argument('--render', action='store_true',
                    help='live window; with no --scenario, step through eval scenarios (←/→)')
    ap.add_argument('--save-gif', action='store_true',
                    help='save per-episode GIF (headless-safe offscreen rendering)')
    ap.add_argument('--save-plots', action='store_true', help='save per-episode time-series PNG')
    ap.add_argument('--log-data', metavar='DIR', default='data', nargs='?', const='__default__',
                    help=f'Save run data (.mat/.npz) to DIR/{method}/<road>/<run_N>/. '
                         'Omit DIR to use <repo>/data/.')


def log_root(args) -> Path | None:
    if args.log_data == '__default__':
        return DATA_ROOT
    return Path(args.log_data) if args.log_data else None


def obst_str(r: dict) -> str:
    s = f'{r["bumps_passed"]}/{r["bumps_total"]}' if r.get('bumps_total') else '-'
    if r.get('holes_total'):
        s += f' H{r["holes_passed"]}/{r["holes_total"]}'
    return s


def save_gif(frames: list, path: Path) -> None:
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     optimize=False, duration=20, loop=0)
        print(f'  GIF saved → {path}')
    except ImportError:
        print('  [PIL not installed - cannot save GIF; run: pip install Pillow]')


def save_plot(result: dict, ep_i: int, save_dir: Path, title: str, stem: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(result['n_steps']) * DT
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, [v * 3.6 for v in result['_speeds']], label='speed km/h')
    axes[0].set_ylabel('speed [km/h]')
    axes[1].plot(t, result['_accels'], color='tab:orange', label='body accel')
    axes[1].axhline(0, color='gray', lw=0.7)
    axes[1].set_ylabel('z̈_B [m/s²]')
    axes[2].plot(t, result['_actions'], color='tab:green', label='action u')
    axes[2].axhline(0, color='gray', lw=0.7)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].set_ylabel('action [-1,1]')
    axes[2].set_xlabel('time [s]')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f'{title} - ep {ep_i + 1}  return={result["episode_return"]:.1f}', fontsize=11)
    fig.tight_layout()
    out = save_dir / f'{stem}_ep{ep_i + 1}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  plot saved → {out}')


def steps_for_road(road, speed: float, no_bumps: int, floor: int,
                   dt: float = DT, margin: float = 1.5) -> int:
    if road is None or not hasattr(road, '_bumps') or not road._bumps:
        return no_bumps
    x0, _, L = road._bumps[-1]
    return max(int((x0 + L + 20.0) / speed / dt * margin), floor)


def run_episode(env, ctrl, seed: int, steps_fn, scenario_road=None,
                collect_frames: bool = False, render_live: bool = False,
                perception_track=None, preview_cfg=None, on_start=None) -> dict:
    road_for_ep = copy.deepcopy(scenario_road) if scenario_road is not None else None

    raw = env.unwrapped
    raw._max_episode_steps = steps_fn(road_for_ep)

    opts = {"road": road_for_ep} if road_for_ep is not None else {}
    env.reset(seed=seed, options=opts or None)

    if perception_track is not None and preview_cfg is not None:
        from baseline.perceived_road import make_perceived_road
        plan_road = make_perceived_road(raw._road, perception_track, preview_cfg)
    else:
        plan_road = raw._road
    restore = on_start(plan_road, road_for_ep) if on_start is not None else None

    ep_return = 0.0
    accel_sq = 0.0
    pitch_sq = 0.0
    n_steps = 0
    act_times: list[float] = []
    cpu_times: list[float] = []
    speeds: list[float] = []
    v_inits: list[float] = []
    accels: list[float] = []
    actions: list[float] = []
    rewards: list[float] = []
    susp_defl: list[float] = []
    tire_defl: list[float] = []
    frames: list = []
    done = False

    try:
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
            u = ctrl.act(raw._state.copy(), raw._s_pos, plan_road)
            act_times.append(time.perf_counter() - t0)
            cpu_times.append(time.process_time() - c0)

            _, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
            ep_return += reward
            accel_sq += info.get('z_B_ddot', 0.0) ** 2
            pitch_sq += info.get('phi_ddot', 0.0) ** 2
            n_steps += 1
            speeds.append(info.get('speed', 0.0))
            v_inits.append(info.get('v_init', 0.0))
            accels.append(info.get('z_B_ddot', 0.0))
            actions.append(float(u))
            rewards.append(float(reward))
            susp_defl.append(info.get('susp_defl', 0.0))
            tire_defl.append(info.get('tire_defl', 0.0))
            done = terminated or truncated
    finally:
        if restore is not None:
            restore()

    rms_accel = float(np.sqrt(accel_sq / max(n_steps, 1)))
    rms_pitch = float(np.sqrt(pitch_sq / max(n_steps, 1)))
    return {
        'episode_return': round(float(ep_return), 2),
        'rms_accel': round(rms_accel, 4),
        'rms_pitch_accel': round(rms_pitch, 4),
        'comfort_score': round(max(0.0, 1.0 - rms_accel / raw._rcfg.a_limit), 4),
        'n_steps': n_steps,
        'bumps_passed': int(raw._bumps_passed),
        'bumps_total': int(raw._bumps_total),
        'holes_passed': int(raw._holes_passed),
        'holes_total': int(raw._holes_total),
        'act_ms_mean': round(float(np.mean(act_times)) * 1e3, 4),
        'act_ms_max': round(float(np.max(act_times)) * 1e3, 4),
        'cpu_ms_mean': round(float(np.mean(cpu_times)) * 1e3, 4),
        'n_params': 0.0,
        '_speeds': speeds,
        '_v_inits': v_inits,
        '_accels': accels,
        '_actions': actions,
        '_rewards': rewards,
        '_susp': susp_defl,
        '_tire': tire_defl,
        '_frames': frames,
    }


def log_episode(log_root, method: str, road_name: str, road, cfg, r: dict) -> None:
    logger = RunLogger(
        method=method, road=road_name, out_root=log_root,
        dt=DT, v_max_kmh=cfg.v_limit * 3.6,
        a_comfort=cfg.a_comfort, a_limit=cfg.a_limit,
        run_subdir=False, bumps=getattr(road, '_bumps', None),
    )
    logger.add(EpisodeData(
        v=r['_speeds'], v_init=r['_v_inits'],
        z_B_ddot=r['_accels'], action=r['_actions'],
        reward=r['_rewards'],
        susp_defl=r.get('_susp'), tire_defl=r.get('_tire'),
        episode_return=r['episode_return'],
        rms_accel=r['rms_accel'],
        comfort_score=r['comfort_score'],
        solve_ms_mean=r['act_ms_mean'],
        cpu_ms_mean=r['cpu_ms_mean'],
        n_params=r['n_params'],
    ))
    logger.save()


def render_scenarios(env, run_fn, args, nav_fn, preview_cfg=None,
                     label: str = '', report=print) -> None:
    scenarios = [(name, sc) for name, scs in enabled_eval_suites(args.suites) for sc in scs]
    if not scenarios:
        print(NO_SCENARIOS)
        return

    raw = env.unwrapped
    n = len(scenarios)
    idx = 0
    while True:
        suite_name, sc_name = scenarios[idx]
        road, track, _v, sc_desc = load_scenario_any(sc_name)
        raw.set_render_label(f'{label}{sc_name} - {sc_desc}   [{idx + 1}/{n} · {suite_name}]')
        r = run_fn(env, args.seed, scenario_road=road, collect_frames=False, render_live=True,
                   perception_track=track, preview_cfg=preview_cfg)
        raw = env.unwrapped
        nav = getattr(raw, '_nav_request', None)
        raw._nav_request = None
        if nav is None:
            report(f'  [{idx + 1}/{n}] {sc_name:<24}  return={r["episode_return"]:+.1f}  '
                   f'RMS={r["rms_accel"]:.3f}  obst={obst_str(r)}')
            nav = nav_fn(env, allow_prev=True)
        if nav == 'quit':
            break
        elif nav == 'next':
            idx = min(idx + 1, n - 1)
        elif nav == 'prev':
            idx = max(idx - 1, 0)


def run_suite(env, run_fn, args, save_dir: Path, log_root, cfg, method: str, slug: str,
              preview_cfg=None, progress_out=None, time_header: str | None = None) -> None:
    suites = enabled_eval_suites(args.suites)
    if not suites:
        print(NO_SCENARIOS)
        return

    half = getattr(args, 'vehicle', 'quarter') == 'half'
    p_hdr = f'  {"Pitch r/s²":>10}' if half else ''
    p_sep = f'  {"-" * 10}' if half else ''
    t_hdr = f'  {time_header:>9}' if time_header else ''
    t_sep = f'  {"-" * 9}' if time_header else ''
    total = sum(len(s) for _, s in suites)
    print(f'\n  {method} - suite run')
    print(f'  suites : {[n for n, _ in suites]}')
    print(f'  {total} scenarios\n')
    print(f'  {"Scenario":<24}  {"Return":>9}  {"RMS m/s²":>8}{p_hdr}  {"Comfort":>7}{t_hdr}  Bumps')
    print(f'  {"-" * 24}  {"-" * 9}  {"-" * 8}{p_sep}  {"-" * 7}{t_sep}  {"-" * 5}')

    all_returns: list[float] = []
    export_suites = []
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    bar = (tqdm(total=total, desc=f'{method} scenarios', unit='sc', file=progress_out, leave=False)
           if tqdm is not None and progress_out is not None else None)

    for suite_name, scenarios in suites:
        suite_rows = []
        print(f'\n  ── {suite_name} ──')

        for sc_name in scenarios:
            road, track, _v, _desc = load_scenario_any(sc_name)
            r = run_fn(env, args.seed, scenario_road=road, collect_frames=False, render_live=False,
                       perception_track=track, preview_cfg=preview_cfg)
            p_col = f'  {r.get("rms_pitch_accel", 0.0):>10.3f}' if half else ''
            t_col = f'  {r["act_ms_mean"]:>9.2f}' if time_header else ''
            print(f'  {sc_name:<24}  {r["episode_return"]:>+9.1f}'
                  f'  {r["rms_accel"]:>8.3f}{p_col}  {r["comfort_score"]:>7.3f}{t_col}  {obst_str(r)}')
            all_returns.append(r['episode_return'])
            if bar is not None:
                bar.set_postfix({'scenario': sc_name, 'return': f'{r["episode_return"]:+.0f}'}, refresh=False)
                bar.update(1)
            if log_root is not None:
                log_episode(log_root, method, sc_name, road, cfg, r)
            suite_rows.append({'scenario': sc_name, **{k: v for k, v in r.items() if not k.startswith('_')}})

        suite_mean = float(np.mean([row['episode_return'] for row in suite_rows])) if suite_rows else 0.0
        print(f'  {"Mean":<24}  {suite_mean:>+9.1f}')
        export_suites.append({'suite': suite_name, 'scenarios': suite_rows})

    if bar is not None:
        bar.close()

    if all_returns:
        print(f'\n  OVERALL  {len(all_returns)} scenarios  mean_return={float(np.mean(all_returns)):+.1f}')

    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f'suite_{slug}_{ts}.json'
    out.write_text(json.dumps({'timestamp': ts, 'method': method, 'suites': export_suites}, indent=2))
    print(f'\n  JSON saved → {out}')
    if log_root:
        print(f'  data logged → {log_root}/{method}/')
