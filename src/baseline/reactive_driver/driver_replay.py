from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import savgol_filter

_ROOT = Path(__file__).resolve().parents[3]

from baseline.reactive_driver.driver import ReactiveController
from utils.scenario_loader import load_scenario, get_sequence_path


def _load_sequence(seq_root: Path):
    traj_path   = seq_root / 'export' / 'output.laz.trajectory.json'
    sensor_path = seq_root / 'export' / 'output.laz.sensor_export.json'
    if not traj_path.is_file():
        raise FileNotFoundError(f"trajectory not found: {traj_path}")
    with open(traj_path) as f:
        traj = json.load(f)
    sensor = None
    if sensor_path.is_file():
        with open(sensor_path) as f:
            sensor = json.load(f)
    return traj, sensor


def _build_arc_speed(traj: dict):
    poses = traj['trajectory_poses']
    ts  = np.array([p[0] for p in poses], dtype=float)
    xyz = np.array([p[1][:3] for p in poses], dtype=float)
    ds  = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
    s   = np.concatenate([[0.0], np.cumsum(ds)])
    dt  = np.diff(ts) / 1e6
    v   = np.concatenate([[0.0], ds / np.maximum(dt, 1e-9)])
    return ts, s, v


def _resample(ts_us, s, v, hz: float = 50.0):
    t0, t1   = ts_us[0], ts_us[-1]
    dt_us    = 1e6 / hz
    ts_grid  = np.arange(t0, t1, dt_us)
    s_grid   = np.interp(ts_grid, ts_us, s)
    v_grid   = np.interp(ts_grid, ts_us, v)

    wl = max(5, int(hz * 0.3) | 1)
    v_smooth = savgol_filter(v_grid, window_length=wl, polyorder=3)
    v_smooth = np.clip(v_smooth, 0.0, None)

    return ts_grid, s_grid, v_smooth


def _imu_az(sensor: dict | None, ts_traj_us, s_traj):
    if sensor is None or 'imu' not in sensor:
        return None, None
    imu    = sensor['imu']
    imu_ts = np.array([x['timestamp'] for x in imu], dtype=float)
    imu_az = np.array([x['accel_z']   for x in imu], dtype=float)

    mask   = (imu_ts >= ts_traj_us[0]) & (imu_ts <= ts_traj_us[-1])
    if not np.any(mask):
        return None, None

    imu_ts = imu_ts[mask]
    imu_az = imu_az[mask]
    az_body = imu_az - np.median(imu_az)

    s_imu = np.interp(imu_ts, ts_traj_us, s_traj)
    return s_imu, az_body


def replay(seq_root: Path, scenario_name: str,
           resample_hz: float = 50.0,
           a_max: float = 5.0) -> dict:
    traj, sensor = _load_sequence(seq_root)
    ts_us, s_raw, v_raw = _build_arc_speed(traj)
    ts_grid, s_grid, v_grid = _resample(ts_us, s_raw, v_raw, hz=resample_hz)

    dt_s    = 1.0 / resample_hz
    a_real  = np.gradient(v_grid, dt_s)
    u_real  = np.clip(a_real / a_max, -1.0, 1.0)

    bumps, speed_ms, sc_name, sc_desc = load_scenario(scenario_name)
    road = SimpleNamespace(_bumps=bumps)

    v_max = max(speed_ms, float(np.max(v_grid)))
    ctrl  = ReactiveController(
        v_max=v_max,
        v_min=0.0,
        a_max=a_max,
    )

    u_ctrl  = np.zeros(len(ts_grid))
    v_state = np.zeros(9)
    for i, (v, s) in enumerate(zip(v_grid, s_grid)):
        v_state[4] = v
        u_ctrl[i]  = ctrl.act(v_state, s, road)

    s_imu, az_body = _imu_az(sensor, ts_us, s_raw)

    t_sec = (ts_grid - ts_grid[0]) / 1e6

    return {
        't_s':      t_sec.tolist(),
        's_m':      s_grid.tolist(),
        'v_kmh':    (v_grid * 3.6).tolist(),
        'u_real':   u_real.tolist(),
        'u_ctrl':   u_ctrl.tolist(),
        'a_real':   a_real.tolist(),
        's_imu':    s_imu.tolist()  if s_imu  is not None else None,
        'az_body':  az_body.tolist() if az_body is not None else None,
        'bumps':    bumps,
        'scenario':    scenario_name,
        'sequence':    str(seq_root),
        'resample_hz': resample_hz,
        'n_steps':     len(ts_grid),
        'mae_action':  float(np.mean(np.abs(u_ctrl - u_real))),
        'corr_action': float(np.corrcoef(u_ctrl, u_real)[0, 1]),
        'mean_v_real_kmh': float(np.mean(v_grid[v_grid > 0.5]) * 3.6),
    }


def _make_figure(result: dict):
    import matplotlib.pyplot as plt

    s   = np.array(result['s_m'])
    v   = np.array(result['v_kmh'])
    ur  = np.array(result['u_real'])
    uc  = np.array(result['u_ctrl'])
    ar  = np.array(result['a_real'])
    bumps = result['bumps']

    n_rows = 3 + (1 if result['az_body'] is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)

    def _shade_bumps(ax):
        for x0, A, L in bumps:
            ax.axvspan(x0, x0 + L, alpha=0.15, color='tab:red', zorder=0)

    axes[0].plot(s, v, lw=1.2, color='tab:blue', label='real speed')
    _shade_bumps(axes[0])
    axes[0].set_ylabel('speed [km/h]')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(s, ar, lw=0.8, color='tab:orange', label='real a = dv/dt')
    axes[1].axhline(0, color='gray', lw=0.6)
    _shade_bumps(axes[1])
    axes[1].set_ylabel('a_real [m/s²]')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(s, ur, lw=1.0, color='tab:green',  label='u_real  (driver)',     alpha=0.8)
    axes[2].plot(s, uc, lw=1.0, color='tab:red',    label='u_ctrl  (rule-based)', alpha=0.8, ls='--')
    axes[2].axhline(0, color='gray', lw=0.6)
    axes[2].set_ylim(-1.2, 1.2)
    _shade_bumps(axes[2])
    axes[2].set_ylabel('action [-1, +1]')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    if result['az_body'] is not None:
        s_imu = np.array(result['s_imu'])
        az    = np.array(result['az_body'])
        axes[3].plot(s_imu, az, lw=0.7, color='tab:purple', label='IMU az − baseline')
        axes[3].axhline(0, color='gray', lw=0.6)
        _shade_bumps(axes[3])
        axes[3].set_ylabel('az_body [m/s²]')
        axes[3].legend(fontsize=8)
        axes[3].grid(alpha=0.3)

    axes[-1].set_xlabel('arc-length [m]')

    mae  = result['mae_action']
    corr = result['corr_action']
    fig.suptitle(
        f"Replay: {result['scenario']}  -  "
        f"MAE(action)={mae:.3f}  corr(action)={corr:.3f}\n"
        "(red shading = bump region from scenario YAML)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def save_plot(result: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = _make_figure(result)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  plot → {out_path}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sequence', default=None,
                    help='Path to the sequence root. '
                         'If omitted, read sequence_path from the scenario YAML.')
    ap.add_argument('--scenario', required=True,
                    help='Scenario name or YAML path (must have matching bump geometry)')
    ap.add_argument('--resample-hz', type=float, default=50.0,
                    help='Replay rate in Hz (default 50, matching env DT=0.02s)')
    ap.add_argument('--a-max', type=float, default=5.0,
                    help='Normalisation limit for actions [m/s²] (default 5.0)')
    ap.add_argument('--save-plot', action='store_true',
                    help='Save comparison plot as PNG')
    ap.add_argument('--render', action='store_true',
                    help='Show interactive plot window')
    ap.add_argument('--out', default=None,
                    help='Path for the JSON output (default: auto-named next to script)')
    args = ap.parse_args()

    if args.sequence:
        seq_root = Path(args.sequence).expanduser().resolve()
    else:
        seq_root = get_sequence_path(args.scenario)
        if seq_root is None:
            ap.error(
                f"Scenario '{args.scenario}' has no sequence_path field. "
                "Pass --sequence explicitly."
            )
    print(f'\nReplaying {seq_root.name} with scenario "{args.scenario}"')
    print(f'  resample_hz = {args.resample_hz}  a_max = {args.a_max} m/s²\n')

    result = replay(
        seq_root,
        args.scenario,
        resample_hz=args.resample_hz,
        a_max=args.a_max,
    )

    print(f'  steps         : {result["n_steps"]}')
    print(f'  mean v (real) : {result["mean_v_real_kmh"]:.1f} km/h')
    print(f'  MAE action    : {result["mae_action"]:.4f}')
    print(f'  corr(u)       : {result["corr_action"]:.4f}')

    out_json = (
        Path(args.out).expanduser()
        if args.out
        else _ROOT / 'logs' / 'eval' / 'results' / 'replay' /
             f'replay_{seq_root.name}_{args.scenario}.json'
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in result.items()
               if k not in ('t_s', 's_m', 'v_kmh', 'u_real', 'u_ctrl',
                            'a_real', 's_imu', 'az_body')}
    out_json.write_text(json.dumps(summary, indent=2))
    print(f'  summary JSON  → {out_json}')

    if args.save_plot:
        plot_path = out_json.with_suffix('.png')
        save_plot(result, plot_path)

    if args.render:
        import matplotlib.pyplot as plt
        fig = _make_figure(result)
        plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
