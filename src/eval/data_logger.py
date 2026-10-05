from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np


def _save_formats() -> set[str]:
    raw = os.environ.get('EVAL_SAVE_FORMATS')
    if not raw:
        return {'mat', 'npz', 'json'}
    return {p.strip().lower() for p in raw.split(',') if p.strip()}


def _to_arr(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64) if v is not None else np.empty(0)


def _pad_episodes(cols: list[np.ndarray]) -> np.ndarray:
    if not cols:
        return np.empty((0, 0))
    max_len = max(len(c) for c in cols)
    out = np.full((max_len, len(cols)), np.nan, dtype=np.float64)
    for j, c in enumerate(cols):
        out[:len(c), j] = c
    return out


class EpisodeData:
    __slots__ = ('v', 'v_init', 'z_B_ddot', 'z_W_ddot',
                 'action', 'reward', 's_pos', 'susp_defl', 'tire_defl',
                 'episode_return', 'rms_accel', 'comfort_score',
                 'solve_ms_mean', 'cpu_ms_mean', 'n_params')

    def __init__(
        self,
        v:              list | np.ndarray,
        v_init:          list | np.ndarray,
        z_B_ddot:       list | np.ndarray,
        action:         list | np.ndarray,
        episode_return: float,
        rms_accel:      float,
        comfort_score:  float,
        z_W_ddot:       Optional[list | np.ndarray] = None,
        reward:         Optional[list | np.ndarray] = None,
        s_pos:          Optional[list | np.ndarray] = None,
        susp_defl:      Optional[list | np.ndarray] = None,
        tire_defl:      Optional[list | np.ndarray] = None,
        solve_ms_mean:  float = float('nan'),
        cpu_ms_mean:    float = float('nan'),
        n_params:       float = float('nan'),
    ):
        self.v              = _to_arr(v)
        self.v_init          = _to_arr(v_init)
        self.z_B_ddot       = _to_arr(z_B_ddot)
        self.z_W_ddot       = _to_arr(z_W_ddot) if z_W_ddot is not None else None
        self.action         = _to_arr(action)
        self.reward         = _to_arr(reward)   if reward    is not None else None
        self.s_pos          = _to_arr(s_pos)    if s_pos     is not None else None
        self.susp_defl      = _to_arr(susp_defl) if susp_defl is not None else None
        self.tire_defl      = _to_arr(tire_defl) if tire_defl is not None else None
        self.episode_return = float(episode_return)
        self.rms_accel      = float(rms_accel)
        self.comfort_score  = float(comfort_score)
        self.solve_ms_mean  = float(solve_ms_mean)
        self.cpu_ms_mean    = float(cpu_ms_mean)
        self.n_params       = float(n_params)

    @property
    def n_steps(self) -> int:
        return len(self.v)


class RunLogger:
    def __init__(
        self,
        method:      str,
        road:        str,
        out_root:    Path,
        dt:          float = 0.02,
        v_max_kmh:   float = 72.0,
        a_comfort:   float = 3.0,
        a_limit:     float = 5.0,
        run_id:      str | None = None,
        run_subdir:  bool = True,
        bumps:       list | None = None,
    ):
        self.method    = method
        self.road      = road
        self.dt        = dt
        self.v_max_kmh = v_max_kmh
        self.a_comfort = a_comfort
        self.a_limit   = a_limit
        self.bumps     = [(float(x0), float(A), float(L)) for x0, A, L in bumps] if bumps else []
        self._episodes: list[EpisodeData] = []

        if run_subdir:
            if run_id is None:
                parent = out_root / method / road
                existing: set[int] = set()
                if parent.is_dir():
                    for d in parent.iterdir():
                        m = re.match(r'run_(\d+)$', d.name)
                        if m:
                            existing.add(int(m.group(1)))
                n = 1
                while n in existing:
                    n += 1
                run_id = f'run_{n}'
            self.run_id   = run_id
            self.save_dir = out_root / method / road / self.run_id
        else:
            self.run_id   = run_id or 'run'
            self.save_dir = out_root / method / road
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def add(self, ep: EpisodeData) -> None:
        self._episodes.append(ep)

    def save(self) -> dict[str, Path]:
        eps = self._episodes
        n   = len(eps)
        if n == 0:
            return {}

        dt = self.dt

        arrays: dict[str, np.ndarray] = {}

        arrays['t'] = _pad_episodes([np.arange(e.n_steps, dtype=np.float64) * dt
                                     for e in eps])

        arrays['v_kmh']     = _pad_episodes([e.v     * 3.6 for e in eps])
        arrays['v_init_kmh'] = _pad_episodes([e.v_init * 3.6 for e in eps])
        arrays['z_B_ddot']  = _pad_episodes([e.z_B_ddot    for e in eps])
        arrays['action']    = _pad_episodes([e.action       for e in eps])

        if any(e.z_W_ddot is not None and len(e.z_W_ddot) > 0 for e in eps):
            arrays['z_W_ddot'] = _pad_episodes([
                e.z_W_ddot if e.z_W_ddot is not None else np.array([]) for e in eps
            ])
        if any(e.reward is not None and len(e.reward) > 0 for e in eps):
            arrays['reward'] = _pad_episodes([
                e.reward if e.reward is not None else np.array([]) for e in eps
            ])
        if any(e.s_pos is not None and len(e.s_pos) > 0 for e in eps):
            arrays['s_pos'] = _pad_episodes([
                e.s_pos if e.s_pos is not None else np.array([]) for e in eps
            ])
        if any(e.susp_defl is not None and len(e.susp_defl) > 0 for e in eps):
            arrays['susp_defl'] = _pad_episodes([
                e.susp_defl if e.susp_defl is not None else np.array([]) for e in eps
            ])
        if any(e.tire_defl is not None and len(e.tire_defl) > 0 for e in eps):
            arrays['tire_defl'] = _pad_episodes([
                e.tire_defl if e.tire_defl is not None else np.array([]) for e in eps
            ])

        arrays['episode_return'] = np.array([e.episode_return for e in eps])
        arrays['rms_accel']      = np.array([e.rms_accel      for e in eps])
        arrays['comfort_score']  = np.array([e.comfort_score  for e in eps])
        arrays['solve_ms_mean']  = np.array([e.solve_ms_mean  for e in eps])
        arrays['cpu_ms_mean']    = np.array([e.cpu_ms_mean    for e in eps])
        arrays['n_params']       = np.array([e.n_params       for e in eps])
        arrays['n_steps']        = np.array([e.n_steps        for e in eps], dtype=np.float64)

        if self.bumps:
            arrays['bump_x0']     = np.array([b[0] for b in self.bumps], dtype=np.float64)
            arrays['bump_height'] = np.array([b[1] for b in self.bumps], dtype=np.float64)
            arrays['bump_length'] = np.array([b[2] for b in self.bumps], dtype=np.float64)

        meta_num = {
            'meta_dt':         float(dt),
            'meta_v_max_kmh':  float(self.v_max_kmh),
            'meta_a_comfort':  float(self.a_comfort),
            'meta_a_limit':    float(self.a_limit),
            'meta_n_episodes': float(n),
        }
        meta_str = {
            'meta_method':  self.method,
            'meta_road':    self.road,
            'meta_run_id':  self.run_id,
        }

        saved: dict[str, Path] = {}
        fmts = _save_formats()

        if 'npz' in fmts:
            npz_path = self.save_dir / 'run.npz'
            np.savez(str(npz_path), **arrays, **meta_num)
            saved['npz'] = npz_path

        if 'mat' in fmts:
            try:
                from scipy.io import savemat
                mat_dict = dict(arrays)
                mat_dict.update(meta_num)
                mat_dict.update(meta_str)
                mat_path = self.save_dir / 'run.mat'
                savemat(str(mat_path), mat_dict, do_compression=True)
                saved['mat'] = mat_path
            except ImportError:
                pass

        if 'json' not in fmts:
            return saved
        info = {
            'metadata': {**meta_str, **meta_num},
            'summary': {
                'mean_return':  float(np.mean([e.episode_return for e in eps])),
                'std_return':   float(np.std( [e.episode_return for e in eps])),
                'max_return':   float(np.max( [e.episode_return for e in eps])),
                'min_return':   float(np.min( [e.episode_return for e in eps])),
                'mean_rms_accel': float(np.mean([e.rms_accel for e in eps])),
                'mean_comfort':   float(np.mean([e.comfort_score for e in eps])),
                'mean_solve_ms':  float(np.nanmean([e.solve_ms_mean for e in eps])),
                'mean_cpu_ms':    float(np.nanmean([e.cpu_ms_mean for e in eps])),
                'n_params':       float(np.nanmean([e.n_params for e in eps])),
            },
            'arrays': {
                k: {'shape': list(v.shape), 'unit': _UNITS.get(k, '?')}
                for k, v in arrays.items()
            },
        }
        json_path = self.save_dir / 'run_info.json'
        json_path.write_text(json.dumps(info, indent=2))
        saved['json'] = json_path

        return saved


_UNITS: dict[str, str] = {
    't':              's',
    'v_kmh':          'km/h',
    'v_init_kmh':      'km/h',
    'z_B_ddot':       'm/s²',
    'z_W_ddot':       'm/s²',
    'action':         '[-1, 1]',
    'reward':         'reward',
    's_pos':          'm',
    'susp_defl':      'm',
    'tire_defl':      'm',
    'episode_return': 'reward',
    'rms_accel':      'm/s²',
    'comfort_score':  '[0-1]',
    'solve_ms_mean':  'ms',
    'cpu_ms_mean':    'ms',
    'n_params':       'count',
    'n_steps':        'steps',
    'bump_x0':        'm',
    'bump_height':    'm',
    'bump_length':    'm',
}
