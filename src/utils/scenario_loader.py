from __future__ import annotations

import json
import re

import numpy as np
from pathlib import Path

from utils.paths import ROOT
from typing import Optional

import yaml


def _load_catalog() -> dict[int, dict]:
    root = ROOT
    p = root / 'config' / 'catalog' / 'speed_bumps.json'
    data = json.loads(p.read_text())
    return {b['id']: b for b in data['bumps']}


def list_scenarios() -> list[str]:
    root = ROOT
    seen: set[str] = set()
    for p in (root / 'config' / 'eval' / 'scenarios').rglob('*.yaml'):
        if p.name != 'manifest.yaml':
            seen.add(p.stem)
    dataset_dir = root / 'config' / 'eval' / 'dataset'
    if dataset_dir.is_dir():
        for p in dataset_dir.rglob('*.yaml'):
            seen.add(p.stem)
    return sorted(seen)


def _load_eval_suites_cfg() -> dict:
    root = ROOT
    p = root / 'config' / 'eval' / 'eval_suites.yaml'
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _find_scenario_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists():
        return path
    root = ROOT
    scenarios_dir = root / 'config' / 'eval' / 'scenarios'
    candidate = scenarios_dir / f'{name_or_path}.yaml'
    if candidate.exists():
        return candidate
    matches = [p for p in scenarios_dir.rglob(f'{name_or_path}.yaml')
               if p.name != 'manifest.yaml']
    if matches:
        return matches[0]
    dataset_dir = root / 'config' / 'eval' / 'dataset'
    if dataset_dir.is_dir():
        matches = list(dataset_dir.rglob(f'{name_or_path}.yaml'))
        if matches:
            return matches[0]
    available = list_scenarios()
    raise FileNotFoundError(
        f"Scenario '{name_or_path}' not found. Available: {available}"
    )


def load_scenario(
    name_or_path: str,
    include_holes: bool = True,
) -> tuple[list, float, str, str]:
    path = _find_scenario_path(name_or_path)

    with open(path) as fh:
        cfg = yaml.safe_load(fh)

    catalog  = _load_catalog()
    raw_list = cfg.get('road_obstacles') or cfg.get('bumps') or []
    obstacles: list[tuple[float, float, float]] = []
    for entry in raw_list:
        x_start = float(entry['x_start_m'])
        if 'catalog_id' in entry:
            cat = catalog[int(entry['catalog_id'])]
            h   = float(cat['height_m'])
            w   = float(cat['width_m'])
        else:
            h = float(entry['height_m'])
            w = float(entry['width_m'])
        if not include_holes and h < 0:
            continue
        obstacles.append((x_start, h, w))

    speed_m_s = float(cfg['vehicle_speed_kmh']) / 3.6
    return obstacles, speed_m_s, cfg['name'], cfg.get('description', '')


def _prettify_scenario_name(name: str) -> str:
    pretty = str(name).replace('_', ' ').strip()
    return pretty.rstrip('0123456789').strip() or pretty


def _real_label(name: str) -> str:
    m = re.fullmatch(r'(.+)_sb0*(\d+)', str(name))
    if not m:
        return _prettify_scenario_name(name)
    return f'{m.group(1).replace("_", " ").title()} SB {m.group(2)}'


def scenario_title(name: str, method: str = '', index: int | None = None, total: int | None = None) -> str:
    base = _real_label(name) if re.fullmatch(r'.+_sb0*\d+', str(name)) else str(name).replace('_', ' ')
    title = f'{method}: {base}' if method else base
    if index is not None and total:
        title += f'   ({index} of {total})'
    return title


def _real_scenarios_dir() -> Path:
    return ROOT / 'config' / 'eval' / 'scenarios' / 'real'


REAL_SUITES = ('real', 'real_train')


def list_real_scenarios(kind: str = 'real') -> list[str]:
    d = _real_scenarios_dir().parent / kind
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob('*.json') if p.name != 'manifest.json')


def load_scenario_any(name_or_path: str):
    real = next((f for f in (_real_scenarios_dir().parent / k / f'{name_or_path}.json'
                             for k in REAL_SUITES) if f.is_file()), _real_scenarios_dir() / f'{name_or_path}.json')
    p = Path(name_or_path)
    if real.is_file() or (p.suffix == '.json' and p.is_file()):
        road, track = load_scenario_json(real if real.is_file() else p)
        return road, track, road.speed, 'real dataset'
    bumps, speed, name, desc = load_scenario(name_or_path)
    return make_road_generator(name_or_path), None, speed, desc


def _load_trim_cfg() -> dict:
    p = ROOT / 'config' / 'gym_env' / 'scenario_trim.yaml'
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def edge_trim_indices(z, ds: float, bumps: list, cfg: dict) -> tuple[int, int]:
    z = np.asarray(z, dtype=float)
    n = len(z)
    if not cfg.get('trim_edges', True) or not bumps or n == 0:
        return 0, n
    tol = float(cfg.get('edge_tol_m', 0.01))
    margin = float(cfg.get('bump_margin_m', 0.5))
    lo = min(float(b['s_start_m']) for b in bumps)
    hi = max(float(b['s_end_m']) for b in bumps)
    pos = np.arange(n) * ds
    level = np.abs(z) <= tol

    start = 0
    if not level[0]:
        ok = np.flatnonzero(level & (pos <= lo - margin))
        start = int(ok[0]) if len(ok) else 0
    end = n
    if not level[-1]:
        ok = np.flatnonzero(level & (pos >= hi + margin))
        end = int(ok[-1]) + 1 if len(ok) else n
    return start, end


def load_scenario_json(path):
    from road.road_generator import RoadGenerator

    d = json.loads(Path(path).read_text())
    r = d["road"]
    ds = float(r["ds_m"])
    start, end = edge_trim_indices(r["z"], ds, d.get("bumps") or [], _load_trim_cfg())
    prior = float((d.get("meta") or {}).get("trim", {}).get("offset_m", 0.0))
    offset = start * ds
    z = r["z"][start:end]
    s_bump = float(d["bumps"][0]["s_bump_m"]) - offset if d.get("bumps") else d.get("s_bump_m")
    gen = RoadGenerator.from_profile(z, ds, s_bump=s_bump)
    gen._trim_offset_m = prior + offset
    gen._bumps = [(float(b["s_bump_m"]) - offset, float(b.get("peak_z_m", 0.0)),
                   float(b.get("width_m", 1.0))) for b in (d.get("bumps") or [])]
    label = _real_label(d["name"]) if d.get("name") else _prettify_scenario_name(d.get("source_sequence", ""))
    if label:
        gen._bump_labels = [label] * max(len(gen._bumps), 1)
    track = d.get("perception")
    return gen, track


def enabled_eval_suites(selected: Optional[list] = None) -> list[tuple[str, list[str]]]:
    cfg = _load_eval_suites_cfg().get("suites", {})
    names = selected if selected else [k for k, v in cfg.items()
                                       if isinstance(v, dict) and v.get("enabled")]
    scen_root = ROOT / 'config' / 'eval' / 'scenarios'
    out: list[tuple[str, list[str]]] = []
    for name in names:
        if name in REAL_SUITES:
            reals = list_real_scenarios(name)
            if reals:
                out.append((name, reals))
        elif name == "real_all":
            reals = sorted(set(list_real_scenarios('real')) | set(list_real_scenarios('real_train')))
            if reals:
                out.append(("real_all", reals))
        else:
            d = scen_root / name
            if d.is_dir():
                out.append((name, sorted(p.stem for p in d.glob("*.yaml"))))
    return out


def make_road_generator(name_or_path: str):
    from road.road_generator import RoadGenerator

    bumps, speed, name, _ = load_scenario(name_or_path)
    gen = RoadGenerator(vehicle_speed=speed)
    gen._bumps = bumps
    base = _prettify_scenario_name(name)
    if base:
        gen._bump_labels = [
            base if len(bumps) == 1 else f'{base} {i + 1}'
            for i in range(len(bumps))
        ]
    return gen
