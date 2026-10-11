from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

SCENARIOS_REPO = 'CALM-Dataset/CALM-Scenarios'
REGIONS = ['italy', 'germany']


def _root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'config').is_dir():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml and config/ not found")


def _real_dir() -> Path:
    return _root() / 'config' / 'train' / 'scenarios' / 'real'


def _eval_dir() -> Path:
    return _root() / 'config' / 'eval' / 'scenarios'


def real_files() -> list[tuple[Path, str]]:
    seen = {p.name: (p, 'seen') for p in sorted(_real_dir().glob('*.json')) if p.name not in ('manifest.json', 'curriculum.json')}
    held = {p.name: (p, 'unseen') for p in sorted((_eval_dir() / 'real').glob('*.json')) if p.name != 'manifest.json'}
    return sorted({**seen, **held}.values(), key=lambda t: t[0].name)


ITALY_LOCATIONS = ('carmiano', 'salice_salentino', 'nardo', 'monteroni_di_lecce')


def region_of(name: str) -> str:
    return 'italy' if name.startswith(ITALY_LOCATIONS) else 'germany'


def index_path() -> Path:
    return _root() / 'config' / 'dataset' / 'index.json'


def _source_of(name: str) -> str:
    m = re.match(r'[A-Za-z]+', name)
    return m.group(0).lower() if m else 'unknown'


def _real_rows() -> list[dict]:
    rows: list[dict] = []
    for p, split in real_files():
        d = json.loads(p.read_text())
        bumps = d.get('bumps') or []
        peak = max((float(b.get('peak_z_m', 0.0)) for b in bumps), default=0.0)
        rows.append({
            'id': d.get('name', p.stem),
            'group': 'real',
            'split': split,
            'source': _source_of(d.get('name', p.stem)),
            'n_bumps': len(bumps),
            'peak_z_m': round(peak, 4),
            'road_length_m': float(d.get('meta', {}).get('road_length_m', 0.0)),
            'has_perception': 'perception' in d,
            'rec_speed_mpc_kmh': None,
            'rec_speed_ppo_kmh': None,
        })
    return rows


def _eval_rows() -> list[dict]:
    rows: list[dict] = []
    for p in sorted(_eval_dir().rglob('*.yaml')):
        if p.name == 'manifest.yaml':
            continue
        d = yaml.safe_load(p.read_text()) or {}
        obstacles = d.get('road_obstacles') or d.get('bumps') or []
        rows.append({
            'id': d.get('name', p.stem),
            'group': 'eval',
            'suite': p.parent.name,
            'source': str(d.get('source', 'unknown')),
            'n_bumps': len(obstacles),
            'vehicle_speed_kmh': float(d.get('vehicle_speed_kmh', 0.0)),
        })
    return rows


def build_index(write: bool = False) -> list[dict]:
    rows = _real_rows() + _eval_rows()
    if write:
        out = index_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2))
    return rows


def load_index() -> list[dict]:
    p = index_path()
    if p.is_file():
        return json.loads(p.read_text())
    return build_index()


def ensure_local() -> None:
    train_dir, eval_dir = _real_dir(), _eval_dir() / 'real'
    seen_dir = _eval_dir() / 'real_train'
    if all(d.is_dir() and any(d.glob('*.json')) for d in (train_dir, eval_dir, seen_dir)):
        return
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download

    for d in (_real_dir(), _eval_dir() / 'real', seen_dir):
        d.mkdir(parents=True, exist_ok=True)
    perception_dir = _root() / 'config' / 'perception'
    perception_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(repo_id=SCENARIOS_REPO, repo_type='dataset', local_dir=tmp)
        for region in REGIONS:
            for split, dsts in (('train', (_real_dir(), seen_dir)), ('eval', (_eval_dir() / 'real',))):
                for f in Path(tmp, region, split).glob('*.json'):
                    # low_confidence means the road-height sensors found no real,
                    # trustworthy localized bump near the camera-flagged crossing
                    # (painted marking, unresolved texture, bad detection range) -
                    # training/evaluating on it means modeling a bump that may not
                    # physically exist as labeled (see CALM-Scenarios README)
                    bumps = json.loads(f.read_text()).get('bumps') or []
                    if bumps and bumps[0].get('low_confidence'):
                        skipped += 1
                        continue
                    for dst in dsts:
                        shutil.copy2(f, dst / f.name)
            model = Path(tmp, region, 'perception_model.yaml')
            if model.is_file() and not (perception_dir / 'perception_model.yaml').is_file():
                shutil.copy2(model, perception_dir / 'perception_model.yaml')
    if skipped:
        print(f"skipped {skipped} low_confidence scenario(s)")


def main() -> None:
    rows = build_index(write=True)
    print(f"wrote {index_path()} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
