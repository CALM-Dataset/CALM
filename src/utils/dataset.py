from __future__ import annotations

import json
from pathlib import Path

import yaml

from utils.paths import ROOT

SCENARIOS_REPO = 'CALM-Dataset/CALM-Scenarios'
REGIONS = ['italy', 'germany']


def _real_dir() -> Path:
    return ROOT / 'config' / 'train' / 'scenarios' / 'real'


def _eval_dir() -> Path:
    return ROOT / 'config' / 'eval' / 'scenarios'


def assign_split(stem: str, folder_split: str | None, eval_names: set[str] | None) -> str:
    if eval_names is None:
        return folder_split
    return 'eval' if stem in eval_names else 'train'


def _split_list(split_file: Path, key: str) -> list[str] | None:
    if not split_file.is_file():
        return None
    d = yaml.safe_load(split_file.read_text()) or {}
    return d.get(key) or (d.get('paper_split') or {}).get(key)


def split_eval_names(split_file: Path) -> set[str] | None:
    names = _split_list(split_file, 'eval')
    return set(names) if names else None


def split_exclude_names(split_file: Path) -> set[str]:
    return set(_split_list(split_file, 'exclude') or [])


def _region_files(region_dir: Path) -> list[tuple[Path, str | None]]:
    folders = [s for s in ('train', 'eval') if (region_dir / s).is_dir()]
    if folders:
        return [(f, s) for s in folders for f in sorted((region_dir / s).glob('*.json'))]
    return [(f, None) for f in sorted((region_dir / 'crossings').glob('*.json'))]


def place_snapshot(src: Path, train_dirs: tuple[Path, ...], eval_dirs: tuple[Path, ...], perception_file: Path) -> int:
    import shutil

    eval_names = split_eval_names(src / 'split.yaml')
    excluded = split_exclude_names(src / 'split.yaml')
    both = sorted(excluded & (eval_names or set()))
    if both:
        raise SystemExit(f"{SCENARIOS_REPO} split.yaml lists {', '.join(both)} under both eval and exclude")
    files = [item for region in REGIONS for item in _region_files(src / region) if item[0].stem not in excluded]
    if eval_names is None and any(folder_split is None for _, folder_split in files):
        raise SystemExit(
            f"{SCENARIOS_REPO} holds the scenarios without train/eval folders, "
            "but its split.yaml has no eval list naming the evaluation roads"
        )
    targets = {'train': train_dirs, 'eval': eval_dirs}
    for d in (*train_dirs, *eval_dirs):
        d.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for f, folder_split in files:
        bumps = json.loads(f.read_text()).get('bumps') or []
        if bumps and bumps[0].get('low_confidence'):
            skipped += 1
            continue
        for dst in targets[assign_split(f.stem, folder_split, eval_names)]:
            shutil.copy2(f, dst / f.name)
    models = [src / 'perception_model_de_it.yaml', src / 'germany' / 'perception_model' / 'perception_model_de.yaml', src / 'italy' / 'perception_model' / 'perception_model_it.yaml']
    model = next((m for m in models if m.is_file()), None)
    if model is not None and not perception_file.is_file():
        perception_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, perception_file)
    return skipped


def ensure_local() -> None:
    train_dir, eval_dir = _real_dir(), _eval_dir() / 'real'
    seen_dir = _eval_dir() / 'real_train'
    if all(d.is_dir() and any(d.glob('*.json')) for d in (train_dir, eval_dir, seen_dir)):
        return
    import tempfile

    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory() as tmp:
        try:
            snapshot_download(repo_id=SCENARIOS_REPO, repo_type='dataset', local_dir=tmp)
        except Exception as exc:
            from huggingface_hub.utils import HfHubHTTPError
            if isinstance(exc, HfHubHTTPError):
                raise SystemExit(
                    f"cannot download {SCENARIOS_REPO} ({type(exc).__name__}). The dataset is gated: "
                    "request access on its Hugging Face page, then run `hf auth login` with a token that can read it."
                ) from None
            raise
        skipped = place_snapshot(Path(tmp), (train_dir, seen_dir), (eval_dir,),
                                 ROOT / 'config' / 'perception' / 'perception_model.yaml')
    if skipped:
        print(f"skipped {skipped} low_confidence scenario(s)")

