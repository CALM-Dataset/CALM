from __future__ import annotations

from pathlib import Path


def find_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'pyproject.toml').is_file() and (parent / 'config').is_dir():
            return parent
    raise FileNotFoundError('repo root with pyproject.toml and config/ not found')


ROOT = find_root()
CONFIG = ROOT / 'config'
LOGS = ROOT / 'logs'
