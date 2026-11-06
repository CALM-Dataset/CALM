from __future__ import annotations

import importlib
import os
import shutil
import sys

from utils.paths import ROOT

OK, WARN, FAIL = 'ok', 'warn', 'FAIL'


def _package(name: str, hint: str, required: bool = True) -> tuple[str, str, str]:
    try:
        module = importlib.import_module(name)
        return OK, name, getattr(module, '__version__', '')
    except Exception as exc:
        return (FAIL if required else WARN), name, f'{type(exc).__name__}: {exc}. {hint}'


def _count(path, pattern='*.json') -> int:
    return len(list(path.glob(pattern))) if path.is_dir() else 0


def _python() -> list[tuple[str, str, str]]:
    ok = sys.version_info >= (3, 10)
    return [(OK if ok else FAIL, 'python', f'{sys.version.split()[0]}' + ('' if ok else ' (need 3.10 or newer)'))]


def _packages() -> list[tuple[str, str, str]]:
    fix = 'run: just install'
    rows = [_package(n, fix) for n in ('numpy', 'scipy', 'gymnasium', 'stable_baselines3', 'torch', 'numba', 'yaml', 'matplotlib')]
    rows.append(_package('pygame', 'only needed for the render music', required=False))
    rows.append(_package('huggingface_hub', fix))
    try:
        import torch
        rows.append((OK, 'torch device', 'cuda available (Stable-Baselines3 picks the device automatically)' if torch.cuda.is_available() else 'cpu'))
    except Exception:
        pass
    return rows


def _acados() -> list[tuple[str, str, str]]:
    lib = ROOT / 'acados' / 'lib'
    rows = []
    if not (ROOT / 'acados' / 'CMakeLists.txt').is_file():
        return [(WARN, 'acados', 'submodule not checked out: git submodule update --init --recursive (only the MPC needs it)')]
    rows.append((OK if lib.is_dir() else WARN, 'acados library', str(lib) if lib.is_dir() else 'not built: run just build-acados (only the MPC needs it)'))
    rows.append(_package('acados_template', 'run just build-acados', required=False))
    renderer = ROOT / 'acados' / 'bin' / 't_renderer'
    rows.append((OK if renderer.is_file() else WARN, 'acados t_renderer',
                 'present' if renderer.is_file() else 'downloaded on the first MPC run (needs network once)'))
    return rows


def _data() -> list[tuple[str, str, str]]:
    suites = ROOT / 'config' / 'eval' / 'scenarios'
    synthetic = sum(_count(d, '*.yaml') for d in suites.glob('[0-9]*') if d.is_dir())
    rows = [(OK if synthetic else FAIL, 'synthetic eval scenarios', f'{synthetic} files' if synthetic else 'missing: they ship with the repo, check git status')]
    curriculum = _count(ROOT / 'config' / 'train' / 'scenarios' / 'curriculum' / 'single' / 'easy', '*.yaml')
    rows.append((OK if curriculum else WARN, 'training scenarios', f'{curriculum} easy' if curriculum else 'not generated: run just gen-scene'))
    real_train = _count(ROOT / 'config' / 'train' / 'scenarios' / 'real')
    real_seen = _count(suites / 'real_train')
    real_held = _count(suites / 'real')
    have = real_train and real_seen and real_held
    rows.append((OK if have else WARN, 'real scenarios',
                 f'{real_train} train, {real_seen} seen, {real_held} held out' if have
                 else 'missing: request access to CALM-Dataset/CALM-Scenarios, hf auth login, then calm fetch'))
    try:
        from huggingface_hub import get_token
        rows.append((OK if get_token() else WARN, 'huggingface login', 'token found' if get_token() else 'not logged in: hf auth login (needed for calm fetch)'))
    except Exception:
        pass
    seq = [p for p in os.environ.get('CALM_SEQUENCES', '').split(':') if p]
    good = [p for p in seq if os.path.isdir(p)]
    rows.append((OK if good else WARN, 'CALM_SEQUENCES', f'{len(good)} root(s)' if good
                 else 'unset: only needed for the human profile in calm compare'))
    return rows


def _tools() -> list[tuple[str, str, str]]:
    in_env = sys.prefix != sys.base_prefix or (ROOT / '.venv').is_dir() or os.path.exists('/.dockerenv')
    return [(OK if shutil.which('just') else WARN, 'just', shutil.which('just') or 'optional: every recipe is also a calm command'),
            (OK if in_env else WARN, 'virtual environment', sys.prefix if in_env else 'not in one: just install creates .venv')]


def run_checks(data_only: bool = False) -> int:
    sections = [('data', _data)] if data_only else [
        ('python', _python), ('packages', _packages), ('acados (MPC)', _acados), ('data', _data), ('tools', _tools)]
    failed = 0
    for title, fn in sections:
        print(f'\n{title}')
        for status, name, detail in fn():
            print(f'  [{status:>4}] {name:<26} {detail}')
            failed += status == FAIL
    print('\n' + ('everything required is in place' if not failed else f'{failed} required item(s) missing, see the hints above'))
    return 1 if failed else 0
