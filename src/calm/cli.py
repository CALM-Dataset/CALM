from __future__ import annotations

import subprocess
import sys

from utils.paths import ROOT

TOOLS = {
    'train': ('train.train', 'PPO training through the curriculum', ['--algo', 'PPO']),
    'tune': ('tune.tune', 'Optuna hyperparameter search', []),
    'mpc': ('baseline.mpc.mpc', 'MPC baseline (needs acados)', []),
    'reactive': ('baseline.reactive_driver.driver_eval', 'rule-based driver baseline', []),
    'gen-scenes': ('utils.generate_train_scenarios', 'procedural training scenarios', []),
}


def _run(module: str, args: list[str]) -> int:
    return subprocess.call([sys.executable, '-m', module, *args], cwd=ROOT)


def _model_path(run: str, ckpt: str) -> str:
    return str(ROOT / 'logs' / 'models' / 'PPO' / run / ckpt)


def _split_checkpoint(rest: list[str], default: str = 'PPO_final.zip') -> tuple[str, list[str]]:
    if rest and not rest[0].startswith('-'):
        return rest[0], rest[1:]
    return default, rest


def cmd_eval(rest: list[str]) -> int:
    if not rest or rest[0].startswith('-'):
        print('usage: calm eval <run> [checkpoint] [eval options]')
        return 2
    run, (ckpt, extra) = rest[0], _split_checkpoint(rest[1:])
    return _run('eval.eval', ['--algo', 'PPO', '--model_path', _model_path(run, ckpt), *extra])


def cmd_compare(rest: list[str]) -> int:
    if not rest or rest[0].startswith('-'):
        print('usage: calm compare <run> [checkpoint] [comfort_score options]')
        return 2
    run, (ckpt, extra) = rest[0], _split_checkpoint(rest[1:])
    return _run('eval.comfort_score', ['--ppo-model', _model_path(run, ckpt), '--mpc', *extra])


def cmd_fetch(rest: list[str]) -> int:
    from utils.dataset import ensure_local

    ensure_local()
    return cmd_doctor(['--data-only'])


def cmd_doctor(rest: list[str]) -> int:
    from calm.doctor import run_checks

    return run_checks(data_only='--data-only' in rest)


def cmd_reproduce(rest: list[str]) -> int:
    from calm.reproduce import main as reproduce

    return reproduce(rest)


def cmd_tb(rest: list[str]) -> int:
    return _run('tensorboard.main', ['--logdir', str(ROOT / 'logs' / 'tensorboard'), *rest])


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    special = {
        'eval': ('evaluate a trained run: calm eval <run> [checkpoint]', cmd_eval),
        'compare': ('ISO 2631 comfort of PPO, MPC and the recorded drive: calm compare <run>', cmd_compare),
        'fetch': ('download the gated real scenarios, then check what is present', cmd_fetch),
        'doctor': ('check Python, packages, acados, data and login, with hints', cmd_doctor),
        'reproduce': ('train several seeds and score each one (see calm reproduce --help)', cmd_reproduce),
        'tb': ('TensorBoard on logs/tensorboard', cmd_tb),
    }
    if not argv or argv[0] in ('-h', '--help'):
        print('usage: calm <command> [options]   (options after a command go to that command)\n')
        for name, (_, text, _) in TOOLS.items():
            print(f'  {name:<13}{text}')
        for name, (text, _) in special.items():
            print(f'  {name:<13}{text}')
        print('\nstart with: calm doctor')
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in special:
        return special[cmd][1](rest)
    if cmd in TOOLS:
        module, _, preset = TOOLS[cmd]
        return _run(module, [*preset, *rest])
    print(f'unknown command {cmd!r}; run calm --help')
    return 2
