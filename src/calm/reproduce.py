from __future__ import annotations

import argparse
import subprocess
import sys

from utils.paths import LOGS, ROOT


def _run(label: str, module: str, args: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, '-m', module, *args]
    print(f'\n== {label}\n   {" ".join(cmd[1:])}', flush=True)
    if dry_run:
        return
    if subprocess.call(cmd, cwd=ROOT) != 0:
        raise SystemExit(f'step failed: {label}')


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog='calm reproduce',
                                 description='train several seeds and score each one with the MPC and the human drive '
                                             '(needs the real scenarios and CALM_SEQUENCES)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[69, 70, 71, 72, 73])
    ap.add_argument('--timesteps', type=int, default=None, help='training steps per seed (default: algo_configs.yaml)')
    ap.add_argument('--prefix', default='repro', help='run names are <prefix>_s<seed>')
    ap.add_argument('--suites', nargs='+', default=['real_all'])
    ap.add_argument('--skip-train', action='store_true', help='score existing runs only')
    ap.add_argument('--dry-run', action='store_true', help='print the steps without running them')
    args = ap.parse_args(argv)

    runs_dir = LOGS / 'comfort' / 'runs'
    names = {seed: f'{args.prefix}_s{seed}' for seed in args.seeds}
    first = names[args.seeds[0]]

    if not (ROOT / 'config' / 'train' / 'scenarios' / 'curriculum').is_dir():
        _run('generate the training scenarios', 'utils.generate_train_scenarios', [], args.dry_run)

    for seed, name in names.items():
        model = LOGS / 'models' / 'PPO' / name / 'PPO_final.zip'
        if args.skip_train or model.is_file():
            print(f'\n== train {name}: using the existing model')
            continue
        extra = ['--timesteps', str(args.timesteps)] if args.timesteps else []
        _run(f'train {name}', 'train.train', ['--algo', 'PPO', '--seed', str(seed), '--run-name', name, *extra], args.dry_run)

    runs_dir.mkdir(parents=True, exist_ok=True)
    for seed, name in names.items():
        model = LOGS / 'models' / 'PPO' / name / 'PPO_final.zip'
        is_reference = name == first
        score_args = ['--ppo-model', str(model), '--suites', *args.suites, '--out', str(runs_dir / f'{name}.csv')]
        if is_reference:
            score_args += ['--mpc']
        _run(f'score {name}' + (' with the MPC reference' if is_reference else ''), 'eval.comfort_score', score_args, args.dry_run)
    return 0
