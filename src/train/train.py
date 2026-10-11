from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]

from train.agent import build_model, load_algo_config, supported_algos
from stable_baselines3.common.callbacks import CallbackList
from train.monitoring import build_callbacks, RealValidationCallback, SpeedWeightController
from train.environment import (
    make_eval_vec_env, make_vec_env, restore_vecnormalize,
    make_real_val_env, load_real_val_scenarios,
)
from train.seed import seed_everything
from gym_env.quarter_car.config.reward_params import load_reward_config
from gym_env.quarter_car.config.preview_params import load_preview_config
from gym_env.quarter_car.config.env_params import EPISODE_STEPS
from gym_env.quarter_car.reward.utils import reward_bounds


def _vehicle_cfg(vehicle: str):
    if vehicle == "half":
        from gym_env.half_car.config.reward_params import load_reward_config as lrc
        from gym_env.half_car.config.preview_params import load_preview_config as lpc
        from gym_env.half_car.config.env_params import EPISODE_STEPS as es
        from gym_env.half_car.reward.utils import reward_bounds as rb
        return lrc, lpc, es, rb
    return load_reward_config, load_preview_config, EPISODE_STEPS, reward_bounds
from gym_env.quarter_car.wrappers.curriculum import load_curriculum_config


_CONFIG_PATH      = _ROOT / "config" / "agent" / "algo_configs.yaml"
_CURRICULUM_PATH  = _ROOT / "config" / "curriculum" / "curriculum.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a speed-planning RL agent on a quarter- or half-car environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
                    "--algo",
                    default="PPO",
                    choices=supported_algos())
    p.add_argument(
                    "--vehicle",
                    default="half",
                    choices=("quarter", "half"),
                    help="Vehicle dynamics model: quarter-car (2-DOF) or half-car (heave+pitch).")
    p.add_argument(
                    "--seed",
                    type=int,
                    default=69,
                    help="Random seed.")
    p.add_argument(
                    "--timesteps",
                    type=int,
                    default=None,
                    help="Total env steps.")
    p.add_argument(
                    "--n-envs",
                    type=int,
                    default=None,
                    help="Parallel training envs.")
    p.add_argument(
                    "--resume",
                    default=None,
                    help="Checkpoint .zip to continue training from.")
    p.add_argument(
                    "--run-name",
                    default=None,
                    help="Tag appended to the output directory name.")
    p.add_argument(
                    "--start-level",
                    type=int,
                    default=0,
                    help="Curriculum level to start on; if it is the top level the policy stays pinned there.")
    p.add_argument(
                    "-m", "--message",
                    default=None,
                    help="Free-text comment saved as comment.txt in the experiment directory.")
    p.add_argument(
                    "--sparse-reward",
                    action="store_true",
                    default=False,
                    help="Delay all per-step reward to episode end (TerminalRewardWrapper); episode return is unchanged.")

    return p.parse_args()


def _summarize_monitor(monitor_dir: Path) -> dict | None:
    files = sorted(monitor_dir.glob("*.monitor.csv"))
    if not files:
        sibling = monitor_dir.parent / f"{monitor_dir.name}.monitor.csv"
        if sibling.exists():
            files = [sibling]
    if not files:
        return None

    rewards: list[float] = []
    lengths: list[int] = []
    for file_path in files:
        with file_path.open(newline="") as handle:
            reader = csv.DictReader(row for row in handle if not row.startswith("#"))
            for row in reader:
                if "r" in row and "l" in row:
                    rewards.append(float(row["r"]))
                    lengths.append(int(row["l"]))
    if not rewards:
        return None

    return {
        "episodes":    len(rewards),
        "mean_reward": round(sum(rewards) / len(rewards), 3),
        "max_reward":  round(max(rewards), 3),
        "min_reward":  round(min(rewards), 3),
        "last_reward": round(rewards[-1], 3),
        "mean_ep_len": round(sum(lengths) / len(lengths), 1),
    }


def _eval_curve(best_dir: Path, snap_every: int = 100_000) -> dict | None:
    eval_file = best_dir / "evaluations.npz"
    if not eval_file.exists():
        return None
    with np.load(eval_file) as data:
        if "timesteps" not in data or "results" not in data:
            return None
        steps   = np.asarray(data["timesteps"]).reshape(-1)
        results = np.asarray(data["results"])

    if steps.size == 0:
        return None

    mean_r    = results.mean(axis=1)
    best_idx  = int(np.argmax(mean_r))

    max_step  = int(steps[-1])
    snaps = {}
    for target in range(snap_every, max_step + snap_every, snap_every):
        idx = np.searchsorted(steps, target, side="right") - 1
        if 0 <= idx < len(steps):
            snaps[str(int(target))] = round(float(mean_r[idx]), 2)

    return {
        "best_step":         int(steps[best_idx]),
        "best_mean_return":  round(float(mean_r[best_idx]), 2),
        "final_mean_return": round(float(mean_r[-1]), 2),
        "snapshots":         snaps,
    }


def _curriculum_summary(level_report: dict | None, curriculum_cfg: dict | None) -> dict | None:
    if not level_report or not curriculum_cfg:
        return None
    n_levels    = len(curriculum_cfg.get("levels", {}))
    final_level = max((int(k) for k in level_report), default=0)
    mastered    = [k for k, v in level_report.items() if v["mastered"]]
    return {
        "n_levels":    n_levels,
        "final_level": final_level,
        "levels_mastered": len(mastered),
        "per_level":   level_report,
    }


def _write_summary(
    path: Path,
    args,
    algo_kwargs: dict,
    timesteps: int,
    n_envs: int,
    seed: int,
    curriculum: bool,
    curriculum_cfg: dict | None,
    final_path: Path,
    best_path: Path,
    model_dir: Path,
    monitor_summary: dict | None,
    interrupted: bool,
    error: str | None,
    level_report: dict | None = None,
    eval_curve: dict | None = None,
) -> None:
    if eval_curve is None:
        eval_curve = _eval_curve(model_dir / "best")
    curriculum_info  = _curriculum_summary(level_report, curriculum_cfg if curriculum else None)

    payload = {
        "date":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status":  "INTERRUPTED" if interrupted else "ERROR" if error else "COMPLETED",
        "comment": getattr(args, "message", None),

        "config": {
            "algo":        args.algo,
            "road":        "speed_bump",
            "curriculum":  curriculum,
            "sparse_reward": args.sparse_reward,
            "timesteps":   timesteps,
            "n_envs":      n_envs,
            "seed":        seed,
            "norm_obs":    True,
            "norm_reward": True,
            "resume":      args.resume,
        },

        "hyperparameters": algo_kwargs,

        "paths": {
            "final_model": str(final_path) + ".zip",
            "best_model":  str(best_path)  + ".zip",
            "vecnorm":     str(final_path.parent / "vecnormalize.pkl"),
        },

        "eval_curve": eval_curve,

        "training_stats": monitor_summary,

        "curriculum": curriculum_info,

        "error_traceback": error,
    }

    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main() -> None:
    args = parse_args()

    full_cfg   = load_algo_config(_CONFIG_PATH)
    train_meta = full_cfg["training"]
    if args.algo in full_cfg:
        algo_kwargs = dict(full_cfg[args.algo])
    else:
        algo_kwargs = load_algo_config(
            _ROOT / "config" / "baseline" / f"{args.algo}.yaml"
        )

    seed      = args.seed      if args.seed      is not None else train_meta["seed"]
    timesteps = args.timesteps if args.timesteps is not None else train_meta["total_timesteps"]
    n_envs    = args.n_envs    if args.n_envs    is not None else train_meta["n_envs"]
    normalize = True

    seed_everything(seed)

    _load_rcfg, _load_pcfg, _EPISODE_STEPS, _reward_bounds = _vehicle_cfg(args.vehicle)
    rcfg           = _load_rcfg()
    curriculum_cfg = load_curriculum_config(_CURRICULUM_PATH)

    start_level = max(0, int(args.start_level))

    exp_root = _ROOT / "logs" / "models" / args.algo
    if args.run_name is None:
        exp_re = re.compile(r"^exp_(\d+)$")
        exp_ids = []
        if exp_root.exists():
            for path in exp_root.iterdir():
                if path.is_dir():
                    match = exp_re.match(path.name)
                    if match:
                        exp_ids.append(int(match.group(1)))
        next_id = (max(exp_ids) + 1) if exp_ids else 1
        run_tag = f"exp_{next_id}"
    else:
        run_tag = args.run_name

    model_dir = exp_root / run_tag
    tb_dir    = _ROOT / "logs" / "tensorboard" / run_tag
    mon_dir   = _ROOT / "logs" / "monitor" / run_tag
    best_path = model_dir / "best" / "best_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    mon_dir.mkdir(parents=True, exist_ok=True)

    if args.message:
        (model_dir / "comment.txt").write_text(args.message + "\n")

    gamma = algo_kwargs.get("gamma", 0.99)
    norm_reward = normalize and train_meta.get("norm_reward", True)

    _env_kwargs: dict = {"render_mode": "none"}
    train_venv = make_vec_env(
        n_envs=n_envs,
        base_seed=seed,
        monitor_dir=str(mon_dir / "train"),
        gamma=gamma,
        norm_obs=normalize,
        norm_reward=norm_reward,
        env_kwargs=_env_kwargs,
        curriculum_cfg=curriculum_cfg,
        sparse_reward=args.sparse_reward,
        vehicle=args.vehicle,
        action_penalty=float(train_meta.get("action_saturation_penalty", 0.0)),
    )
    if args.resume and normalize:
        vn_path = Path(args.resume).parent / "vecnormalize.pkl"
        if restore_vecnormalize(train_venv, str(vn_path)):
            print(f"  restored VecNormalize stats from {vn_path}")
    eval_venv = make_eval_vec_env(
        n_envs=train_meta["n_eval_envs"],
        base_seed=seed + 10_000,
        train_venv=train_venv,
        monitor_dir=str(mon_dir / "eval"),
        curriculum_cfg=curriculum_cfg,
        sparse_reward=args.sparse_reward,
        vehicle=args.vehicle,
    )

    model = build_model(
        algo=args.algo,
        venv=train_venv,
        algo_kwargs=algo_kwargs,
        tensorboard_log=str(tb_dir),
        seed=seed,
        resume=args.resume,
    )

    callbacks, curriculum_cb = build_callbacks(
        model_dir=model_dir,
        eval_venv=eval_venv,
        train_venv=train_venv,
        eval_freq=max(train_meta["eval_freq"] // n_envs, 1),
        n_eval_episodes=train_meta["n_eval_episodes"],
        checkpoint_freq=max(train_meta["checkpoint_freq"] // n_envs, 1),
        curriculum_cfg=curriculum_cfg,
        curriculum_advance=True,
        start_level=start_level,
    )

    real_val_scenarios = load_real_val_scenarios(curriculum_cfg.get("real_scenarios_dir"), n=20, seed=seed)
    if real_val_scenarios:
        real_val_venv = make_real_val_env(vehicle=args.vehicle, seed=seed + 20_000)
        real_val_cb = RealValidationCallback(
            real_venv  = real_val_venv,
            scenarios  = real_val_scenarios,
            train_venv = train_venv,
            save_path  = model_dir / "best_real",
            eval_freq  = max(2 * train_meta["eval_freq"] // n_envs, 1),
        )
        callbacks = CallbackList([callbacks, real_val_cb])
        print(f"  real-val   : {len(real_val_scenarios)} scenarios → best_real/best_model")

    ctrl = train_meta.get("speed_controller")
    if ctrl:
        callbacks = CallbackList([callbacks, SpeedWeightController(ctrl["target"], ctrl["init"], ctrl["gain"], ctrl["max"])])
        print(f"  speed weight controller: target speed ratio {ctrl['target']}")

    bounds = _reward_bounds(rcfg, _EPISODE_STEPS, n_bumps=5)

    print(f"\n{'─'*58}")
    print(f"  algo       : {args.algo}")
    print("  road       : speed_bump")
    print(f"  seed       : {seed}")
    print(f"  timesteps  : {timesteps:,}")
    print(f"  n_envs     : {n_envs}")
    print(f"  normalize  : obs={normalize}, reward={norm_reward}")
    print(f"  v_limit    : {rcfg.v_limit * 3.6:.0f} km/h")
    print(f"  v_min      : {rcfg.v_min * 3.6:.1f} km/h")
    pcfg = _load_pcfg()
    print(f"  preview    : {pcfg.n_peaks} peaks × 3 = {pcfg.n_peaks * 3} features over {pcfg.preview_distance}m")
    print(f"  curriculum : {len(curriculum_cfg['levels'])} levels, start={start_level}")
    print(f"  sparse     : {args.sparse_reward}")
    print(f"  output     : {model_dir}")
    if args.message:
        print(f"  comment    : {args.message}")
    print(f"  reward range  episode  [{bounds['episode_min']:+.0f},  {bounds['episode_max']:+.0f}]")
    print(f"                per-step [{bounds['per_step_min']:+.2f}, {bounds['per_step_max']:+.2f}]")
    print(f"{'─'*58}\n")

    _interrupted = False
    _error: str | None = None
    final_path = model_dir / f"{args.algo}_final"

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=(args.resume is None),
        )
    except KeyboardInterrupt:
        _interrupted = True
        print("\nInterrupted - saving checkpoint...")
    except Exception:
        _error = traceback.format_exc()
        print(f"\nTraining crashed:\n{_error}")
    finally:
        model.save(str(final_path))
        train_venv.save(str(model_dir / "vecnormalize.pkl"))
        train_venv.close()
        eval_venv.close()

    print(f"\nModel    → {final_path}.zip")
    print(f"VecNorm  → {model_dir / 'vecnormalize.pkl'}")
    print(f"TB logs  → tensorboard --logdir {tb_dir}")

    mon_summary  = _summarize_monitor(mon_dir / "train")
    eval_curve   = _eval_curve(model_dir / "best")
    best_step    = eval_curve["best_step"] if eval_curve else None
    level_report = curriculum_cb.level_report() if curriculum_cb is not None else None

    print(f"\nBest Model     → {best_path}.zip")
    print(f"Best Step      → {f'{best_step:,}' if best_step is not None else 'unknown'}")

    if eval_curve:
        print("\nEval return curve (mean over eval episodes):")
        for step_str, val in eval_curve["snapshots"].items():
            marker = " ← best" if int(step_str) == best_step else ""
            print(f"  {int(step_str):>10,} steps : {val:+.1f}{marker}")

    if mon_summary:
        emin, emax = bounds['episode_min'], bounds['episode_max']
        mean_r = mon_summary['mean_reward']
        max_r  = mon_summary['max_reward']
        min_r  = mon_summary['min_reward']

        def gap(v): return f"{emax - v:+.0f} below ceiling"

        print("\nTraining summary")
        print(f"  episodes     : {mon_summary['episodes']}")
        print(f"  mean_reward  : {mean_r:+.1f}   ({gap(mean_r)})")
        print(f"  max_reward   : {max_r:+.1f}   ({gap(max_r)})")
        print(f"  min_reward   : {min_r:+.1f}   ({gap(min_r)})")
        print(f"  last_reward  : {mon_summary['last_reward']:+.1f}")
        print(f"  mean_ep_len  : {mon_summary['mean_ep_len']:.1f}")
        print(f"\n  reward range  episode  [{emin:+.0f}, {emax:+.0f}]  (theoretical worst case)")
        print(f"                per-step [{bounds['per_step_min']:+.2f}, {bounds['per_step_max']:+.2f}]")
    else:
        print("\nTraining summary: no monitor data found.")

    if curriculum_cb is not None:
        curriculum_cb.print_report()

    summary_path = model_dir / "summary.json"
    _write_summary(
        path=summary_path,
        args=args,
        algo_kwargs=algo_kwargs,
        timesteps=timesteps,
        n_envs=n_envs,
        seed=seed,
        curriculum=True,
        curriculum_cfg=curriculum_cfg,
        final_path=final_path,
        best_path=best_path,
        model_dir=model_dir,
        monitor_summary=mon_summary,
        interrupted=_interrupted,
        error=_error,
        level_report=level_report,
        eval_curve=eval_curve,
    )
    print(f"Summary  → {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
