from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.paths import ROOT as _ROOT

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_env.common.wrappers.action_repeat import with_action_repeat
from gym_env.common.wrappers.decision import with_decisions
from gym_env.common import PreviewWrapper, TerminationWrapper
from gym_env.common.render.render import RenderQuit
from gym_env.vehicles import get_vehicle
from gym_env.common.utils.reward_utils import reward_bounds
from gym_env.common.reward.reward_params import load_reward_config
from gym_env.common.env_params import EPISODE_STEPS, DT
from utils.scenario_loader import load_scenario_any, enabled_eval_suites, scenario_title

from utils.gif import keep, save_gif, shrink
from eval.data_logger import RunLogger, EpisodeData
from utils.portable import load_vecnormalize


_ALGO_MAP: dict[str, type] = {"PPO": PPO}


def _render_nav(vehicle: str):
    return get_vehicle(vehicle).render().wait_for_episode_nav


_REWARD_TERM_KEYS = [
    "J_heave",
    "J_wheel",
    "J_long_acc",
    "J_long_jerk",
    "J_long_vel",
    "J_pitch",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained agent on the CALM environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo",
                   choices=list(_ALGO_MAP), required=True)
    p.add_argument("--vehicle", default="half", choices=("quarter", "half"),
                   help="Vehicle dynamics model (must match the trained model).")
    p.add_argument("--model_path", required=True,
                   help="Path to trained model .zip file.")
    p.add_argument("--scenario", default=None,
                   help="Fixed single eval scenario name or path")
    p.add_argument("--suites", nargs="+", metavar="FOLDER", default=None,
                   help="Suite folder names to run (default: enabled suites from eval_suites.yaml). "
                        "Example: --suites 01_single_bump 09_italy_full")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--save-plots", action="store_true",
                   help="Save matplotlib figures next to the JSON results.")
    p.add_argument("--save-gif", action="store_true",
                   help="Save an animated GIF for each episode.")
    p.add_argument("--log-data", metavar="DIR", default='data',
                   help="Save run data (.mat/.npz) to DIR/PPO/<road>/<exp_N>/. "
                        "Default root is <repo>/data/ when flag is given without value.",
                   nargs="?", const="__default__")
    return p.parse_args()


def _infer_vecnorm(model_path: Path) -> Path:
    if model_path.suffix == ".safetensors":
        for parent in [model_path.parent, *model_path.parents]:
            candidate = parent / "vecnormalize.json"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find vecnormalize.json for {model_path}.\n"
            "Place vecnormalize.json next to the model."
        )
    paired = model_path.with_name(f"{model_path.stem}_vec_normalize.pkl")
    if paired.exists():
        return paired
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "vecnormalize.pkl"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find vecnormalize.pkl for {model_path}.\n"
        "Place vecnormalize.pkl next to the model."
    )


def _build_env(vehicle: str, render_mode: str):
    env = gym.make(get_vehicle(vehicle).env_id, render_mode=render_mode,
                   random_road_on_reset=False)
    env = TerminationWrapper(env)
    return with_decisions(with_action_repeat(PreviewWrapper(env), record_inner=True), record_inner=True)


def load_model(algo: str, model_path: str, vecnorm_path: str | None = None):
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")

    vp = Path(vecnorm_path) if vecnorm_path else _infer_vecnorm(mp)
    if not vp.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vp}")

    if mp.suffix == ".safetensors":
        from utils.portable import load_policy
        from train.agent import load_algo_config

        env = DummyVecEnv([lambda: _build_env("half", "none")])
        model = load_policy(mp, env, load_algo_config(_ROOT / "config" / "agent" / "algo_configs.yaml")[algo.upper()]["policy_kwargs"])
        return model, vp

    cls   = _ALGO_MAP[algo.upper()]
    model = cls.load(str(mp))
    return model, vp


def _record(ep: dict, action: float, reward: float, info: dict) -> None:
    ep["rewards"].append(reward)
    ep["actions"].append(action)
    ep["speeds"].append(info.get("speed", 0.0))
    ep["v_inits"].append(info.get("v_init", 0.0))
    ep["body_accels"].append(info.get("z_B_ddot", 0.0))
    ep["wheel_accels"].append(info.get("z_W_ddot", 0.0))
    ep["pitch_accels"].append(info.get("phi_ddot", 0.0))
    ep["susp_defl"].append(info.get("susp_defl", 0.0))
    ep["tire_defl"].append(info.get("tire_defl", 0.0))
    ep["rms_accel_running"].append(info.get("rms_accel", 0.0))
    ep["comfort_score_running"].append(info.get("comfort_score", 0.0))
    ep["bumps_passed_running"].append(info.get("bumps_passed", 0))
    ep["bumps_total_running"].append(info.get("bumps_total", 0))
    ep["holes_passed_running"].append(info.get("holes_passed", 0))
    ep["holes_total_running"].append(info.get("holes_total", 0))
    ep["termination_reason_running"].append(info.get("termination_reason", ""))
    for k in _REWARD_TERM_KEYS:
        ep[k].append(info.get(k, 0.0))


def run_episode(
    model,
    vecnorm_path: Path,
    seed: int,
    render: bool,
    deterministic: bool,
    scenario_road=None,
    perception_track=None,
    capture_frames: bool = False,
    pause_at_end: bool = False,
    render_label: str | None = None,
    vehicle: str = "half",
) -> tuple[dict, list, str]:
    if render:
        render_mode = "human"
    elif capture_frames:
        render_mode = "rgb_array"
    else:
        render_mode = "none"

    venv = DummyVecEnv([lambda: _build_env(vehicle, render_mode)])
    venv = load_vecnormalize(vecnorm_path, venv)
    venv.training    = False
    venv.norm_reward = False

    if render_label is not None:
        venv.venv.envs[0].unwrapped.set_render_label(render_label)

    if hasattr(venv.venv.envs[0], "set_perception_track"):
        venv.venv.envs[0].set_perception_track(perception_track)

    reset_opts = {"road": scenario_road} if scenario_road is not None else None
    if reset_opts is not None:
        raw_obs_single, reset_info = venv.venv.envs[0].reset(seed=seed, options=reset_opts)
        raw_obs = np.expand_dims(raw_obs_single, axis=0)
        venv.ret = np.zeros(venv.num_envs)
        obs: np.ndarray = venv.normalize_obs(raw_obs)
    else:
        venv.seed(seed)
        reset_out       = venv.reset()
        obs: np.ndarray = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        reset_info      = venv.venv.reset_infos[0]

    done   = np.array([False])
    ep: dict = defaultdict(list)
    for sub in (reset_info or {}).get("inner", []):
        _record(ep, 0.0, sub["step_reward"], sub)
    frames: list = []
    nav_override = None

    while not done[0]:
        t0, c0 = time.perf_counter(), time.process_time()
        action, _ = model.predict(obs, deterministic=deterministic)
        ep["solve_ms"].append((time.perf_counter() - t0) * 1e3)
        ep["cpu_ms"].append((time.process_time() - c0) * 1e3)
        step_result = venv.step(action)
        obs    = step_result[0]
        reward = step_result[1]
        if len(step_result) == 5:
            done      = np.logical_or(step_result[2], step_result[3])
            info_list = step_result[4]
        else:
            done      = step_result[2]
            info_list = step_result[3]
        inner = info_list[0].get("inner")
        if inner:
            for sub in inner:
                _record(ep, float(np.asarray(action).ravel()[0]), sub["step_reward"], sub)
        else:
            _record(ep, float(np.asarray(action).ravel()[0]), float(reward[0]), info_list[0])

        if capture_frames:
            frame = venv.venv.envs[0].render()
            if frame is not None and keep(len(ep["speeds"])):
                frames.append(shrink(frame))

        if render_mode == 'human':
            base_env = venv.venv.envs[0].unwrapped
            nr = getattr(base_env, '_nav_request', None)
            if nr is not None:
                base_env._nav_request = None
                nav_override = 'replay' if nr in ('prev', 'replay') else 'next'
                ep["termination_reason_running"].append("skipped")
                break

    if nav_override is not None:
        nav = nav_override
    elif pause_at_end and render_mode == 'human':
        nav = _render_nav(vehicle)(venv.venv.envs[0])
    else:
        nav = 'next'

    venv.close()
    return dict(ep), frames, nav


def _count_params(model) -> float:
    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "parameters"):
        return 0.0
    try:
        return float(sum(p.numel() for p in policy.parameters()))
    except Exception:
        return 0.0


def episode_metrics(ep: dict) -> dict:
    if not ep["rewards"]:
        return {"n_steps": 0}
    rewards = np.asarray(ep["rewards"])
    accels  = np.asarray(ep["body_accels"])
    speeds  = np.asarray(ep["speeds"])
    v_inits  = np.asarray(ep["v_inits"])
    actions = np.asarray(ep["actions"])

    metrics = {
        "total_return":          float(rewards.sum()),
        "mean_step_reward":      float(rewards.mean()),
        "n_steps":               int(len(rewards)),
        "rms_accel":             float(np.sqrt(np.mean(accels ** 2))),
        "peak_accel":            float(np.max(np.abs(accels))),
        "rms_pitch_accel":       (float(np.sqrt(np.mean(np.asarray(ep["pitch_accels"]) ** 2)))
                                  if ep.get("pitch_accels") else 0.0),
        "speed_rmse":            float(np.sqrt(np.mean((speeds - v_inits) ** 2))) * 3.6,
        "mean_speed":            float(speeds.mean()) * 3.6,
        "comfort_score":         float(ep["comfort_score_running"][-1]),
        "action_rms":            float(np.sqrt(np.mean(actions ** 2))),
        "action_smoothness_rms": float(np.sqrt(np.mean(np.diff(actions) ** 2))) if len(actions) > 1 else 0.0,
        "bumps_passed":          int(ep["bumps_passed_running"][-1]) if ep["bumps_passed_running"] else 0,
        "bumps_total":           int(ep["bumps_total_running"][-1])  if ep["bumps_total_running"]  else 0,
        "holes_passed":          int(ep["holes_passed_running"][-1]) if ep["holes_passed_running"] else 0,
        "holes_total":           int(ep["holes_total_running"][-1])  if ep["holes_total_running"]  else 0,
        "termination_reason":    ep["termination_reason_running"][-1] if ep["termination_reason_running"] else "",
    }
    for k in _REWARD_TERM_KEYS:
        metrics[f"total_{k}"] = float(np.sum(ep.get(k, [0.0])))

    return metrics


def aggregate_episodes(all_metrics: list[dict]) -> dict:
    keys = list(all_metrics[0].keys())
    agg: dict = {"n_episodes": len(all_metrics)}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        if k == "termination_reason":
            counts: dict[str, int] = {}
            for v in vals:
                counts[v] = counts.get(v, 0) + 1
            agg[k] = counts
            continue
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
        }
    return agg


def print_episode_line(ep_i: int, n: int, m: dict) -> None:
    bumps_str = f'{m["bumps_passed"]}/{m["bumps_total"]}' if m.get("bumps_total") else "-"
    if m.get("holes_total"):
        bumps_str += f' H{m["holes_passed"]}/{m["holes_total"]}'
    print(
        f"  ep {ep_i+1:>3d}/{n}"
        f"  return={m['total_return']:+9.1f}"
        f"  rms_accel={m['rms_accel']:.3f} m/s²"
        f"  comfort={m['comfort_score']:.3f}"
        f"  bumps={bumps_str}"
        f"  speed_rmse={m['speed_rmse']:.2f} km/h"
        f"  end={m.get('termination_reason', '')}"
    )


def print_summary(agg: dict, bounds: dict, algo: str, vehicle: str = "half") -> None:
    W   = 30
    sep = "─" * 60

    print(f"\n{'═'*60}")
    print("  EVALUATION SUMMARY")
    print(f"  algo={algo}  episodes={agg['n_episodes']}")
    print(f"  reward bounds: [{bounds['episode_min']:.0f}, {bounds['episode_max']:.0f}]"
          f"   per-step: [{bounds['per_step_min']:.2f}, {bounds['per_step_max']:.1f}]")
    print(f"{'═'*60}")

    rows = [
        ("Episode return",        "total_return",           ""),
        ("Mean step reward",      "mean_step_reward",        ""),
        ("RMS body accel",        "rms_accel",               "m/s²"),
        ("Peak body accel",       "peak_accel",              "m/s²"),
        *( [("RMS pitch accel",   "rms_pitch_accel",         "rad/s²")] if vehicle == "half" else [] ),
        ("Speed tracking RMSE",   "speed_rmse",              "km/h"),
        ("Mean speed",            "mean_speed",              "km/h"),
        ("Comfort score",         "comfort_score",           "[0-1]"),
        ("Action smoothness RMS", "action_smoothness_rms",   ""),
    ]

    print(f"  {sep}")
    print(f"  {'Metric':<{W}}  {'mean ± std':>14}   {'min':>8}   {'max':>8}  unit")
    print(f"  {sep}")
    for label, key, unit in rows:
        d = agg[key]
        print(f"  {label:<{W}}  {d['mean']:>+9.3f} ±{d['std']:>6.3f}   {d['min']:>+8.3f}   {d['max']:>+8.3f}  {unit}")
    print(f"  {sep}")

    print("\n  Per-term reward totals (summed over steps, mean ± std across episodes):")
    print(f"  {sep}")
    for k in _REWARD_TERM_KEYS:
        fk = f"total_{k}"
        if fk in agg:
            d = agg[fk]
            print(f"  {k:<{W}}  {d['mean']:>+9.2f} ±{d['std']:>6.2f}")
    print(f"  {sep}")

    counts = agg.get("termination_reason", {})
    if counts:
        n = agg["n_episodes"]
        print(f"\n  Termination cause  ({n} episodes):")
        for reason, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {reason or '(unset)':<14}  {c:>4}  ({c/n:.0%})")
    print(f"  {sep}\n")


def plot_timeseries(ep: dict, ep_i: int, rcfg, bounds: dict, save_dir: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(ep["rewards"])) * DT

    fig = plt.figure(figsize=(13, 16), layout="constrained")
    gs  = fig.add_gridspec(7, 1)
    axes = [fig.add_subplot(gs[i]) for i in range(7)]
    ax_accel, ax_wheel, ax_speed, ax_rms, ax_action, ax_reward, ax_breakdown = axes

    ax_accel.plot(t, ep["body_accels"], color="#1565C0", lw=1.4, label="body accel z̈_B")
    ax_accel.axhspan( rcfg.a_comfort,  rcfg.a_limit,   alpha=0.06, color="orange")
    ax_accel.axhspan(-rcfg.a_limit,   -rcfg.a_comfort, alpha=0.06, color="orange")
    ax_accel.axhline( rcfg.a_comfort,  color="#F57C00", ls="--", lw=1.2,
                      label=f"comfort ±{rcfg.a_comfort} m/s²")
    ax_accel.axhline(-rcfg.a_comfort,  color="#F57C00", ls="--", lw=1.2)
    ax_accel.axhline( rcfg.a_limit,    color="#B71C1C", ls="--", lw=1.0,
                      label=f"limit ±{rcfg.a_limit} m/s²")
    ax_accel.axhline(-rcfg.a_limit,    color="#B71C1C", ls="--", lw=1.0)
    ax_accel.set_ylabel("body accel [m/s²]", fontsize=9)
    ax_accel.legend(fontsize=7, loc="upper right", ncol=2)

    ax_wheel.plot(t, ep["wheel_accels"], color="#00838F", lw=1.4, label="wheel accel z̈_W")
    ax_wheel.axhline( 30.0, color="#26C6DA", ls="--", lw=1.2,
                      label="comfort ±30 m/s²")
    ax_wheel.axhline(-30.0, color="#26C6DA", ls="--", lw=1.2)
    ax_wheel.set_ylabel("wheel accel [m/s²]", fontsize=9)
    ax_wheel.legend(fontsize=7, loc="upper right")

    ax_speed.plot(t, ep["speeds"], color="#1B5E20", lw=1.4, label="speed v")
    ax_speed.set_ylabel("speed [m/s]", fontsize=9)
    ax_speed.legend(fontsize=7, loc="upper right")

    ax_rms.plot(t, ep["rms_accel_running"], color="#6A1B9A", lw=1.4, label="running RMS accel")
    ax_rms.axhline(rcfg.a_comfort, color="#F57C00", ls="--", lw=1.2,
                   label=f"comfort ({rcfg.a_comfort} m/s²)")
    ax_rms.set_ylabel("RMS accel [m/s²]", fontsize=9)
    ax_rms.legend(fontsize=7, loc="upper right")

    ax_action.plot(t, ep["actions"], color="#E65100", lw=1.4, label="action u")
    ax_action.axhline(0, color="gray", ls=":", lw=0.8)
    ax_action.set_ylim(-1.15, 1.15)
    ax_action.set_ylabel("action [-1, 1]", fontsize=9)
    ax_action.legend(fontsize=7, loc="upper right")

    ax_reward.plot(t, ep["rewards"], color="#212121", lw=1.4, label="step reward")
    ax_reward.axhline(bounds["per_step_max"], color="green",   ls="--", lw=1.2,
                      label=f"step max ({bounds['per_step_max']:.1f})")
    ax_reward.axhline(bounds["per_step_min"], color="crimson", ls="--", lw=1.2,
                      label=f"step min ({bounds['per_step_min']:.2f})")
    ax_reward.axhline(0, color="gray", ls=":", lw=0.8)
    ax_reward.set_ylabel("step reward", fontsize=9)
    ax_reward.legend(fontsize=7, loc="upper right", ncol=2)

    term_colors = {
        "J_heave":     "#00897B",
        "J_wheel":     "#26A69A",
        "J_long_acc":  "#1E88E5",
        "J_long_jerk": "#FB8C00",
        "J_long_vel":  "#E53935",
        "J_pitch":     "#8E24AA",
    }
    for k, color in term_colors.items():
        vals = ep.get(k, [0.0] * len(t))
        if any(v != 0.0 for v in vals):
            ax_breakdown.plot(t, vals, color=color, lw=1.2, alpha=0.85, label=k)
    ax_breakdown.axhline(0, color="gray", ls=":", lw=0.8)
    ax_breakdown.set_ylabel("per-term reward", fontsize=9)
    ax_breakdown.set_xlabel("time [s]", fontsize=9)
    ax_breakdown.legend(fontsize=7, loc="lower right", ncol=3)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Episode {ep_i + 1}", fontsize=12, fontweight="bold")

    if save_dir:
        fname = save_dir / f"timeseries_ep{ep_i + 1}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"    saved → {fname.name}")
    plt.close(fig)


def plot_episode_comparison(all_metrics: list[dict], rcfg, bounds: dict,
                             save_dir: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_metrics)
    ep_labels = [f"ep{i+1}" for i in range(n)]
    x = np.arange(n)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Per-Episode Metrics Comparison", fontsize=12, fontweight="bold")

    def _bar(ax, vals, ylabel, title, ref_lines=None):
        ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", lw=0.5)
        if ref_lines:
            for val, label, color, ls in ref_lines:
                ax.axhline(val, color=color, ls=ls, lw=1.4, label=label, alpha=0.85)
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(ep_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    _bar(axes[0, 0],
         [m["total_return"]  for m in all_metrics], "episode return", "Episode Return",
         [(bounds["episode_max"], f"max={bounds['episode_max']:.0f}", "green", "--"),
          (0, "zero", "gray", ":")])
    _bar(axes[0, 1],
         [m["rms_accel"]     for m in all_metrics], "RMS accel [m/s²]", "RMS Body Acceleration",
         [(rcfg.a_comfort, f"comfort ({rcfg.a_comfort})", "#F57C00", "--"),
          (rcfg.a_limit,   f"limit ({rcfg.a_limit})",    "crimson",  "--")])
    _bar(axes[1, 0],
         [m["comfort_score"] for m in all_metrics], "comfort score [0-1]", "Comfort Score",
         [(1.0, "perfect=1.0", "green", "--")])
    _bar(axes[1, 1],
         [m["speed_rmse"]    for m in all_metrics], "speed RMSE [m/s]", "Speed Tracking RMSE",
         [(0, "perfect=0", "green", ":")])

    fig.tight_layout()
    if save_dir:
        fname = save_dir / "episode_comparison.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"    saved → {fname.name}")
    plt.close(fig)


def plot_reward_breakdown(all_metrics: list[dict], save_dir: Path | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_metrics)
    ep_labels = [f"ep{i+1}" for i in range(n)]
    x = np.arange(n)

    terms  = [k for k in _REWARD_TERM_KEYS
               if any(m.get(f"total_{k}", 0) != 0 for m in all_metrics)]
    colors = ["#00897B", "#26A69A", "#1E88E5", "#E53935", "#FB8C00", "#8E24AA", "#00ACC1"]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5))
    fig.suptitle("Reward Term Breakdown (total per episode)", fontsize=12, fontweight="bold")

    pos_bottom = np.zeros(n)
    neg_bottom = np.zeros(n)

    for i, k in enumerate(terms):
        vals     = np.array([m.get(f"total_{k}", 0.0) for m in all_metrics])
        pos_vals = np.where(vals >= 0, vals, 0)
        neg_vals = np.where(vals <  0, vals, 0)
        color    = colors[i % len(colors)]
        ax.bar(x, pos_vals, bottom=pos_bottom, label=k, color=color, alpha=0.82, edgecolor="white")
        ax.bar(x, neg_vals, bottom=neg_bottom,            color=color, alpha=0.82, edgecolor="white")
        pos_bottom += pos_vals
        neg_bottom += neg_vals

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ep_labels, fontsize=9)
    ax.set_ylabel("summed reward contribution", fontsize=9)
    ax.legend(fontsize=8, loc="lower right", ncol=len(terms))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_dir:
        fname = save_dir / "reward_breakdown.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"    saved → {fname.name}")
    plt.close(fig)


def save_json(agg: dict, all_metrics: list[dict], algo: str,
              model_path: str, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"eval_{algo}_{ts}.json"

    export = {
        "timestamp":   ts,
        "algo":        algo,
        "model_path":  str(model_path),
        "aggregate":   agg,
        "per_episode": all_metrics,
    }

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)

    print(f"  JSON saved → {out_path}")


def _run_one_scenario(
    sc_name: str,
    model,
    vecnorm_path: Path,
    args,
    deterministic: bool,
    bounds: dict,
    rcfg,
    save_dir: Path,
    ep_index: int,
    log_root: Path | None = None,
) -> tuple[str, str, float, dict] | None:
    while True:
        scenario_road, perception_track, speed_ms, sc_desc = load_scenario_any(sc_name)
        ep, frames, nav = run_episode(
            model, vecnorm_path,
            seed=args.seed,
            render=args.render,
            deterministic=deterministic,
            scenario_road=scenario_road,
            perception_track=perception_track,
            capture_frames=args.save_gif,
            pause_at_end=args.render,
            render_label=scenario_title(sc_name, str(args.algo).upper()),
            vehicle=args.vehicle,
        )
        if nav != 'replay':
            break
    if nav == 'quit':
        return None

    m = episode_metrics(ep)
    if m["n_steps"] == 0:
        return sc_name, sc_desc, speed_ms, m
    bumps_str = f'{m["bumps_passed"]}/{m["bumps_total"]}' if m.get("bumps_total") else "-"
    if m.get("holes_total"):
        bumps_str += f' H{m["holes_passed"]}/{m["holes_total"]}'
    print(
        f"  [{sc_name}]"
        f"  return={m['total_return']:+9.1f}"
        f"  rms={m['rms_accel']:.3f} m/s²"
        f"  speed_err={m['speed_rmse']:.1f} km/h"
        f"  bumps={bumps_str}"
        f"  v={speed_ms*3.6:.0f} km/h"
        f"  end={m.get('termination_reason', '')}"
    )

    if args.save_gif and frames:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_gif(frames, save_dir / f"{sc_name}.gif")
    if args.save_plots:
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_timeseries(ep, ep_index, rcfg, bounds, save_dir)

    if log_root is not None:
        logger = RunLogger(
            method=args.algo, road=sc_name, out_root=log_root,
            dt=DT, v_max_kmh=rcfg.v_limit * 3.6,
            a_comfort=rcfg.a_comfort, a_limit=rcfg.a_limit,
            run_subdir=False,
            bumps=getattr(scenario_road, '_bumps', None),
        )
        logger.add(EpisodeData(
            v=ep["speeds"], v_init=ep["v_inits"],
            z_B_ddot=ep["body_accels"], z_W_ddot=ep["wheel_accels"],
            susp_defl=ep["susp_defl"], tire_defl=ep["tire_defl"],
            action=ep["actions"], reward=ep["rewards"],
            episode_return=m["total_return"],
            rms_accel=m["rms_accel"],
            comfort_score=m["comfort_score"],
            solve_ms_mean=float(np.mean(ep["solve_ms"])) if ep.get("solve_ms") else float("nan"),
            cpu_ms_mean=float(np.mean(ep["cpu_ms"])) if ep.get("cpu_ms") else float("nan"),
            n_params=_count_params(model),
        ))
        logger.save()

    return sc_name, sc_desc, speed_ms, m


def _print_suite_table(suite_name: str, rows: list[tuple[str, str, float, dict]],
                       bounds: dict, vehicle: str = "half") -> None:
    W = 24
    half = vehicle == "half"
    sep = "─" * (78 if half else 66)
    pitch_hdr = f"  {'Pitch r/s²':>10}" if half else ""
    print(f"\n  ── {suite_name} ──")
    print(f"  {sep}")
    print(f"  {'Scenario':<{W}}  {'Return':>8}  {'RMS m/s²':>8}{pitch_hdr}  {'SpeedErr':>8}  {'Comfort':>7}  Bumps  End")
    print(f"  {sep}")
    returns = []
    for sc_name, _, speed_ms, m in rows:
        bumps_str = f'{m["bumps_passed"]}/{m["bumps_total"]}' if m.get("bumps_total") else "-"
        if m.get("holes_total"):
            bumps_str += f' H{m["holes_passed"]}/{m["holes_total"]}'
        pitch_col = f"  {m.get('rms_pitch_accel', 0.0):>10.3f}" if half else ""
        print(
            f"  {sc_name:<{W}}  {m['total_return']:>+8.1f}"
            f"  {m['rms_accel']:>8.3f}{pitch_col}"
            f"  {m['speed_rmse']:>7.1f} k"
            f"  {m['comfort_score']:>7.3f}"
            f"  {bumps_str}"
            f"  {m.get('termination_reason', '')}"
        )
        returns.append(m["total_return"])
    print(f"  {sep}")
    if returns:
        print(f"  {'Mean':<{W}}  {float(np.mean(returns)):>+8.1f}")
    print(f"  {sep}")


def _run_suite(args, model, vecnorm_path: Path, deterministic: bool,
               bounds: dict, rcfg, exp_tag: str | None) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = _ROOT / "logs" / "eval" / "results" / f"eval_{args.algo}_suite_{exp_tag or 'run'}"
    log_root = (
        _ROOT / "data"
        if args.log_data == "__default__"
        else Path(args.log_data) if args.log_data
        else None
    )

    suites = enabled_eval_suites(args.suites)
    if not suites:
        print("No scenarios found. Enable a suite in config/eval/eval_suites.yaml "
              "(incl. `real`), or pass --suites <folder|real>.")
        return

    total_scenarios = sum(len(s) for _, s in suites)

    print(f"\n{'═'*60}")
    print(f"  CALM EVAL SUITE - {args.algo}")
    print(f"  model: {args.model_path}")
    print(f"  suites: {[n for n, _ in suites]}")
    print(f"  {total_scenarios} scenarios  deterministic={deterministic}")
    print(f"{'═'*60}")

    all_rows: list[tuple[str, str, float, dict]] = []
    export_suites: list[dict] = []
    ep_index = 0
    aborted = False

    for suite_name, scenarios in suites:
        suite_rows: list[tuple[str, str, float, dict]] = []
        print(f"\n  {suite_name}  ({len(scenarios)} scenarios)")

        for sc_name in scenarios:
            result = _run_one_scenario(
                sc_name, model, vecnorm_path, args,
                deterministic, bounds, rcfg, save_dir / suite_name, ep_index,
                log_root=log_root,
            )
            if result is None:
                aborted = True
                break
            if result[3]["n_steps"] == 0:
                continue
            suite_rows.append(result)
            all_rows.append(result)
            ep_index += 1

        _print_suite_table(suite_name, suite_rows, bounds, args.vehicle)
        export_suites.append({
            "suite": suite_name,
            "scenarios": [
                {"scenario": sc, "description": desc,
                 "v_ref_kmh": round(v * 3.6, 1), **m}
                for sc, desc, v, m in suite_rows
            ],
        })
        if aborted:
            break

    if all_rows:
        all_returns = [m["total_return"] for _, _, _, m in all_rows]
        all_rms     = [m["rms_accel"]    for _, _, _, m in all_rows]
        print(f"\n{'═'*60}")
        print(f"  OVERALL  {len(all_rows)} scenarios"
              f"  mean_return={float(np.mean(all_returns)):+.1f}"
              f"  mean_rms={float(np.mean(all_rms)):.3f} m/s²")
        print(f"{'═'*60}")

    export = {
        "timestamp":  ts,
        "algo":       args.algo,
        "model_path": str(args.model_path),
        "suites":     export_suites,
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"suite_{args.algo}_{ts}.json"
    out.write_text(json.dumps(export, indent=2))
    print(f"  JSON saved → {out}")
    if log_root:
        print(f"  data logged → {log_root}/{args.algo}/")


def main() -> None:
    args = parse_args()
    deterministic = True

    rcfg   = load_reward_config()
    bounds = reward_bounds(rcfg, EPISODE_STEPS)

    _m = re.search(r'(exp_\d+)', str(args.model_path))
    exp_tag = _m.group(1) if _m else None

    model, vecnorm_path = load_model(args.algo, args.model_path)

    if not args.scenario:
        _run_suite(args, model, vecnorm_path, deterministic, bounds, rcfg, exp_tag)
        return

    scenario_road  = None
    scenario_track = None
    scenario_label = 'speed_bump'
    if args.scenario:
        scenario_road, scenario_track, _, sc_desc = load_scenario_any(args.scenario)
        sc_name        = args.scenario
        scenario_label = Path(args.scenario).stem

    save_dir = _ROOT / "logs" / "eval" / "results" / f"eval_{args.algo}_{scenario_label}_{exp_tag or 'run'}"

    print(f"\n{'═'*60}")
    print("  CALM EVAL")
    print(f"  algo={args.algo}  road={scenario_label}  episodes={args.n_episodes}")
    if args.scenario:
        print(f"  scenario: {sc_name} - {sc_desc}")
    print(f"  model: {args.model_path}")
    print(f"  deterministic={deterministic}  render={args.render}  save_plots={args.save_plots}  save_gif={args.save_gif}")
    print(f"{'═'*60}\n")

    log_root = (
        _ROOT / "data"
        if args.log_data == "__default__"
        else Path(args.log_data) if args.log_data
        else None
    )
    run_logger: RunLogger | None = (
        RunLogger(
            method=args.algo,
            road=scenario_label,
            out_root=log_root,
            dt=DT,
            v_max_kmh=rcfg.v_limit * 3.6,
            a_comfort=rcfg.a_comfort,
            a_limit=rcfg.a_limit,
            run_id=exp_tag,
            run_subdir=False,
            bumps=getattr(scenario_road, '_bumps', None),
        )
        if log_root is not None else None
    )

    all_eps:     list[dict] = []
    all_metrics: list[dict] = []

    ep_i = 0
    while ep_i < args.n_episodes:
        ep, frames, nav = run_episode(
            model, vecnorm_path,
            seed=args.seed + ep_i,
            render=args.render,
            deterministic=deterministic,
            scenario_road=scenario_road,
            perception_track=scenario_track,
            capture_frames=args.save_gif,
            pause_at_end=args.render,
            vehicle=args.vehicle,
        )
        if nav == 'replay':
            continue
        m = episode_metrics(ep)
        if m["n_steps"] == 0:
            if nav == 'quit':
                break
            ep_i += 1
            continue
        all_eps.append(ep)
        all_metrics.append(m)
        print_episode_line(ep_i, args.n_episodes, m)
        ep_i += 1
        if nav == 'quit':
            break

        if run_logger is not None:
            run_logger.add(EpisodeData(
                v=ep["speeds"],
                v_init=ep["v_inits"],
                z_B_ddot=ep["body_accels"],
                z_W_ddot=ep["wheel_accels"],
                susp_defl=ep["susp_defl"],
                tire_defl=ep["tire_defl"],
                action=ep["actions"],
                reward=ep["rewards"],
                episode_return=m["total_return"],
                rms_accel=m["rms_accel"],
                comfort_score=m["comfort_score"],
                solve_ms_mean=float(np.mean(ep["solve_ms"])) if ep.get("solve_ms") else float("nan"),
                cpu_ms_mean=float(np.mean(ep["cpu_ms"])) if ep.get("cpu_ms") else float("nan"),
                n_params=_count_params(model),
            ))

        if args.save_gif and frames:
            save_dir.mkdir(parents=True, exist_ok=True)
            gif_path = save_dir / f"ep{ep_i + 1}.gif"
            save_gif(frames, gif_path)

        if args.save_plots:
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_timeseries(ep, ep_i, rcfg, bounds, save_dir)

    agg = aggregate_episodes(all_metrics)
    print_summary(agg, bounds, args.algo, args.vehicle)

    save_dir.mkdir(parents=True, exist_ok=True)
    save_json(agg, all_metrics, args.algo, args.model_path, save_dir)

    if run_logger is not None:
        saved = run_logger.save()
        print(f"  data logged → {run_logger.save_dir}")
        if "mat" not in saved:
            print("  (install scipy for .mat output: pip install scipy)")

    if args.save_plots:
        print("\n  Generating summary figures ...")
        plot_episode_comparison(all_metrics, rcfg, bounds, save_dir)
        plot_reward_breakdown(all_metrics, save_dir)
        print(f"  All plots saved to {save_dir}")
    else:
        print("  (pass --save-plots to generate figures)")


if __name__ == "__main__":
    try:
        main()
    except RenderQuit:
        print("\n  closed")
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
