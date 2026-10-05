from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecNormalize


class MetricsCallback(BaseCallback):

    def _on_step(self) -> bool:
        infos: list[dict] = self.locals.get("infos", [])

        comfort, rms_a, peak_a      = [], [], []
        spd_kmh, spd_err   = [], []
        ep_reward                   = []
        pitch_a                     = []
        J_heave, J_pitch, J_speed, J_jerk, J_floor, J_total = [], [], [], [], [], []
        term_reasons: list[str] = []

        for info in infos:
            if "comfort_score"  in info: comfort.append(info["comfort_score"])
            if "rms_accel"      in info: rms_a.append(info["rms_accel"])
            if "peak_accel"     in info: peak_a.append(info["peak_accel"])
            if "speed_kmh"      in info: spd_kmh.append(info["speed_kmh"])
            if "speed_error"    in info: spd_err.append(abs(info["speed_error"]))
            if "episode_reward" in info: ep_reward.append(info["episode_reward"])
            if "phi_ddot"       in info: pitch_a.append(abs(info["phi_ddot"]))
            if "J_heave"        in info: J_heave.append(info["J_heave"])
            if "J_pitch"        in info: J_pitch.append(info["J_pitch"])
            if "J_long_vel"     in info: J_speed.append(info["J_long_vel"])
            if "J_long_jerk"    in info: J_jerk.append(info["J_long_jerk"])
            if "J_floor"        in info: J_floor.append(info["J_floor"])
            if "J_total"        in info: J_total.append(info["J_total"])
            if info.get("termination_reason"): term_reasons.append(info["termination_reason"])

        log = self.logger.record
        if comfort:   log("env/comfort_score",  float(np.mean(comfort)))
        if rms_a:     log("env/rms_accel",  float(np.mean(rms_a)))
        if peak_a:    log("env/peak_accel", float(np.mean(peak_a)))
        if pitch_a:   log("env/pitch_accel", float(np.mean(pitch_a)))
        if spd_kmh:   log("env/speed_kmh",      float(np.mean(spd_kmh)))
        if spd_err:   log("env/speed_error", float(np.mean(spd_err)))
        if ep_reward: log("env/ep_reward",      float(np.mean(ep_reward)))

        if J_heave: log("reward/J_heave", float(np.mean(J_heave)))
        if J_pitch: log("reward/J_pitch", float(np.mean(J_pitch)))
        if J_speed: log("reward/J_speed", float(np.mean(J_speed)))
        if J_jerk:  log("reward/J_jerk",  float(np.mean(J_jerk)))
        if J_floor: log("reward/J_floor", float(np.mean(J_floor)))
        if J_total: log("reward/J_total", float(np.mean(J_total)))

        try:
            levels = self.training_env.get_attr("current_level")
            if levels:
                log("curriculum/level", float(levels[0]))
        except Exception:
            pass

        if term_reasons:
            n = len(term_reasons)
            for reason in ("completed", "timeout", "crash_susp", "crash_body", "crash_damper", "crash_speed"):
                log(f"termination/{reason}_frac", term_reasons.count(reason) / n)

        return True


class VecNormalizeSyncCallback(BaseCallback):

    def __init__(self, train_venv: VecNormalize, eval_venv: VecNormalize, verbose: int = 0):
        super().__init__(verbose)
        self._train = train_venv
        self._eval  = eval_venv

    def _on_step(self) -> bool:
        self._eval.obs_rms = self._train.obs_rms
        self._eval.ret_rms = self._train.ret_rms
        try:
            levels = self._train.venv.get_attr("current_level")
            level  = int(levels[0]) if levels else 0
            self._eval.venv.env_method("set_forced_level", level)
        except (AttributeError, TypeError, IndexError):
            pass
        return True


class PerformanceCurriculumCallback(EvalCallback):

    def __init__(
        self,
        eval_env,
        train_venv:     VecNormalize,
        curriculum_cfg: dict,
        start_level:    int = 0,
        **eval_kwargs,
    ):
        super().__init__(eval_env, **eval_kwargs)
        self._train_venv        = train_venv
        self._mode              = curriculum_cfg.get("mode", "reward")
        self._thresholds        = curriculum_cfg.get("advance_reward_threshold", {})
        self._time_thresholds   = curriculum_cfg.get("advance_time_threshold", {})
        self._window            = int(curriculum_cfg.get("advance_window", 3))
        self._min_dwell         = int(curriculum_cfg.get("advance_min_steps", 0))
        self._max_level         = len(curriculum_cfg["levels"]) - 1
        self._current_level = min(max(start_level, 0), self._max_level)
        self._start_level   = self._current_level

        self._level_returns:       dict[int, list[float]] = defaultdict(list)
        self._level_best_return:   dict[int, float]       = {}
        self._level_step_unlocked: dict[int, int]  = {self._current_level: 0}
        self._level_step_advanced: dict[int, int]  = {}


    def _on_training_start(self) -> None:
        super()._on_training_start()
        if self._start_level > 0:
            try:
                self._train_venv.venv.env_method("set_level", self._start_level)
            except Exception:
                pass
            try:
                self.eval_env.venv.env_method("set_forced_level", self._start_level)
            except Exception:
                pass


    def _on_step(self) -> bool:
        eval_ran = self.eval_freq > 0 and (self.n_calls % self.eval_freq) == 0
        result   = super()._on_step()
        if eval_ran and self.last_mean_reward > -np.inf:
            self._on_eval_result(float(self.last_mean_reward))

        if self._mode == "timesteps":
            time_threshold = self._time_thresholds.get(self._current_level, np.inf)
            if (self._current_level < self._max_level
                    and self.num_timesteps >= time_threshold):
                self._advance()

        return result


    def _on_eval_result(self, mean_return: float) -> None:
        level = self._current_level
        self._level_returns[level].append(mean_return)
        if getattr(self, "model", None) is not None:
            self.logger.record("curriculum/level",       float(level))
            self.logger.record("curriculum/mean_return", mean_return)

        if mean_return > self._level_best_return.get(level, -np.inf):
            self._level_best_return[level] = mean_return
            if getattr(self, "model", None) is not None and self.best_model_save_path is not None:
                level_path = Path(self.best_model_save_path) / f"level_{level}"
                level_path.mkdir(parents=True, exist_ok=True)
                self.model.save(str(level_path / "best_model"))
                if isinstance(self.eval_env, VecNormalize):
                    self.eval_env.save(str(level_path / "best_model_vec_normalize.pkl"))
                print(
                    f"\n  [Curriculum]  Level {level} best updated: {mean_return:+.2f}"
                    f"  → {level_path}/best_model\n"
                )

        threshold = float(self._thresholds.get(level, np.inf))
        window    = self._level_returns[level][-self._window:]

        dwell = self.num_timesteps - self._level_step_unlocked.get(level, 0)
        if (self._mode == "reward"
                and level < self._max_level
                and len(window) >= self._window
                and dwell >= self._min_dwell
                and float(np.mean(window)) >= threshold):
            self._advance()

    def _advance(self) -> None:
        prev = self._current_level
        self._current_level += 1
        self._level_step_advanced[prev]                   = self.num_timesteps
        self._level_step_unlocked[self._current_level]    = self.num_timesteps

        try:
            self._train_venv.venv.env_method("set_level", self._current_level)
        except Exception:
            pass

        recent = self._level_returns[prev][-self._window:]
        if self._mode == "timesteps":
            threshold_str = f"{self._time_thresholds.get(prev, '∞')} steps"
        else:
            threshold_str = str(self._thresholds.get(prev, '∞'))
        print(
            f"\n  [Curriculum]  Level {prev} → {self._current_level}  "
            f"window_mean={float(np.mean(recent)) if recent else float('nan'):+.1f}  "
            f"threshold={threshold_str}  "
            f"step={self.num_timesteps:,}\n"
        )


    def level_report(self) -> dict:
        report: dict = {}
        for level in sorted(self._level_returns):
            returns  = self._level_returns[level]
            advanced = self._level_step_advanced.get(level)
            report[str(level)] = {
                "n_evals":            len(returns),
                "mean_return":        round(float(np.mean(returns)), 2) if returns else None,
                "best_return":        round(float(np.max(returns)),  2) if returns else None,
                "worst_return":       round(float(np.min(returns)),  2) if returns else None,
                "step_unlocked":      self._level_step_unlocked.get(level, 0),
                "step_advanced":      advanced,
                "advance_threshold":  (
                    self._time_thresholds.get(level)
                    if self._mode == "timesteps"
                    else self._thresholds.get(level)
                ),
                "mastered":           advanced is not None,
            }
        return report

    def print_report(self) -> None:
        report = self.level_report()
        if not report:
            return
        print("\nCurriculum Level Performance")
        print(f"  {'Level':<7} {'Evals':>5}  {'Mean':>8}  {'Best':>8}  {'Threshold':>9}  Status")
        print(f"  {'-'*6}  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*22}")
        for level_str, s in report.items():
            thresh_str = f"{s['advance_threshold']:+.1f}" if s["advance_threshold"] is not None else "  final"
            if s["mastered"]:
                status = f"advanced at step {s['step_advanced']:,}"
            else:
                status = f"active  (unlocked {s['step_unlocked']:,})"
            print(
                f"  {level_str:<7} {s['n_evals']:>5}  "
                f"{s['mean_return']:>+8.1f}  "
                f"{s['best_return']:>+8.1f}  "
                f"{thresh_str:>9}  "
                f"{status}"
            )


class RealValidationCallback(BaseCallback):

    def __init__(
        self,
        real_venv:  VecNormalize,
        scenarios:  list,
        train_venv: VecNormalize,
        save_path:  Path,
        eval_freq:  int,
        deterministic: bool = True,
        verbose:    int = 1,
    ):
        super().__init__(verbose)
        self._venv          = real_venv
        self._scenarios     = scenarios
        self._train_venv    = train_venv
        self._save_path     = Path(save_path)
        self._eval_freq     = int(eval_freq)
        self._deterministic = deterministic
        self._best_return   = -np.inf
        self._save_path.mkdir(parents=True, exist_ok=True)

    def _rollout(self, road, track) -> float:
        base = self._venv.venv.envs[0]
        if hasattr(base, "set_perception_track"):
            base.set_perception_track(track)
        raw_obs, _ = base.reset(options={"road": road})
        self._venv.ret = np.zeros(self._venv.num_envs)
        obs = self._venv.normalize_obs(np.expand_dims(raw_obs, axis=0))
        done = False
        total = 0.0
        while not done:
            action, _ = self.model.predict(obs, deterministic=self._deterministic)
            step = self._venv.step(action)
            obs, reward = step[0], step[1]
            done = bool(step[2][0] if len(step) == 4 else np.logical_or(step[2], step[3])[0])
            info = (step[3] if len(step) == 4 else step[4])[0]
            total = info.get("episode_reward", total + float(reward[0]))
        return float(total)

    def _on_step(self) -> bool:
        if self._eval_freq <= 0 or self.n_calls % self._eval_freq != 0 or not self._scenarios:
            return True
        self._venv.obs_rms = self._train_venv.obs_rms
        returns = [self._rollout(road, track) for _, road, track in self._scenarios]
        mean_ret = float(np.mean(returns))
        self.logger.record("real_val/mean_return", mean_ret)
        self.logger.record("real_val/n_scenarios", len(returns))
        if mean_ret > self._best_return:
            self._best_return = mean_ret
            self.model.save(str(self._save_path / "best_model"))
            self._venv.save(str(self._save_path / "best_model_vec_normalize.pkl"))
            if self.verbose:
                print(f"\n  [RealVal]  new best real return {mean_ret:+.1f} "
                      f"over {len(returns)} scenarios → {self._save_path}/best_model\n")
        return True


def build_callbacks(
    model_dir:          Path,
    eval_venv:          VecNormalize,
    train_venv:         VecNormalize,
    eval_freq:          int,
    n_eval_episodes:    int,
    checkpoint_freq:    int,
    curriculum_cfg:     dict,
    curriculum_advance: bool = False,
    start_level:        int = 0,
) -> tuple[CallbackList, Optional[PerformanceCurriculumCallback]]:
    best_model_path = model_dir / "best"
    ckpt_path       = model_dir / "checkpoints"
    best_model_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    eval_kwargs = dict(
        best_model_save_path = str(best_model_path),
        log_path             = str(best_model_path),
        eval_freq            = eval_freq,
        n_eval_episodes      = n_eval_episodes,
        deterministic        = True,
        render               = False,
    )

    curriculum_cb: Optional[PerformanceCurriculumCallback] = None

    if curriculum_advance:
        curriculum_cb = PerformanceCurriculumCallback(
            eval_venv,
            train_venv     = train_venv,
            curriculum_cfg = curriculum_cfg,
            start_level    = start_level,
            **eval_kwargs,
        )
        eval_cb = curriculum_cb
    else:
        eval_cb = EvalCallback(eval_venv, **eval_kwargs)

    ckpt_cb     = CheckpointCallback(
        save_freq       = checkpoint_freq,
        save_path       = str(ckpt_path),
        name_prefix     = "ckpt",
        save_vecnormalize = True,
    )
    metrics_cb  = MetricsCallback()
    sync_cb     = VecNormalizeSyncCallback(train_venv, eval_venv)

    return CallbackList([eval_cb, ckpt_cb, metrics_cb, sync_cb]), curriculum_cb
