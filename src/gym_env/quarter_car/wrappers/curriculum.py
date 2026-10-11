from __future__ import annotations

import json
import numpy as np
import gymnasium as gym
import yaml
from pathlib import Path
from typing import Optional

from road.road_generator import RoadGenerator
from utils.paths import ROOT



def _choose_real(rng, real_fraction: float) -> bool:
    return bool(rng.random() < real_fraction)


class CurriculumWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, config: dict, n_envs: int = 1):
        super().__init__(env)
        self._levels              = config["levels"]
        self._max_level           = len(self._levels) - 1
        self._active_level: int   = 0
        self._forced_level: Optional[int] = None
        self._rng                 = np.random.default_rng()
        self._scenarios: dict[int, list[dict]] = self._load_scenarios(config)

        self._real_fraction = float(config.get("real_fraction", 0.0))
        self._real_by_level: dict[int, list[Path]] = {i: [] for i in range(len(self._levels))}
        rsd = config.get("real_scenarios_dir")
        if rsd and Path(rsd).is_dir():
            self._load_real_scenarios(Path(rsd))

    def _load_real_scenarios(self, rsd: Path) -> None:
        tier_to_level = {self._levels[i].get("difficulty"): i
                         for i in range(len(self._levels))}
        cur = rsd / "curriculum.json"
        graded = False
        if cur.is_file():
            for e in json.loads(cur.read_text()).get("scenarios", []):
                lvl = tier_to_level.get(e.get("difficulty"))
                p = rsd / e.get("file", "")
                if lvl is not None and p.is_file():
                    self._real_by_level[lvl].append(p)
                    graded = True
        if not graded:
            skip = {"manifest.json", "curriculum.json"}
            self._real_by_level[0] = sorted(p for p in rsd.glob("*.json")
                                            if p.name not in skip)

    def _preview_wrapper(self):
        e = self.env
        while e is not None:
            if hasattr(e, "set_perception_track"):
                return e
            e = getattr(e, "env", None)
        return None

    def set_level(self, level: int) -> None:
        self._active_level = min(max(self._active_level, int(level)), self._max_level)

    def set_forced_level(self, level: Optional[int]) -> None:
        self._forced_level = level if level is None else min(int(level), self._max_level)

    @property
    def current_level(self) -> int:
        if self._forced_level is not None:
            return self._forced_level
        return self._active_level

    @staticmethod
    def _load_pool(d: Path) -> list[dict]:
        if not d.is_dir():
            return []
        pool = []
        for f in sorted(d.glob("scenario_*.yaml")):
            with open(f) as fh:
                pool.append(yaml.safe_load(fh))
        return pool

    @classmethod
    def _load_scenarios(cls, config: dict) -> dict[int, list[dict]]:
        levels    = config["levels"]
        scenarios: dict[int, list[dict]] = {i: [] for i in range(len(levels))}

        sd_str = config.get("scenarios_dir")
        if not sd_str:
            return scenarios
        sd = Path(sd_str)
        if not sd.is_dir():
            return scenarios

        for lvl_idx, lvl_cfg in levels.items():
            diff = lvl_cfg.get("difficulty")
            scenarios[lvl_idx] = cls._load_pool(sd / "single" / diff if diff else sd / "single")

        return scenarios

    def reset(self, **kwargs):
        preview = self._preview_wrapper()

        real_pool = [p for lvl in range(self.current_level + 1)
                     for p in self._real_by_level.get(lvl, [])]
        if real_pool and _choose_real(self._rng, self._real_fraction):
            from utils.scenario_loader import load_scenario_json
            path = real_pool[int(self._rng.integers(0, len(real_pool)))]
            road, track = load_scenario_json(path)
            if preview is not None:
                preview.set_perception_track(track)
            opts = kwargs.get("options") or {}
            opts["road"] = road
            kwargs["options"] = opts
            return super().reset(**kwargs)

        if preview is not None:
            preview.set_perception_track(None)

        lvl  = self.current_level
        pool = self._scenarios.get(lvl, [])
        if not pool:
            opts = kwargs.get("options") or {}
            opts.setdefault("randomize_road", True)
            kwargs["options"] = opts
            return super().reset(**kwargs)

        sc = pool[int(self._rng.integers(0, len(pool)))]
        if "obstacles" in sc:
            obstacles = [(o["height_m"], o["width_m"]) for o in sc["obstacles"]]
            gaps      = sc.get("gaps_m", [])
        else:
            obstacles = [(sc["height_m"], sc["width_m"])]
            gaps      = []

        road = RoadGenerator.from_scenario(
            vehicle_speed = sc["speed_kmh"] / 3.6,
            flat_start    = sc["flat_start_m"],
            obstacles     = obstacles,
            gaps          = gaps,
        )

        opts = kwargs.get("options") or {}
        opts["road"] = road
        kwargs["options"] = opts
        return super().reset(**kwargs)


def load_curriculum_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    raw["levels"] = {int(k): v for k, v in raw["levels"].items()}
    if "advance_reward_threshold" in raw:
        raw["advance_reward_threshold"] = {
            int(k): float(v)
            for k, v in raw["advance_reward_threshold"].items()
        }
    if "advance_time_threshold" in raw:
        raw["advance_time_threshold"] = {
            int(k): int(v)
            for k, v in raw["advance_time_threshold"].items()
        }
    raw.setdefault("mode", "reward")
    if "scenarios_dir" in raw:
        sd = raw["scenarios_dir"]
        if not Path(sd).is_absolute():
            raw["scenarios_dir"] = str(ROOT / sd)
    if raw.get("real_scenarios_dir"):
        rsd = raw["real_scenarios_dir"]
        if not Path(rsd).is_absolute():
            raw["real_scenarios_dir"] = str(ROOT / rsd)
    raw.setdefault("real_fraction", 0.0)
    return raw
