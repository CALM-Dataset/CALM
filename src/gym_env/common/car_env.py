from typing import Callable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from road.road_generator import RoadGenerator
from gym_env.common.reward.reward_params import RewardConfig, load_reward_config
from gym_env.common.reward.reward import compute_reward, compute_terminal_reward
from eval.iso2631 import WkStream
from gym_env.common.env_params import DT, EPISODE_STEPS, ACTION_MODE, SPEED_GAIN, SPEED_FLOOR_RATIO, DEPTH_OFFSET, DEPTH_SLOPE
from gym_env.common.render.render_params import (
    RENDER_Y_SCALE,
    RENDER_SHOW_TS,
    RENDER_TS_Z,
    RENDER_TS_Z_DDOT,
    RENDER_TS_SPEED,
    RENDER_FREEZE_EPISODE,
)
from gym_env.vehicles import get_vehicle


class CarEnv(gym.Env):
    VEHICLE = None

    metadata = {
        'render_modes': ['human', 'rgb_array', 'none'],
        'render_fps': int(round(1.0 / DT)),
    }

    def __init__(
        self,
        vehicle_speed: float = 6.0,
        render_mode: str = 'none',
        physics_params: Optional[dict] = None,
        road_params: Optional[dict] = None,
        reward_config: Optional[RewardConfig] = None,
        render_y_scale: int = RENDER_Y_SCALE,
        show_time_series: bool = RENDER_SHOW_TS,
        show_ts_z: bool = RENDER_TS_Z,
        show_ts_z_ddot: bool = RENDER_TS_Z_DDOT,
        show_ts_speed: bool = RENDER_TS_SPEED,
        freeze_episode: bool = RENDER_FREEZE_EPISODE,
        start_at_equilibrium: bool = True,
        ref_speed_profile: str = "constant",
        max_episode_steps: int = EPISODE_STEPS,
        random_road_on_reset: bool = True,
        road_override_kwargs: Optional[dict] = None,
        action_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode        = render_mode
        self._v0                = float(vehicle_speed)
        self._random_road_on_reset = bool(random_road_on_reset)
        self._y_scale           = int(render_y_scale)
        self._ref_speed_profile = ref_speed_profile
        self._max_episode_steps = int(max_episode_steps)
        self._start_at_eq       = bool(start_at_equilibrium)

        self._v_init_fn: Optional[Callable[[float], float]] = (road_params or {}).get('v_init_fn', None)

        self._vehicle = get_vehicle(self.VEHICLE)
        self._rcfg = reward_config or load_reward_config()
        self._action_mode = action_mode or ACTION_MODE

        bound = 8.0 if self._action_mode == "depth" else 1.0
        self.action_space = spaces.Box(
            low=np.array([-bound], dtype=np.float32),
            high=np.array([ bound], dtype=np.float32),
        )

        obs_low, obs_high = self._vehicle.obs_bounds()
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._ode  = self._vehicle.make_ode(physics_params or self._vehicle.default_plant_params())
        self._road = RoadGenerator(vehicle_speed, road_params)

        self._state          = self._ode.reset(self._v0)
        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._last_z_B_ddot  = 0.0
        self._last_z_W_ddot  = 0.0
        self._last_phi_ddot  = 0.0
        self._episode_reward = 0.0

        self._v              = self._v0
        self._v_episode_ref  = self._rcfg.v_limit
        self._v_init_last     = self._rcfg.v_limit
        self._speed_err_sq   = 0.0
        self._s_pos          = 0.0

        self._bumps_passed   = 0
        self._holes_passed   = 0
        self._obstacle_ends: list = []
        self._bumps_total    = 0
        self._holes_total    = 0
        self._last_bump_end: Optional[float] = None
        self._termination_reason = ""

        self._random_road_kwargs = {
            "catalog_ids":     [0, 1, 2, 3, 4],
            "flat_start":      60.0,
            "hole_fraction":   0.0,
        }
        if road_override_kwargs:
            self._random_road_kwargs.update(road_override_kwargs)
        self._v_random_low = self._rcfg.v_min * 2.0

        self._prev_a           = 0.0

        self._fig            = None
        self._ren_hist       = None
        self._render_label   = None
        self._episode_count  = 0
        self._show_ts        = bool(show_time_series)
        self._freeze_episode = bool(freeze_episode)
        self._freeze_render  = False
        self._ts_flags = {
            "z": bool(show_ts_z),
            "speed": bool(show_ts_speed),
            "z_ddot": bool(show_ts_z_ddot),
        }


    @property
    def hold_action(self) -> np.ndarray:
        value = -float(self.action_space.high[0]) if self._action_mode == "depth" else 0.0
        return np.array([value], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        cfg = self._rcfg
        opts = options or {}

        randomize       = opts.get("randomize_road",  self._random_road_on_reset)
        randomize_speed = opts.get("randomize_speed", True)

        if "road" in opts:
            self._road = opts["road"]
            self._v    = self._road.speed
        elif randomize:
            road_kwargs = opts.get("road_kwargs", self._random_road_kwargs)
            v_low  = float(opts.get("v_random_low",  self._v_random_low))
            v_high = float(opts.get("v_random_high", cfg.v_limit))
            v = float(rng.uniform(v_low, v_high)) if randomize_speed else self._v0
            self._road = RoadGenerator.from_random(
                rng, vehicle_speed=v, **road_kwargs
            )
            self._v = self._road.speed
        else:
            self._road.reset(seed=int(rng.integers(0, 2**31)))
            self._road.set_speed(self._v0)
            self._v = self._v0

        if "road" in opts:
            v_ref = self._v
        else:
            v_ref = float(self.np_random.uniform(cfg.v_init_low, cfg.v_limit))
        self._v = float(np.clip(self._v, cfg.v_hard_min, max(cfg.v_hard_min, v_ref)))
        self._road.set_speed(self._v)
        x = self._ode.reset(self._v)
        if not self._start_at_eq:
            self._ode.perturb(x, rng)
        self._state = x

        self._t              = 0.0
        self._step_count     = 0
        self._accel_sq       = 0.0
        self._peak_accel     = 0.0
        self._last_z_B_ddot  = 0.0
        self._last_z_W_ddot  = 0.0
        self._last_phi_ddot  = 0.0
        self._episode_reward = 0.0
        self._episode_count += 1

        if self._fig is not None and self.render_mode == 'human':
            if self._ren_hist is not None:
                for buf in self._ren_hist.values():
                    buf.clear()
        else:
            if self._fig is not None:
                self._vehicle.render().close_env(self)
            self._ren_hist = None

        self._v_episode_ref = v_ref
        self._v_init_last = self._v_episode_ref
        self._speed_err_sq   = 0.0
        self._s_pos          = 0.0

        self._bumps_passed   = 0
        self._holes_passed   = 0
        self._obstacle_ends  = sorted(
            (x0 + L + 1.0, h < 0) for (x0, h, L) in self._road._bumps
        )
        self._bumps_total    = sum(1 for (_, h, _) in self._road._bumps if h >= 0)
        self._holes_total    = sum(1 for (_, h, _) in self._road._bumps if h < 0)
        self._last_bump_end  = (
            float(self._road._bumps[-1][0] + self._road._bumps[-1][2])
            if self._road._bumps else None
        )
        prof = getattr(self._road, "_profile", None)
        self._road_end = float(prof[0][-1]) if prof is not None and len(prof[0]) else None
        self._termination_reason = ""

        self._prev_a           = 0.0
        self._wk               = WkStream(DT)

        self._freeze_render = False

        return self._obs(), self._info(0.0)

    def step(self, action):
        cfg = self._rcfg

        bound = float(self.action_space.high[0])
        u     = float(np.clip(action[0], -bound, bound))
        v_old = self._v
        v_top = max(cfg.v_hard_min, self._v_episode_ref)
        v_floor = max(cfg.v_hard_min, SPEED_FLOOR_RATIO * v_top) if self._action_mode in ("slowdown", "depth") else cfg.v_hard_min
        v_floor = min(v_floor, v_top)
        if self._action_mode == "slowdown":
            v_target = v_top + (v_top - v_floor) * min(u, 0.0)
            a_cmd = float(np.clip(SPEED_GAIN * (v_target - v_old), -cfg.a_max, cfg.a_max))
        elif self._action_mode == "depth":
            depth = 1.0 / (1.0 + float(np.exp(-DEPTH_SLOPE * (u - DEPTH_OFFSET))))
            v_target = v_top - (v_top - v_floor) * depth
            a_cmd = float(np.clip(SPEED_GAIN * (v_target - v_old), -cfg.a_max, cfg.a_max))
        else:
            a_cmd = u * cfg.a_max
        v_new = float(np.clip(v_old + a_cmd * DT, v_floor, v_top))
        a_actual = (v_new - v_old) / DT
        self._v        = v_new
        self._state[self._ode.v_index] = v_new
        self._road.set_speed(v_new)
        s_pos_start    = self._s_pos
        self._s_pos   += v_new * DT

        while self._obstacle_ends and self._s_pos >= self._obstacle_ends[0][0]:
            _, is_hole = self._obstacle_ends.pop(0)
            if is_hole:
                self._holes_passed += 1
            else:
                self._bumps_passed += 1

        v_road = self._road.speed
        new_state, z_B_ddot, phi_ddot, z_Wf_ddot, z_Wr_ddot = self._ode.step(
            self._state, self._road, s_pos_start, v_road, a_actual
        )
        self._state          = new_state
        self._t             += DT
        self._step_count    += 1
        self._last_z_B_ddot  = z_B_ddot
        self._last_phi_ddot  = phi_ddot
        self._last_z_W_ddot  = z_Wf_ddot

        self._accel_sq  += z_B_ddot ** 2
        self._peak_accel = max(self._peak_accel, abs(z_B_ddot))

        v_init    = self._compute_v_init(self._t)
        self._v_init_last = v_init

        z_W_ddot_both = 0.5 * (z_Wf_ddot + z_Wr_ddot)
        reward, breakdown = compute_reward(
            v=v_new,
            z_B_ddot=self._wk.step(z_B_ddot) if cfg.wk_heave else self._last_z_B_ddot,
            z_W_ddot=z_W_ddot_both,
            a_actual=a_actual,
            a_prev=self._prev_a,
            cfg=cfg,
            v_ref=v_init,
            phi_ddot=self._last_phi_ddot,
            dt=DT,
        )
        self._speed_err_sq   += (v_init - v_new) ** 2

        self._episode_reward += reward

        self._prev_a = a_actual

        truncated  = False
        terminated = False

        if self._step_count >= self._max_episode_steps:
            _wb = float(self._ode.wheelbase)
            if self._road_end is not None:
                _end_pt = self._road_end + _wb + 3.0
            else:
                _end_pt = (self._last_bump_end + _wb) if self._last_bump_end is not None else None
            reached = _end_pt is not None and self._s_pos >= _end_pt
            terminated = reached
            truncated = not reached
            self._termination_reason = "completed" if reached else "timeout"
            reward += compute_terminal_reward(
                success=reached, t=self._t,
                t_max=self._max_episode_steps * DT, cfg=cfg,
            )

        if self.render_mode == 'human':
            self.render()

        if (terminated or truncated) and self.render_mode == 'human' and self._freeze_episode:
            self._freeze_render = True
            self.render()

        info = self._info(z_B_ddot)
        info.update(breakdown)
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return self._vehicle.render().render_env(self)

    def set_render_label(self, label: str | None) -> None:
        self._render_label = label
        fig = self._fig
        if fig is not None:
            fig.suptitle(label or '', fontsize=9, color='#444444', x=0.5, y=0.98)

    def close(self):
        self._vehicle.render().close_env(self)


    def _obs(self) -> np.ndarray:
        body = self._ode.observe(self._state, float(self._last_z_W_ddot),
                                 float(self._last_z_B_ddot), float(self._last_phi_ddot))
        v        = float(self._v)
        a_actual = float(self._prev_a)
        v_err    = float(self._v_init_last - v)
        return np.array([*body, v, a_actual, v_err], dtype=np.float32)

    def _info(self, z_B_ddot: float) -> dict:
        n   = max(self._step_count, 1)
        rms = np.sqrt(self._accel_sq / n)
        susp_defl, tire_defl = self._ode.deflections(self._state, self._road, self._s_pos)
        return {
            'episode_reward': float(self._episode_reward),
            'rms_accel':       float(rms),
            'peak_accel':      float(self._peak_accel),
            'comfort_score':   float(max(0.0, 1.0 - rms / self._rcfg.a_limit)),
            'step_count':      self._step_count,
            'episode_time':    self._t,
            'z_B_ddot':        float(z_B_ddot),
            'z_W_ddot':        float(self._last_z_W_ddot),
            'phi_ddot':        float(self._last_phi_ddot),
            'susp_defl':       susp_defl,
            'tire_defl':       tire_defl,
            'speed':           float(self._v),
            'speed_kmh':       float(self._v * 3.6),
            'v_init':           float(self._v_init_last),
            'v_init_kmh':       float(self._v_init_last * 3.6),
            'speed_error':       float(self._v_init_last - self._v),
            'speed_error_rms':   float(np.sqrt(self._speed_err_sq / n)),
            'bumps_passed':    int(self._bumps_passed),
            'bumps_total':     self._bumps_total,
            'holes_passed':    int(self._holes_passed),
            'holes_total':     self._holes_total,
            'termination_reason': self._termination_reason,
        }

    def _compute_v_init(self, t: float) -> float:
        if self._ref_speed_profile == "custom" and self._v_init_fn is not None:
            return float(self._v_init_fn(t))
        return self._v_episode_ref
