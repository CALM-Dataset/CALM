import atexit
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

from gym_env.common.env_params import DT
from gym_env.common.reward.reward_params import RewardConfig, load_reward_config
from gym_env.vehicles import get_vehicle
from road.road_generator import RoadGenerator
from baseline.mpc.config import ocp_weights
from baseline.mpc.ocp import build_solver_simple
from baseline.mpc.comfort_map import ComfortMap


_OCP_VERSION = 'v8'


@contextmanager
def _suppress_native_output():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)


class MPCController:
    def __init__(
        self,
        cfg:                  Optional[RewardConfig] = None,
        physics:              Optional[dict] = None,
        N:                    int   = 50,
        dt:                   float = DT,
        nlp_solver_max_iter:  int   = 10,
        vehicle:              str   = "half",
        weights:              Optional[dict] = None,
        objective:            str   = "comfort_map",
    ):
        self._cfg     = cfg or load_reward_config()
        veh = get_vehicle(vehicle)
        self._physics = dict(physics or veh.physics())
        self._physics['v_limit'] = float(self._cfg.v_limit)
        self._physics['a_max'] = float(self._cfg.a_max)
        self._vehicle = vehicle
        self._wheelbase = float(veh.make_ode().wheelbase)
        self._v_idx   = veh.v_index

        self._N                   = N
        self._dt                  = dt
        self._nlp_solver_max_iter = nlp_solver_max_iter
        self._prev_u              = 0.0
        self._v_nom               = float(self._cfg.v_limit)
        self._gen_base            = Path(tempfile.gettempdir()) / 'acados_qc'

        self._comfort_map = ComfortMap(self._cfg, self._physics, vehicle=vehicle)
        self._crossing: list[float] = []
        self._planner = None
        if objective == "frontier":
            from baseline.mpc.frontier_objective import FrontierPlanner
            self._planner = FrontierPlanner(self._cfg, vehicle, self._physics)

        gen_dir = str(self._gen_base / f'{_OCP_VERSION}_N{self._N}_{os.getpid()}')
        os.makedirs(gen_dir, exist_ok=True)
        atexit.register(shutil.rmtree, gen_dir, ignore_errors=True)
        with _suppress_native_output():
            self._solver = build_solver_simple(
                self._physics, self._cfg,
                N=self._N, dt=self._dt, gen_dir=gen_dir,
                nlp_solver_max_iter=self._nlp_solver_max_iter,
                weights=weights or ocp_weights(),
            )

    def reset(self, road: RoadGenerator) -> None:
        self._prev_u = 0.0
        self._v_nom = float(getattr(road, 'speed', self._cfg.v_limit))
        if self._planner is not None:
            from eval.comfort_score import end_point
            self._crossing = self._planner.plan(road, self._v_nom, end_point(road, self._wheelbase))
            return
        for _, A, L in getattr(road, '_bumps', []) or []:
            self._comfort_map.optimal_speed(A, L, self._v_nom)

    def _compute_speed_profile(self, s_pos: float, v_current: float,
                               road: RoadGenerator) -> np.ndarray:
        cfg   = self._cfg
        N     = self._N
        dt    = self._dt
        v_max = float(cfg.v_limit)
        v_min = float(cfg.v_min)
        a_max = self._planner.ramp_decel if self._planner is not None else float(cfg.a_max)

        v_nom   = self._v_nom
        s_nodes = s_pos + v_current * np.arange(N + 1) * dt
        v_ref   = np.full(N + 1, v_nom)

        if not (hasattr(road, '_bumps') and road._bumps):
            return v_ref

        for i, (x0, A, L) in enumerate(road._bumps):
            if self._planner is not None:
                v_c = float(np.clip(self._crossing[i], v_min, v_max))
            else:
                v_c = float(np.clip(self._comfort_map.optimal_speed(A, L, v_nom),
                                    v_min, v_max))
            v_approach = max(float(v_current), float(v_nom))
            d_app  = max(0.0, (v_approach**2 - v_c**2)) / (2.0 * max(a_max, 1e-6))
            d_rec  = (v_nom**2 - v_c**2) / (2.0 * max(a_max, 1e-6))
            x_ramp = x0 - d_app
            x_end  = x0 + L + self._wheelbase
            x_done = x_end + d_rec

            for k in range(N + 1):
                s = s_nodes[k]
                if s < x_ramp:
                    continue
                elif s < x0:
                    frac = (s - x_ramp) / max(d_app, 1e-6)
                    v_ref[k] = min(v_ref[k], v_approach - (v_approach - v_c) * frac)
                elif s <= x_end:
                    v_ref[k] = min(v_ref[k], v_c)
                elif s < x_done:
                    frac = (s - x_end) / max(d_rec, 1e-6)
                    v_ref[k] = min(v_ref[k], v_c + (v_nom - v_c) * frac)

        return np.clip(v_ref, v_min, v_max)


    def act(self, x: np.ndarray, s_pos: float, road: RoadGenerator) -> float:
        solver  = self._solver
        v_refs  = self._compute_speed_profile(s_pos, float(x[self._v_idx]), road)

        x_init = np.array([float(x[self._v_idx]), float(s_pos), self._prev_u])
        solver.set(0, 'lbx', x_init)
        solver.set(0, 'ubx', x_init)

        for k in range(self._N + 1):
            solver.set(k, 'p', np.array([v_refs[k]]))

        with _suppress_native_output():
            status = solver.solve()
        if status not in (0, 2):
            return self._prev_u

        u0 = float(np.clip(float(solver.get(0, 'u')[0]), -1.0, 1.0))
        self._prev_u = u0
        return u0
