from __future__ import annotations


def ppo_runner(model_path: str, seed: int):
    from eval.eval import load_model, run_episode

    model, vecnorm = load_model('PPO', model_path)

    def run(road, track):
        ep, _, _ = run_episode(model, vecnorm, seed=seed, render=False,
                               deterministic=True, scenario_road=road,
                               perception_track=track)
        return (ep['speeds'],
                float(ep['comfort_score_running'][-1]) if ep['comfort_score_running'] else 0.0,
                float(ep['rms_accel_running'][-1]) if ep['rms_accel_running'] else 0.0)
    return run


def mpc_runner(vehicle: str, horizon: int, perception: bool, seed: int, objective: str = 'comfort_map'):
    from baseline.mpc.controller import MPCController
    from baseline.mpc.config import load_mpc_cfg
    from baseline.mpc.mpc import _vehicle_bits, make_env, run_episode

    bits = _vehicle_bits(vehicle)
    mcfg = load_mpc_cfg()
    cfg = bits['load_reward_config']()
    ctrl = MPCController(cfg=cfg, physics=dict(bits['PHYSICS']),
                         N=horizon, nlp_solver_max_iter=mcfg.get('nlp_solver_max_iter', 10),
                         vehicle=vehicle, objective=objective)
    preview_cfg = bits['load_preview_config']() if perception else None
    env = make_env(bits, cfg)

    def run(road, track):
        r = run_episode(env, ctrl, seed=seed, scenario_road=road,
                        perception_track=track if perception else None,
                        preview_cfg=preview_cfg)
        return r['_speeds'], float(r['comfort_score']), float(r['rms_accel'])
    return run
