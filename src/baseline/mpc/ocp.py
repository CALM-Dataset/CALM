import numpy as np
import casadi as ca

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


def build_solver_simple(
    physics:              dict,
    cfg,
    N:                    int   = 50,
    dt:                   float = 0.02,
    gen_dir:              str   = '/tmp/acados_qc',
    nlp_solver_max_iter:  int   = 10,
) -> AcadosOcpSolver:
    model      = AcadosModel()
    model.name = 'qc_speed'

    nx, nu, np_ = 3, 1, 1
    x = ca.SX.sym('x', nx)
    u = ca.SX.sym('u', nu)
    p = ca.SX.sym('p', np_)

    v      = x[0]
    u_prev = x[2]
    a_max  = float(physics.get('a_max', 5.0))
    v_max  = float(cfg.v_limit)
    v_min  = float(cfg.v_min)
    g      = float(cfg.g)
    v_ref  = p[0]

    f_expl = ca.vertcat(
        u[0] * a_max,
        v,
        (u[0] - u_prev) / dt,
    )
    model.x           = x
    model.u           = u
    model.p           = p
    model.f_expl_expr = f_expl
    model.xdot        = ca.SX.sym('xdot', nx)
    model.f_impl_expr = model.xdot - f_expl

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf        = N * dt
    ocp.dims.nx = nx
    ocp.dims.nu = nu
    ocp.dims.np = np_

    r = ca.vertcat(
        (v - v_ref) / ca.fmax(v_ref, 1e-6),
        u[0] * a_max / g,
        (u[0] - u_prev) * a_max / g,
    )
    r_e = (v - v_ref) / ca.fmax(v_ref, 1e-6)

    W   = np.diag([float(cfg.Q_v), float(cfg.Q_a), float(cfg.w_jerk)])
    W_e = np.array([[float(cfg.Q_v)]])

    ocp.cost.cost_type          = 'NONLINEAR_LS'
    ocp.cost.cost_type_e        = 'NONLINEAR_LS'
    ocp.model.cost_y_expr       = r
    ocp.model.cost_y_expr_e     = r_e
    ocp.cost.W      = W
    ocp.cost.W_e    = W_e
    ocp.cost.yref   = np.zeros(3)
    ocp.cost.yref_e = np.zeros(1)

    ocp.constraints.lbu   = np.array([-1.0])
    ocp.constraints.ubu   = np.array([ 1.0])
    ocp.constraints.idxbu = np.array([0])
    ocp.constraints.lbx   = np.array([v_min])
    ocp.constraints.ubx   = np.array([v_max * 2.0])
    ocp.constraints.idxbx = np.array([0])
    ocp.constraints.x0    = np.zeros(nx)

    ocp.parameter_values = np.array([v_max])

    ocp.solver_options.integrator_type     = 'ERK'
    ocp.solver_options.num_stages          = 4
    ocp.solver_options.num_steps           = 1
    ocp.solver_options.nlp_solver_type     = 'SQP'
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    ocp.solver_options.qp_solver           = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N    = min(N, 10)
    ocp.solver_options.hessian_approx      = 'GAUSS_NEWTON'
    ocp.solver_options.print_level         = 0
    ocp.code_export_directory              = gen_dir

    solver = AcadosOcpSolver(ocp, json_file=f'{gen_dir}/acados_ocp.json', verbose=False)
    solver.options_set('print_level',    0)
    solver.options_set('qp_print_level', 0)
    return solver
