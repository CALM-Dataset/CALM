import collections
import os

import numpy as np

from gym_env.common.render import (
    Hud, build_bump_vlines, bump_signature, close_env, damper_xy, grab_key_focus, ground_symbol_xy,
    init_music, render_bumps, render_env, spring_xy, wait_for_episode_nav,
)
from gym_env.common.render_params import RENDER_MUSIC

from gym_env.half_car.config.render_params import (
    RENDER_Y_W_NOM, RENDER_Y_B_NOM,
    RENDER_W_MW, RENDER_H_MW, RENDER_C_GROUND, RENDER_CONTACT_STEM, Y_LINE_OFFSET, RENDER_REALISTIC_WHEELS,
    RENDER_H_MB, RENDER_XLIM, RENDER_YLIM,
    RENDER_ROAD_HALF, RENDER_ROAD_N,
    RENDER_C_MB, RENDER_C_MW, RENDER_C_SPRING, RENDER_C_DAMPER,
    RENDER_SP_X, RENDER_SP_W, RENDER_SP_N,
    RENDER_DA_X, RENDER_DA_W, RENDER_DA_PIST_H, RENDER_DA_PIST_FRAC,
    RENDER_DA_LOWER_STEM, RENDER_DA_CYL_H_SUSP, RENDER_SUSP_SCALE, RENDER_DA_CYL_H_TIRE, RENDER_GROUND_Y,
    RENDER_ARR_X, RENDER_ARR_Y, RENDER_ARR_LEN,
    RENDER_L, RENDER_LF, RENDER_LR, RENDER_X_FRONT, RENDER_X_REAR,
    RENDER_MB_OVERHANG, RENDER_OBSTACLE_FLAG_M, RENDER_MAX_EP_STEPS,
)

_ROAD_X = np.linspace(-RENDER_ROAD_HALF, RENDER_ROAD_HALF, RENDER_ROAD_N)
_MAX_EP_STEPS = RENDER_MAX_EP_STEPS

_WHEEL_R = 0.32
_WHEEL_RY = 0.5
_SK = RENDER_SUSP_SCALE
_C_TIRE_PART = '#56606b'
_C_TIRE_HIT = '#d62d20'
_HUB_H = 0.12
_ARR_X0      = RENDER_ARR_X
_ARR_Y_V     = RENDER_ARR_Y
_ARR_MAX_LEN = RENDER_ARR_LEN


__all__ = ["close_env", "init_render", "push_history", "render_env", "update_artists", "wait_for_episode_nav"]


def _damper_xy(x_c: float, y_top: float, y_bot: float, cyl_h: float, k: float = 1.0):
    return damper_xy(x_c, y_top, y_bot, cyl_h, RENDER_DA_W * k, RENDER_DA_PIST_H * k,
                     RENDER_DA_PIST_FRAC, RENDER_DA_LOWER_STEM)


def _chassis_corners(y_Bf: float, y_Br: float):
    o     = RENDER_MB_OVERHANG
    slope = (y_Bf - y_Br) / (RENDER_X_FRONT - RENDER_X_REAR)
    fx, fy = RENDER_X_FRONT + o, y_Bf + slope * o
    rx, ry = RENDER_X_REAR - o, y_Br - slope * o
    hh = RENDER_H_MB / 2
    return [[rx, ry - hh], [fx, fy - hh], [fx, fy + hh], [rx, ry + hh]]


def _make_axle(ax, x_c: float, label: str):
    from matplotlib.patches import Rectangle
    _hw = RENDER_DA_W / 2
    a = {}
    if RENDER_REALISTIC_WHEELS:
        a['tire_spring'],         = ax.plot([], [], '-', color=_C_TIRE_PART, lw=1.6, zorder=4)
        a['tire_damp_rod'],       = ax.plot([], [], '-', color=_C_TIRE_PART, lw=1.3, zorder=4)
        a['tire_damp_lower_rod'], = ax.plot([], [], '-', color=_C_TIRE_PART, lw=1.3, zorder=4)
        a['tire_damp_cyl'],       = ax.plot([], [], '-', color=_C_TIRE_PART, lw=1.8, zorder=4)
        a['tire_damp_pist'] = Rectangle((x_c + RENDER_DA_X - _hw + 0.01, 0),
                                        RENDER_DA_W - 0.02, RENDER_DA_PIST_H,
                                        fc=_C_TIRE_PART, ec='none', zorder=4)
        ax.add_patch(a['tire_damp_pist'])
        from matplotlib.patches import Ellipse
        a['tire'] = Ellipse((x_c, RENDER_Y_W_NOM), 1.0, 2 * _WHEEL_RY,
                            fc='#2b2f36', ec='#15181c', lw=1.2, zorder=3)
        ax.add_patch(a['tire'])
        a['rim'] = Ellipse((x_c, RENDER_Y_W_NOM), 0.6, 1.2 * _WHEEL_RY,
                           fc='#c9d1da', ec='#7b8794', lw=1.0, zorder=3.1)
        ax.add_patch(a['rim'])
        a['spokes'], = ax.plot([], [], '-', color='#7b8794', lw=1.3, zorder=3.2)
        a['mw_patch'] = Rectangle((x_c - RENDER_W_MW / 2, RENDER_Y_W_NOM - _HUB_H / 2),
                                  RENDER_W_MW, _HUB_H,
                                  fc=RENDER_C_MW, ec='#333333', lw=1.2, zorder=5)
        ax.add_patch(a['mw_patch'])
        a['mw_dot']   = ax.plot(x_c, RENDER_Y_W_NOM, 'o', color='white', ms=3, zorder=7)[0]
        a['mw_label'] = ax.text(x_c - RENDER_W_MW / 2 - 0.12, RENDER_Y_W_NOM, label,
                                ha='right', va='center', fontsize=10, color='#1b1f24',
                                zorder=7)

    else:
        a['tire_spring'],         = ax.plot([], [], '-', color=RENDER_C_SPRING, lw=2.0, zorder=4)
        a['tire_damp_rod'],       = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
        a['tire_damp_lower_rod'], = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
        a['tire_damp_cyl'],       = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=2.0, zorder=4)
        a['tire_damp_pist'] = Rectangle((x_c + RENDER_DA_X - _hw + 0.01, 0),
                                        RENDER_DA_W - 0.02, RENDER_DA_PIST_H,
                                        fc=RENDER_C_DAMPER, ec='none', zorder=4)
        ax.add_patch(a['tire_damp_pist'])
        a['mw_patch'] = Rectangle((x_c - RENDER_W_MW / 2, RENDER_Y_W_NOM - RENDER_H_MW / 2),
                                  RENDER_W_MW, RENDER_H_MW,
                                  fc=RENDER_C_MW, ec='#333333', lw=1.2, zorder=5)
        ax.add_patch(a['mw_patch'])
        a['mw_dot']   = ax.plot(x_c, RENDER_Y_W_NOM, 'o', color='white', ms=3, zorder=7)[0]
        a['mw_label'] = ax.text(x_c - RENDER_W_MW / 2 - 0.12, RENDER_Y_W_NOM, label,
                                ha='right', va='center', fontsize=10, color='#1b1f24',
                                zorder=7)

    a['susp_spring'],         = ax.plot([], [], '-', color=RENDER_C_SPRING, lw=2.0 * _SK, zorder=4)
    a['susp_damp_rod'],       = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5 * _SK, zorder=4)
    a['susp_damp_lower_rod'], = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5 * _SK, zorder=4)
    a['susp_damp_cyl'],       = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=2.0 * _SK, zorder=4)
    a['susp_damp_pist'] = Rectangle((x_c + RENDER_DA_X - _hw * _SK + 0.01 * _SK, 0),
                                    (RENDER_DA_W - 0.02) * _SK, RENDER_DA_PIST_H * _SK,
                                    fc=RENDER_C_DAMPER, ec='none', zorder=4)
    ax.add_patch(a['susp_damp_pist'])

    if RENDER_REALISTIC_WHEELS:
        a['contact_dot'], = ax.plot([], [], 'o', color='#e74c3c', ms=6, zorder=7)
    else:
        a['ground_sym'],   = ax.plot([], [], '-', color=RENDER_C_GROUND, lw=1.5, zorder=2)
        a['contact_stem'], = ax.plot([], [], '-', color=RENDER_C_GROUND, lw=1.8, zorder=6)
        a['contact_dot'],  = ax.plot([], [], 'o', color=RENDER_C_GROUND, ms=9,  zorder=7)
    return a


def _update_axle(a, x_c: float, y_W: float, y_B_corner: float,
                 y_road: float, on_obstacle: bool, ys: float, rot: float = 0.0, wheel_rx: float = 0.6):
    sp_x = x_c + RENDER_SP_X
    da_x = x_c + RENDER_DA_X

    if RENDER_REALISTIC_WHEELS:
        a['contact_dot'].set_data([x_c] if on_obstacle else [], [y_road] if on_obstacle else [])
        tire_c = _C_TIRE_HIT if on_obstacle else _C_TIRE_PART
        for k in ('tire_spring', 'tire_damp_rod', 'tire_damp_lower_rod', 'tire_damp_cyl'):
            a[k].set_color(tire_c)
        a['tire_damp_pist'].set_facecolor(tire_c)

        y_tire_top = y_W - _HUB_H / 2
        if y_tire_top > y_road + 0.05:
            a['tire_spring'].set_data(*spring_xy(sp_x, y_tire_top, y_road, RENDER_SP_N, RENDER_SP_W))
            u_rod, l_rod, cyl_xy, pr = _damper_xy(da_x, y_tire_top, y_road, RENDER_DA_CYL_H_TIRE)
            a['tire_damp_rod'].set_data(*u_rod)
            a['tire_damp_lower_rod'].set_data(*l_rod)
            a['tire_damp_cyl'].set_data(*cyl_xy)
            a['tire_damp_pist'].set_xy((pr[0], pr[1]))
            a['tire_damp_pist'].set_height(pr[3])

        a['mw_patch'].set_xy((x_c - RENDER_W_MW / 2, y_W - _HUB_H / 2))
        a['tire'].set_center((x_c, y_W)); a['tire'].width = 2 * wheel_rx
        a['rim'].set_center((x_c, y_W)); a['rim'].width = 1.2 * wheel_rx
        ang = rot + np.arange(5) * (2 * np.pi / 5)
        rx, ry = 0.6 * wheel_rx, 0.6 * _WHEEL_RY
        sx = np.empty(15); sy = np.empty(15)
        sx[0::3], sy[0::3] = x_c, y_W
        sx[1::3], sy[1::3] = x_c + rx * np.cos(ang), y_W + ry * np.sin(ang)
        sx[2::3], sy[2::3] = np.nan, np.nan
        a['spokes'].set_data(sx, sy)
        a['mw_dot'].set_data([x_c], [y_W])
        a['mw_label'].set_position((x_c - RENDER_W_MW / 2 - 0.12, y_W))

    else:
        y_line = y_road + Y_LINE_OFFSET
        gx, gy = ground_symbol_xy(x_c, y_line, half_w=RENDER_W_MW / 2 + 0.15)
        a['ground_sym'].set_data(gx, gy)
        a['contact_stem'].set_data([x_c, x_c], [y_line - Y_LINE_OFFSET + RENDER_CONTACT_STEM, y_line])
        a['contact_dot'].set_data([x_c], [y_line - Y_LINE_OFFSET])
        a['contact_dot'].set_color('#e74c3c' if on_obstacle else RENDER_C_GROUND)

        y_tire_top = y_W - RENDER_H_MW / 2
        a['tire_spring'].set_data(*spring_xy(sp_x, y_tire_top, y_line, RENDER_SP_N, RENDER_SP_W))
        u_rod, l_rod, cyl_xy, pr = _damper_xy(da_x, y_tire_top, y_line, RENDER_DA_CYL_H_TIRE)
        a['tire_damp_rod'].set_data(*u_rod)
        a['tire_damp_lower_rod'].set_data(*l_rod)
        a['tire_damp_cyl'].set_data(*cyl_xy)
        a['tire_damp_pist'].set_xy((pr[0], pr[1]))
        a['tire_damp_pist'].set_height(pr[3])

        a['mw_patch'].set_xy((x_c - RENDER_W_MW / 2, y_W - RENDER_H_MW / 2))
        a['mw_dot'].set_data([x_c], [y_W])
        a['mw_label'].set_position((x_c - RENDER_W_MW / 2 - 0.12, y_W))

    hub_h = _HUB_H if RENDER_REALISTIC_WHEELS else RENDER_H_MW
    y_susp_bot = y_W + hub_h / 2
    y_susp_top = y_B_corner - RENDER_H_MB / 2
    if y_susp_top > y_susp_bot + 0.05:
        a['susp_spring'].set_data(*spring_xy(sp_x, y_susp_top, y_susp_bot, RENDER_SP_N, RENDER_SP_W * _SK))
        u_rod, l_rod, cyl_xy, pr = _damper_xy(da_x, y_susp_top, y_susp_bot, RENDER_DA_CYL_H_SUSP, _SK)
        a['susp_damp_rod'].set_data(*u_rod)
        a['susp_damp_lower_rod'].set_data(*l_rod)
        a['susp_damp_cyl'].set_data(*cyl_xy)
        a['susp_damp_pist'].set_xy((pr[0], pr[1]))
        a['susp_damp_pist'].set_height(pr[3])


def _build_ts_specs(env):
    specs = []
    flags = getattr(env, "_ts_flags", {})
    if flags.get("z", False):
        specs.append(([('z_B',  r'$z_B$',     '#1f4e79', '-'),
                       ('z_Wf', r'$z_{W,f}$', '#c0392b', '--'),
                       ('z_Wr', r'$z_{W,r}$', '#2e7d4f', ':')],
                      r'$z\ \mathrm{[cm]}$'))
        specs.append(([('phi', r'$\varphi$', '#6a3d9a', '-')],
                      r'$\varphi\ \mathrm{[deg]}$'))
    if flags.get("f", False):
        specs.append(([('F', r'$F_D$', '#2e7d4f', '-')], r'$F_D\ \mathrm{[N]}$'))
    if flags.get("speed", False):
        specs.append(([('s_dot', r'$v$', '#0b6e8a', '-')], r'$v\ \mathrm{[km/h]}$'))
    if flags.get("z_ddot", False):
        specs.append(([('z_B_ddot', r'$\ddot{z}_B$', '#222222', '-')],
                      r'$\ddot{z}_B\ \mathrm{[m/s^2]}$'))
    return specs


def _on_obstacle(road, s, zeta, thresh=RENDER_OBSTACLE_FLAG_M):
    for x0, A, L in (getattr(road, "_bumps", None) or []):
        if L is not None and x0 <= s <= x0 + L:
            return True
    return abs(zeta) > thresh


def init_render(env):
    import matplotlib
    if not os.environ.get('DISPLAY'):
        matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.backend_bases import NavigationToolbar2

    NavigationToolbar2.toolitems = tuple(
        t for t in NavigationToolbar2.toolitems if t[0] != 'Subplots'
    )

    ts_specs = _build_ts_specs(env)
    if env._show_ts and not ts_specs:
        env._show_ts = False

    env._ren_hist = {
        't':        collections.deque(maxlen=_MAX_EP_STEPS),
        'z_B':      collections.deque(maxlen=_MAX_EP_STEPS),
        'z_Wf':     collections.deque(maxlen=_MAX_EP_STEPS),
        'z_Wr':     collections.deque(maxlen=_MAX_EP_STEPS),
        'phi':      collections.deque(maxlen=_MAX_EP_STEPS),
        'z_B_ddot': collections.deque(maxlen=_MAX_EP_STEPS),
        'F':        collections.deque(maxlen=_MAX_EP_STEPS),
        's_dot':    collections.deque(maxlen=_MAX_EP_STEPS),
        's_pos':    collections.deque(maxlen=_MAX_EP_STEPS),
    }

    ep_title = (getattr(env, '_render_label', None)
                or f'Half-Car Env  ·  Episode {getattr(env, "_episode_count", "?")}')

    gym_env_title = 'Half-Car gym env'

    if env._show_ts:
        fig = plt.figure(figsize=(14, 7))
        fig.suptitle(ep_title, fontsize=11, color='#1b1f24', x=0.5, y=0.995, va='top')
        gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 2],
                       left=0.06, right=0.97, bottom=0.09, top=0.87, wspace=0.40)
        ax_s = fig.add_subplot(gs[0, 0])
        n_ts = len(ts_specs)
        gs_r = gs[0, 1].subgridspec(n_ts, 1, hspace=0.10)
        ax_r = [fig.add_subplot(gs_r[i]) for i in range(n_ts)]
    else:
        fig  = plt.figure(figsize=(9, 7))
        fig.suptitle(ep_title, fontsize=11, color='#1b1f24', x=0.5, y=0.995, va='top')
        ax_s = fig.add_subplot(1, 1, 1)
        ax_r = []

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(gym_env_title)
    fig.set_facecolor('#fbfbfd')

    import matplotlib.ticker as ticker
    ax_s.set_facecolor('#f4f8fc')
    sky = np.linspace(0, 1, 64)[:, None]
    ax_s.imshow(sky, extent=(-1e3, 1e3, RENDER_YLIM[0], RENDER_YLIM[1]), origin='lower', aspect='auto',
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list('sky', ['#f3f7fb', '#bcd8f0']),
                zorder=0, interpolation='bilinear')
    ax_s.set_xlim(RENDER_XLIM)
    ax_s.set_ylim(RENDER_YLIM)
    ax_s.set_xlabel('Longitudinal position [cm]', fontsize=11, color='#1b1f24')
    ax_s.set_ylabel('Height [cm]', fontsize=11, color='#1b1f24')
    ax_s.tick_params(labelsize=10, direction='in', colors='#1b1f24')
    ax_s.grid(True, axis='x', lw=0.4, alpha=0.6, color='#8fa3b8', linestyle=':')
    ax_s.set_axisbelow(True)
    ax_s.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{(x + env._s_pos) * 100:.0f}')
    )
    ax_s.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f'{y * 100:.0f}')
    )

    ground_ticks, = ax_s.plot([], [], '-', color='#8d939c', lw=1.6, zorder=2, solid_capstyle='butt')
    road_line, = ax_s.plot([], [], '-', color='#e9edf1', lw=1.8, zorder=2,
                           label=r'Road profile $\zeta(x)$')

    axle_f = _make_axle(ax_s, RENDER_X_FRONT, r'$m_{W,f}$')
    axle_r = _make_axle(ax_s, RENDER_X_REAR,  r'$m_{W,r}$')

    from matplotlib.patches import Polygon
    mb_poly = Polygon(_chassis_corners(RENDER_Y_B_NOM, RENDER_Y_B_NOM),
                      closed=True, fc=RENDER_C_MB, ec='#333333', lw=1.2, zorder=5)
    ax_s.add_patch(mb_poly)
    mb_dot   = ax_s.plot((RENDER_X_FRONT + RENDER_X_REAR) / 2, RENDER_Y_B_NOM,
                         'o', color='#333333', ms=3, zorder=7)[0]
    mb_label = ax_s.text((RENDER_X_FRONT + RENDER_X_REAR) / 2,
                          RENDER_Y_B_NOM + RENDER_H_MB / 2 + 0.12,
                          r'$m_B,\ I_B$', ha='center', va='bottom',
                          fontsize=11, color='#1b1f24', zorder=7)

    hud = Hud(ax_s, [('Time', 't'), ('Position', 's'), ('Speed', 'v'), ('Body heave', 'zb'), ('Pitch', 'phi'), ('Reward', 'R'), ('Bumps', 'b')])

    ax_s.legend(fontsize=10, loc='upper right', frameon=False, labelcolor='#1b1f24',
                handlelength=1.5, borderpad=0.4)

    ts = {}
    for i, ax in enumerate(ax_r):
        series, ylabel = ts_specs[i]
        for key, label, colour, style in series:
            ts[key], = ax.plot([], [], style, color=colour, lw=1.2, label=label)

        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7.5, direction='in', top=False, right=False)
        ax.grid(True, lw=0.4, alpha=0.5, color='#8fa3b8', linestyle=':')
        ax.tick_params(labelsize=9.5, colors='#1b1f24')
        ax.axhline(0, color='#8fa3b8', lw=0.8)
        ax.set_ylabel(ylabel, fontsize=11, color='#1b1f24')

        if len(series) > 1:
            ax.legend(fontsize=9, loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=len(series),
                      frameon=False, labelcolor='#1b1f24', columnspacing=1.2, handlelength=1.8)

        if i < len(ax_r) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('$t$ [s]', fontsize=11, color='#1b1f24')

    bump_vlines, bump_sig = build_bump_vlines(env, ax_r)

    v_line, = ax_s.plot([_ARR_X0, _ARR_X0], [_ARR_Y_V, _ARR_Y_V], '-',
                        color='#0b6e8a', lw=2.4, zorder=8)
    v_head, = ax_s.plot([_ARR_X0],          [_ARR_Y_V],            '>',
                        color='#0b6e8a', ms=8,  zorder=8)
    ax_s.text(_ARR_X0 - 0.15, _ARR_Y_V, r'$v$', ha='right', va='center',
              fontsize=11, color='#0b6e8a', zorder=8)
    ax_s.plot([_ARR_X0 + _ARR_MAX_LEN] * 2,
              [_ARR_Y_V - 0.20, _ARR_Y_V + 0.20],
              color='#c0392b', lw=2.2, ls='-', zorder=8)
    ax_s.text(_ARR_X0 + _ARR_MAX_LEN, _ARR_Y_V - 0.32,
              f'{env._rcfg.v_max * 3.6:.0f} km/h', ha='center', va='top',
              fontsize=10, color='#c0392b', fontweight='bold', zorder=8)
    vmax_tick, = ax_s.plot([_ARR_X0, _ARR_X0], [_ARR_Y_V - 0.16, _ARR_Y_V + 0.16],
                           color='#b9770e', lw=1.6, ls='--', zorder=7)
    vmax_label = ax_s.text(_ARR_X0, _ARR_Y_V + 0.28, '', ha='center', va='bottom',
                           fontsize=10, color='#b9770e', fontweight='bold', zorder=8)

    if env.render_mode == 'human':
        plt.ion()
        plt.show(block=False)
        grab_key_focus(fig)

    env._fig  = fig
    env._ax_s = ax_s
    env._ax_r = ax_r
    if RENDER_MUSIC and env.render_mode != 'none':
        init_music(env, fig)
    env._artists = {
        'road_line':           road_line,
        'axle_f':              axle_f,
        'axle_r':              axle_r,
        'mb_poly':             mb_poly,
        'ground_ticks':        ground_ticks,
        'mb_dot':              mb_dot,
        'mb_label':            mb_label,
        'hud':                 hud,
        'ts':                  ts,
        'bump_vlines':         bump_vlines,
        'bump_sig':            bump_sig,
        'v_line':              v_line,
        'v_head':              v_head,
        'vmax_tick':           vmax_tick,
        'vmax_label':          vmax_label,
    }


def push_history(env):
    x   = env._state
    lf  = getattr(env._ode, "_lf", 1.206)
    lr  = getattr(env._ode, "_lr", 1.664)
    z_B  = float(x[9])
    z_Wf = (z_B - lf * float(x[10])) + float(x[6])
    z_Wr = (z_B + lr * float(x[10])) + float(x[4])
    h   = env._ren_hist
    h['t'].append(env._t)
    h['z_B'].append(z_B)
    h['z_Wf'].append(z_Wf)
    h['z_Wr'].append(z_Wr)
    h['phi'].append(float(x[10]))
    h['z_B_ddot'].append(env._last_z_B_ddot)
    h['F'].append(0.0)
    h['s_dot'].append(float(env._v))
    h['s_pos'].append(env._s_pos)


def update_artists(env):
    art = env._artists
    ys  = env._y_scale
    x   = env._state

    lf  = getattr(env._ode, "_lf", RENDER_LF)
    lr  = getattr(env._ode, "_lr", RENDER_LR)
    L   = getattr(env._ode, "_L",  RENDER_L)
    phi = float(x[10])
    z_B = float(x[9])

    z_Bf = z_B - lf * phi
    z_Br = z_B + lr * phi
    z_Wf = z_Bf + float(x[6])
    z_Wr = z_Br + float(x[4])

    zeta_f = float(env._road.get_height_at(env._s_pos))
    zeta_r = float(env._road.get_height_at(env._s_pos - L))

    y_road_f = RENDER_GROUND_Y + zeta_f * ys
    y_road_r = RENDER_GROUND_Y + zeta_r * ys
    if RENDER_REALISTIC_WHEELS:
        y_Wf   = y_road_f + _WHEEL_RY + min(z_Wf - zeta_f, 0.0) * ys
        y_Wr   = y_road_r + _WHEEL_RY + min(z_Wr - zeta_r, 0.0) * ys
        y_gap  = RENDER_Y_B_NOM - RENDER_Y_W_NOM
        y_Bf   = y_Wf + y_gap + (z_Bf - z_Wf) * ys
        y_Br   = y_Wr + y_gap + (z_Br - z_Wr) * ys
    else:
        y_Wf   = RENDER_Y_W_NOM + RENDER_GROUND_Y + z_Wf * ys
        y_Wr   = RENDER_Y_W_NOM + RENDER_GROUND_Y + z_Wr * ys
        y_Bf   = RENDER_Y_B_NOM + RENDER_GROUND_Y + z_Bf * ys
        y_Br   = RENDER_Y_B_NOM + RENDER_GROUND_Y + z_Br * ys

    ax_s   = env._ax_s
    x0, x1 = ax_s.get_xlim()
    y0_, y1_ = ax_s.get_ylim()
    pad    = 0.03 * (x1 - x0)
    base_w = RENDER_XLIM[1] - RENDER_XLIM[0]
    n_pts  = int(np.clip(RENDER_ROAD_N * (x1 - x0) / base_w, RENDER_ROAD_N, 6 * RENDER_ROAD_N))
    road_x = np.linspace(x0 - pad, x1 + pad, n_pts)
    s_q    = env._s_pos + road_x
    road_h = env._road.get_height_array_pos(s_q) * ys + RENDER_GROUND_Y
    art['road_line'].set_data(road_x, road_h)

    if art.get('road_fill') is not None:
        art['road_fill'].remove()
    art['road_fill'] = ax_s.fill_between(
        road_x, road_h, ax_s.get_ylim()[0],
        color='#5b6068', edgecolor='none', zorder=1,
    )
    layer = 0.10 * (ax_s.get_ylim()[1] - ax_s.get_ylim()[0]) / 6
    if art.get('road_top') is not None:
        art['road_top'].remove()
    art['road_top'] = ax_s.fill_between(
        road_x, road_h, road_h - layer, color='#3f434a', edgecolor='none', zorder=1.5,
    )
    spacing = 0.9
    k0 = np.floor((env._s_pos + x0) / spacing)
    xs_ = (k0 + np.arange(int((x1 - x0) / spacing) + 3)) * spacing - env._s_pos
    xs_ = xs_[(xs_ >= x0) & (xs_ <= x1)]
    yt = np.interp(xs_, road_x, road_h) - layer * 2.2
    tx = np.empty(3 * len(xs_)); ty = np.empty(3 * len(xs_))
    tx[0::3], tx[1::3], tx[2::3] = xs_, xs_ + 0.35, np.nan
    ty[0::3], ty[1::3], ty[2::3] = yt, yt, np.nan
    art['ground_ticks'].set_data(tx, ty)

    h = env._ren_hist

    bb = ax_s.get_window_extent()
    wheel_rx = _WHEEL_RY * (x1 - x0) / (y1_ - y0_) * bb.height / max(bb.width, 1.0)

    road = env._road
    _update_axle(art['axle_f'], RENDER_X_FRONT, y_Wf, y_Bf, y_road_f,
                 _on_obstacle(road, env._s_pos, zeta_f), ys, env._s_pos / _WHEEL_R, wheel_rx)
    _update_axle(art['axle_r'], RENDER_X_REAR,  y_Wr, y_Br, y_road_r,
                 _on_obstacle(road, env._s_pos - L, zeta_r), ys, env._s_pos / _WHEEL_R, wheel_rx)

    art['mb_poly'].set_xy(_chassis_corners(y_Bf, y_Br))
    x_mid = (RENDER_X_FRONT + RENDER_X_REAR) / 2
    y_mid = (y_Bf + y_Br) / 2
    art['mb_dot'].set_data([x_mid], [y_mid])
    art['mb_label'].set_position((x_mid, max(y_Bf, y_Br) + RENDER_H_MB / 2 + 0.12))

    obst_str = f'{int(env._bumps_passed)}/{env._bumps_total}' if env._bumps_total else '-'
    if getattr(env, '_holes_total', 0):
        obst_str += f' H{int(env._holes_passed)}/{env._holes_total}'
    art['hud'].update(
        t=f'{env._t:.2f} s', s=f'{env._s_pos:.1f} m', v=f'{env._v*3.6:.1f} km/h',
        b=obst_str, zb=f'{float(x[9])*1000:+.1f} mm', phi=f'{np.degrees(phi):+.2f}°',
        R=f'{env._episode_reward:.1f}',
    )

    v_scale = env._rcfg.v_max
    v_frac  = float(np.clip(env._v / v_scale, 0.03, 1.0))
    v_tip   = _ARR_X0 + v_frac * _ARR_MAX_LEN
    art['v_line'].set_xdata([_ARR_X0, v_tip])
    art['v_head'].set_xdata([v_tip])

    v_des  = float(getattr(env, '_v_episode_ref', v_scale))
    x_des  = _ARR_X0 + float(np.clip(v_des / v_scale, 0.0, 1.0)) * _ARR_MAX_LEN
    art['vmax_tick'].set_xdata([x_des, x_des])
    art['vmax_label'].set_position((x_des, _ARR_Y_V + 0.28))
    art['vmax_label'].set_text(f'$v_{{ref}}$ {v_des*3.6:.0f} km/h')

    if not env._show_ts:
        return

    t_arr = np.array(h['t'])
    ts    = art['ts']
    _map = {
        'z_B':      np.array(h['z_B'])  * 100.0,
        'z_Wf':     np.array(h['z_Wf']) * 100.0,
        'z_Wr':     np.array(h['z_Wr']) * 100.0,
        'phi':      np.degrees(np.array(h['phi'])),
        'z_B_ddot': np.array(h['z_B_ddot']),
        'F':        np.array(h['F']),
        's_dot':    np.array(h['s_dot']) * 3.6,
    }
    for key, line in ts.items():
        line.set_data(t_arr, np.array(_map[key]))

    if len(t_arr) > 1:
        for ax in env._ax_r:
            ax.set_xlim(0, max(t_arr[-1], 1.0))
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

    bump_vlines = art.get('bump_vlines', [])
    bumps   = render_bumps(getattr(env, '_road', None))
    cur_sig = bump_signature(bumps)
    if cur_sig != art.get('bump_sig'):
        for lines in bump_vlines:
            for vl in lines:
                vl.remove()
        bump_vlines, _ = build_bump_vlines(env, env._ax_r)
        art['bump_vlines'] = bump_vlines
        art['bump_sig']    = cur_sig
    if bump_vlines and len(t_arr) > 0:
        s_arr   = np.array(h['s_pos'])
        for i, (x0, _A, _L) in enumerate(bumps):
            if i >= len(bump_vlines):
                break
            idx     = np.searchsorted(s_arr, x0)
            reached = idx < len(t_arr)
            for vl in bump_vlines[i]:
                if reached:
                    vl.set_xdata([t_arr[idx], t_arr[idx]])
                    vl.set_visible(True)
                else:
                    vl.set_visible(False)
