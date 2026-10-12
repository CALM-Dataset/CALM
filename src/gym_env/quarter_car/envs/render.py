import collections
import os

import numpy as np

from gym_env.common.render import (
    Hud, Road, Wheel, build_bump_vlines, draw_sky, wheel_radius_x, bump_signature, close_env, damper_xy, ground_symbol_xy,
    init_music, render_bumps, render_env, spring_xy, wait_for_episode_nav,
)
from gym_env.common.render_params import RENDER_MUSIC

from gym_env.quarter_car.config.render_params import (
    RENDER_Y_W_NOM, RENDER_Y_B_NOM,
    RENDER_H_MW, RENDER_W_MW,
    RENDER_H_MB, RENDER_W_MB,
    RENDER_XLIM, RENDER_YLIM,
    RENDER_ROAD_HALF, RENDER_ROAD_N,
    RENDER_C_MB, RENDER_C_MW, RENDER_C_SPRING, RENDER_C_DAMPER,
    RENDER_C_GROUND,
    RENDER_SP_X, RENDER_SP_W, RENDER_SP_N,
    RENDER_DA_X, RENDER_DA_W, RENDER_DA_PIST_H, RENDER_DA_PIST_FRAC,
    RENDER_DA_LOWER_STEM, RENDER_DA_CYL_H_SUSP, RENDER_DA_CYL_H_TIRE,
    RENDER_CONTACT_STEM, RENDER_GROUND_Y,
    Y_LINE_OFFSET, RENDER_REALISTIC_WHEELS, RENDER_SUSP_SCALE,
    RENDER_ARR_X, RENDER_ARR_Y, RENDER_ARR_LEN,
)

_ROAD_X = np.linspace(-RENDER_ROAD_HALF, RENDER_ROAD_HALF, RENDER_ROAD_N)
_MAX_EP_STEPS = 4000

_WHEEL_RY = 0.5
_HUB_H = 0.12
_WHEEL_R = 0.32
_SK = RENDER_SUSP_SCALE
_WHEEL_GEOMETRY = dict(
    ry=_WHEEL_RY, hub_h=_HUB_H, w_hub=RENDER_W_MW, c_hub=RENDER_C_MW,
    sp_x=RENDER_SP_X, sp_w=RENDER_SP_W, sp_n=RENDER_SP_N,
    da_x=RENDER_DA_X, da_w=RENDER_DA_W, pist_h=RENDER_DA_PIST_H, pist_frac=RENDER_DA_PIST_FRAC,
    lower_stem=RENDER_DA_LOWER_STEM, cyl_h=RENDER_DA_CYL_H_TIRE,
)

_ARR_X0      = RENDER_ARR_X
_ARR_Y_V     = RENDER_ARR_Y
_ARR_MAX_LEN = RENDER_ARR_LEN


__all__ = ["close_env", "init_render", "push_history", "render_env", "update_artists", "wait_for_episode_nav"]


def _damper_xy(x_c: float, y_top: float, y_bot: float, cyl_h: float, k: float = 1.0):
    return damper_xy(x_c, y_top, y_bot, cyl_h, RENDER_DA_W * k, RENDER_DA_PIST_H * k,
                     RENDER_DA_PIST_FRAC, RENDER_DA_LOWER_STEM)


def _build_ts_specs(env):
    specs = []
    flags = getattr(env, "_ts_flags", {})
    if flags.get("z", False):
        specs.append(('z_B', 'z_W', r'$z\ \mathrm{[cm]}$', '#1f4e79', '#c0392b'))
    if flags.get("f", False):
        specs.append(('F', None, r'$F_D\ \mathrm{[N]}$', '#2e7d4f', None))
    if flags.get("speed", False):
        specs.append(('s_dot', None, r'$v\ \mathrm{[km/h]}$', '#0b6e8a', None))
    if flags.get("z_ddot", False):
        specs.append(('z_B_ddot', None, r'$\ddot{z}_B\ \mathrm{[m/s^2]}$', '#222222', None))
    return specs


def init_render(env):
    import matplotlib
    if not os.environ.get('DISPLAY'):
        matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
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
        'z_W':      collections.deque(maxlen=_MAX_EP_STEPS),
        'z_B_ddot': collections.deque(maxlen=_MAX_EP_STEPS),
        'F':        collections.deque(maxlen=_MAX_EP_STEPS),
        's_dot':    collections.deque(maxlen=_MAX_EP_STEPS),
        's_pos':    collections.deque(maxlen=_MAX_EP_STEPS),
    }

    ep_title = (getattr(env, '_render_label', None)
                or f'Quarter-car   (episode {getattr(env, "_episode_count", "?")})')

    gym_env_title = 'Quarter-Car gym env'

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
    draw_sky(ax_s, RENDER_YLIM)
    ax_s.set_xlim(RENDER_XLIM)
    ax_s.set_ylim(RENDER_YLIM)
    ax_s.set_xlabel('Longitudinal position [cm]', fontsize=11, color='#1b1f24')
    ax_s.set_ylabel('Height [cm]', fontsize=11, color='#1b1f24')
    ax_s.tick_params(labelsize=10, direction='in', colors='#1b1f24')
    ax_s.grid(True, axis='x', lw=0.3, alpha=0.35, color='#b8c4d0', linestyle=':')
    ax_s.set_axisbelow(True)
    ax_s.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{(x + env._s_pos) * 100:.0f}')
    )
    ax_s.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f'{y * 100:.0f}')
    )

    road_line, = ax_s.plot([], [], '-', color='#e9edf1', lw=1.8, zorder=2,
                           label=r'Road profile $\zeta(x)$')

    road = Road(ax_s)
    _hw = RENDER_DA_W / 2
    if RENDER_REALISTIC_WHEELS:
        tire_arts = {'wheel': Wheel(ax_s, 0.0, RENDER_Y_W_NOM, r'$m_W$', _WHEEL_GEOMETRY)}
    else:
        ground_sym,   = ax_s.plot([], [], '-', color=RENDER_C_GROUND, lw=1.5, zorder=2)
        contact_stem, = ax_s.plot([], [], '-', color=RENDER_C_GROUND, lw=1.8, zorder=6)
        contact_dot,  = ax_s.plot([], [], 'o', color=RENDER_C_GROUND, ms=10,  zorder=7)

        _hw = RENDER_DA_W / 2
        tire_spring,          = ax_s.plot([], [], '-', color=RENDER_C_SPRING, lw=2.0, zorder=4)
        tire_damp_rod,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
        tire_damp_lower_rod,  = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
        tire_damp_cyl,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=2.0, zorder=4)
        tire_damp_pist = Rectangle(
            (RENDER_DA_X - _hw + 0.01, 0), RENDER_DA_W - 0.02, RENDER_DA_PIST_H,
            fc=RENDER_C_DAMPER, ec='none', zorder=4)
        ax_s.add_patch(tire_damp_pist)

        mw_patch = Rectangle(
            (-RENDER_W_MW / 2, RENDER_Y_W_NOM - RENDER_H_MW / 2),
            RENDER_W_MW, RENDER_H_MW,
            fc=RENDER_C_MW, ec='#333333', lw=1.2, zorder=5,
        )
        ax_s.add_patch(mw_patch)
        mw_dot   = ax_s.plot(0, RENDER_Y_W_NOM, 'o', color='#333333', ms=4, zorder=7)[0]
        mw_label = ax_s.text(-RENDER_W_MW / 2 + 0.05, RENDER_Y_W_NOM,
                              r'$m_W$', ha='left', va='center',
                              fontsize=11, color='white', zorder=7)
        tire_arts = dict(
            contact_stem=contact_stem, contact_dot=contact_dot, ground_sym=ground_sym,
            tire_spring=tire_spring, tire_damp_rod=tire_damp_rod,
            tire_damp_lower_rod=tire_damp_lower_rod, tire_damp_cyl=tire_damp_cyl,
            tire_damp_pist=tire_damp_pist, mw_patch=mw_patch, mw_dot=mw_dot, mw_label=mw_label,
        )

    susp_spring,          = ax_s.plot([], [], '-', color=RENDER_C_SPRING, lw=2.0 * _SK, zorder=4)
    susp_damp_rod,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5 * _SK, zorder=4)
    susp_damp_lower_rod,  = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5 * _SK, zorder=4)
    susp_damp_cyl,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=2.0 * _SK, zorder=4)
    susp_damp_pist = Rectangle(
        (RENDER_DA_X - _hw * _SK + 0.01 * _SK, 0), (RENDER_DA_W - 0.02) * _SK, RENDER_DA_PIST_H * _SK,
        fc=RENDER_C_DAMPER, ec='none', zorder=4)
    ax_s.add_patch(susp_damp_pist)

    mb_patch = Rectangle(
        (-RENDER_W_MB / 2, RENDER_Y_B_NOM - RENDER_H_MB / 2),
        RENDER_W_MB, RENDER_H_MB,
        fc=RENDER_C_MB, ec='#333333', lw=1.2, zorder=5,
    )
    ax_s.add_patch(mb_patch)
    mb_dot   = ax_s.plot(0, RENDER_Y_B_NOM, 'o', color='#333333', ms=4, zorder=7)[0]
    mb_label = ax_s.text(-RENDER_W_MB / 2 + 0.05, RENDER_Y_B_NOM,
                          r'$m_B$', ha='left', va='center',
                          fontsize=11, color='#1b1f24', zorder=7)

    hud = Hud(ax_s, [('Time', 't'), ('Position', 's'), ('Speed', 'v'), ('Body heave', 'zb'), ('Reward', 'R'), ('Bumps', 'b')])

    ax_s.legend(fontsize=10, loc='upper right', frameon=False, labelcolor='#1b1f24',
                handlelength=1.5, borderpad=0.4)

    ts = {}
    for i, ax in enumerate(ax_r):
        k1, k2, ylabel, c1, c2 = ts_specs[i]
        _label1 = r'$z_B$' if k1 == 'z_B' else (r'$v$' if k1 == 's_dot' else None)
        ts[k1], = ax.plot([], [], '-', color=c1, lw=1.2, label=_label1)
        if k2:
            _label2 = r'$z_W$' if k2 == 'z_W' else k2
            ts[k2], = ax.plot([], [], '--', color=c2, lw=1.2, label=_label2)

        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9.5, direction='in', top=False, right=False, colors='#1b1f24')
        ax.grid(True, lw=0.25, alpha=0.35, color='gray', linestyle=':')
        ax.axhline(0, color='#8fa3b8', lw=0.8)
        ax.set_ylabel(ylabel, fontsize=11, color='#1b1f24')

        if k1 == 'z_B':
            ax.legend(fontsize=9, loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=3,
                      frameon=False, labelcolor='#1b1f24', columnspacing=1.2, handlelength=1.8)

        if i < len(ax_r) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('$t$ [s]', fontsize=11, color='#1b1f24')

    bump_vlines, bump_sig = build_bump_vlines(env, ax_r)

    v_line, = ax_s.plot([_ARR_X0, _ARR_X0], [_ARR_Y_V, _ARR_Y_V], '-',
                        color='#0b6e8a', lw=2.2, zorder=8)
    v_head, = ax_s.plot([_ARR_X0],          [_ARR_Y_V],            '>',
                        color='#0b6e8a', ms=7,  zorder=8)
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

    env._fig  = fig
    env._ax_s = ax_s
    env._ax_r = ax_r
    if RENDER_MUSIC and env.render_mode != 'none':
        init_music(env, fig)
    env._artists = {
        'road_line':           road_line,
        'road':                road,
        **tire_arts,
        'susp_spring':         susp_spring,
        'susp_damp_rod':       susp_damp_rod,
        'susp_damp_lower_rod': susp_damp_lower_rod,
        'susp_damp_cyl':       susp_damp_cyl,
        'susp_damp_pist':      susp_damp_pist,
        'mb_patch':            mb_patch,
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
    z_B = float(x[5])
    z_W = z_B + float(x[2])
    h   = env._ren_hist
    h['t'].append(env._t)
    h['z_B'].append(z_B)
    h['z_W'].append(z_W)
    h['z_B_ddot'].append(env._last_z_B_ddot)
    h['F'].append(0.0)
    h['s_dot'].append(float(env._v))
    h['s_pos'].append(env._s_pos)


def update_artists(env):
    art = env._artists
    ys  = env._y_scale
    x   = env._state

    z_B    = float(x[5])
    z_W    = z_B + float(x[2])
    zeta_0 = float(env._road.get_height_at(env._s_pos))

    y_road_0 = RENDER_GROUND_Y + zeta_0 * ys
    if RENDER_REALISTIC_WHEELS:
        y_W = y_road_0 + _WHEEL_RY + min(z_W - zeta_0, 0.0) * ys
        y_B = y_W + (RENDER_Y_B_NOM - RENDER_Y_W_NOM) + (z_B - z_W) * ys
    else:
        y_W = RENDER_Y_W_NOM + RENDER_GROUND_Y + z_W * ys
        y_B = RENDER_Y_B_NOM + RENDER_GROUND_Y + z_B * ys

    s_q    = env._s_pos + _ROAD_X
    road_h = env._road.get_height_array_pos(s_q) * ys + RENDER_GROUND_Y
    art['road_line'].set_data(_ROAD_X, road_h)

    ax_s = env._ax_s
    art['road'].update(_ROAD_X, road_h, env._s_pos)

    h = env._ren_hist

    if RENDER_REALISTIC_WHEELS:
        art['wheel'].update(y_W, y_road_0, abs(zeta_0) > 1e-6, env._s_pos / _WHEEL_R,
                            wheel_radius_x(ax_s, _WHEEL_RY))
        hub_h = _HUB_H
    else:
        y_road_0 += Y_LINE_OFFSET
        gx, gy = ground_symbol_xy(0.0, y_road_0, half_w=RENDER_W_MB / 2 + 0.15)
        art['ground_sym'].set_data(gx, gy)
        art['contact_stem'].set_data([0.0, 0.0], [y_road_0 - Y_LINE_OFFSET + RENDER_CONTACT_STEM, y_road_0])
        art['contact_dot'].set_data([0.0], [y_road_0 - Y_LINE_OFFSET])
        on_obstacle = abs(zeta_0) > 1e-6
        art['contact_dot'].set_color('#e74c3c' if on_obstacle else RENDER_C_GROUND)

        y_tire_top = y_W - RENDER_H_MW / 2
        art['tire_spring'].set_data(
            *spring_xy(RENDER_SP_X, y_tire_top, y_road_0, RENDER_SP_N, RENDER_SP_W))
        u_rod, l_rod, cyl_xy, pr = _damper_xy(RENDER_DA_X, y_tire_top, y_road_0,
                                              RENDER_DA_CYL_H_TIRE)
        art['tire_damp_rod'].set_data(*u_rod)
        art['tire_damp_lower_rod'].set_data(*l_rod)
        art['tire_damp_cyl'].set_data(*cyl_xy)
        art['tire_damp_pist'].set_xy((pr[0], pr[1]))
        art['tire_damp_pist'].set_height(pr[3])

        art['mw_patch'].set_xy((-RENDER_W_MW / 2, y_W - RENDER_H_MW / 2))
        art['mw_dot'].set_data([0], [y_W])
        art['mw_label'].set_position((-RENDER_W_MW / 2 + 0.05, y_W))
        hub_h = RENDER_H_MW

    y_susp_bot = y_W + hub_h / 2
    y_susp_top = y_B - RENDER_H_MB / 2
    if y_susp_top > y_susp_bot + 0.05:
        art['susp_spring'].set_data(
            *spring_xy(RENDER_SP_X, y_susp_top, y_susp_bot, RENDER_SP_N, RENDER_SP_W * _SK))
        u_rod, l_rod, cyl_xy, pr = _damper_xy(RENDER_DA_X, y_susp_top, y_susp_bot,
                                              RENDER_DA_CYL_H_SUSP, _SK)
        art['susp_damp_rod'].set_data(*u_rod)
        art['susp_damp_lower_rod'].set_data(*l_rod)
        art['susp_damp_cyl'].set_data(*cyl_xy)
        art['susp_damp_pist'].set_xy((pr[0], pr[1]))
        art['susp_damp_pist'].set_height(pr[3])

    art['mb_patch'].set_xy((-RENDER_W_MB / 2, y_B - RENDER_H_MB / 2))
    art['mb_dot'].set_data([0], [y_B])
    art['mb_label'].set_position((-RENDER_W_MB / 2 + 0.05, y_B))

    obst_str = f'{int(env._bumps_passed)}/{env._bumps_total}' if env._bumps_total else '-'
    if getattr(env, '_holes_total', 0):
        obst_str += f' H{int(env._holes_passed)}/{env._holes_total}'
    art['hud'].update(
        t=f'{env._t:.2f} s', s=f'{env._s_pos:.1f} m', v=f'{env._v*3.6:.1f} km/h',
        b=obst_str, zb=f'{z_B*1000:+.1f} mm', R=f'{env._episode_reward:.1f}',
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
        'z_B':      np.array(h['z_B']) * 100.0,
        'z_W':      np.array(h['z_W']) * 100.0,
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
