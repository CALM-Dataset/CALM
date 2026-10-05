import os
import collections
from typing import Tuple

import numpy as np

from .jukebox import Jukebox

from gym_env.half_car.config.render_params import (
    RENDER_Y_W_NOM, RENDER_Y_B_NOM,
    RENDER_H_MW, RENDER_W_MW,
    RENDER_H_MB, RENDER_W_MB,
    RENDER_XLIM, RENDER_YLIM, RENDER_Y_SCALE,
    RENDER_MUSIC, RENDER_MUSIC_MUTED, RENDER_MUSIC_TRACK, RENDER_MUSIC_VOLUME, RENDER_MUSIC_JUKEBOX,
    RENDER_ROAD_HALF, RENDER_ROAD_N,
    RENDER_C_MB, RENDER_C_MW, RENDER_C_SPRING, RENDER_C_DAMPER,
    RENDER_C_ROAD, RENDER_C_GROUND,
    RENDER_SP_X, RENDER_SP_W, RENDER_SP_N,
    RENDER_DA_X, RENDER_DA_W, RENDER_DA_PIST_H, RENDER_DA_PIST_FRAC,
    RENDER_DA_LOWER_STEM, RENDER_DA_CYL_H_SUSP, RENDER_DA_CYL_H_TIRE,
    RENDER_CONTACT_STEM, RENDER_GROUND_Y,
    Y_LINE_OFFSET,
    RENDER_BUMP_MARKERS,
    RENDER_ARR_X, RENDER_ARR_Y, RENDER_ARR_LEN,
    RENDER_L, RENDER_LF, RENDER_LR, RENDER_X_FRONT, RENDER_X_REAR,
    RENDER_MB_OVERHANG, RENDER_OBSTACLE_FLAG_M, RENDER_MAX_EP_STEPS,
)

_ROAD_X = np.linspace(-RENDER_ROAD_HALF, RENDER_ROAD_HALF, RENDER_ROAD_N)
_MAX_EP_STEPS = RENDER_MAX_EP_STEPS

_ARR_X0      = RENDER_ARR_X
_ARR_Y_V     = RENDER_ARR_Y
_ARR_MAX_LEN = RENDER_ARR_LEN

_BUMP_COLOURS = ['#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#d35400']


def _spring_xy(x_c: float, y_top: float, y_bot: float,
               n: int = 8, w: float = 0.18) -> Tuple[np.ndarray, np.ndarray]:
    n_pts = 2 * n + 2
    ys = np.linspace(y_top, y_bot, n_pts)
    xs = np.full(n_pts, x_c)
    idx = np.arange(1, n_pts - 1)
    xs[1:-1] = x_c + w * np.where(idx % 2 == 1, 1.0, -1.0)
    return xs, ys


def _damper_xy(x_c: float, y_top: float, y_bot: float, cyl_h: float):
    hw      = RENDER_DA_W / 2
    gap     = max(y_top - y_bot, 0.05)
    cyl_bot = y_bot + RENDER_DA_LOWER_STEM
    cyl_top = cyl_bot + cyl_h
    pist_h  = RENDER_DA_PIST_H
    pist_top = float(np.clip(
        y_bot + gap * RENDER_DA_PIST_FRAC,
        cyl_bot + pist_h + 0.005,
        cyl_top - 0.005,
    ))
    pist_bot = pist_top - pist_h
    upper_rod_xy = ([x_c, x_c], [pist_top, y_top])
    lower_rod_xy = ([x_c, x_c], [y_bot,    cyl_bot])
    cyl_xy       = ([x_c - hw, x_c - hw, x_c + hw, x_c + hw],
                    [cyl_top,  cyl_bot,   cyl_bot,   cyl_top])
    m = 0.01
    pist_rect = (x_c - hw + m, pist_bot, 2 * hw - 2 * m, pist_h)
    return upper_rod_xy, lower_rod_xy, cyl_xy, pist_rect


def _ground_symbol_xy(x_c: float, y: float, half_w: float = 0.55):
    return np.array([x_c - half_w, x_c + half_w]), np.array([y, y])


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
                            ha='right', va='center', fontsize=8, color='#333333',
                            zorder=7)

    a['susp_spring'],         = ax.plot([], [], '-', color=RENDER_C_SPRING, lw=2.0, zorder=4)
    a['susp_damp_rod'],       = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
    a['susp_damp_lower_rod'], = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
    a['susp_damp_cyl'],       = ax.plot([], [], '-', color=RENDER_C_DAMPER, lw=2.0, zorder=4)
    a['susp_damp_pist'] = Rectangle((x_c + RENDER_DA_X - _hw + 0.01, 0),
                                    RENDER_DA_W - 0.02, RENDER_DA_PIST_H,
                                    fc=RENDER_C_DAMPER, ec='none', zorder=4)
    ax.add_patch(a['susp_damp_pist'])

    a['ground_sym'],   = ax.plot([], [], '-', color=RENDER_C_GROUND, lw=1.5, zorder=2)
    a['contact_stem'], = ax.plot([], [], '-', color=RENDER_C_GROUND, lw=1.8, zorder=6)
    a['contact_dot'],  = ax.plot([], [], 'o', color=RENDER_C_GROUND, ms=9,  zorder=7)
    return a


def _update_axle(a, x_c: float, y_W: float, y_B_corner: float,
                 y_road: float, on_obstacle: bool, ys: float):
    sp_x = x_c + RENDER_SP_X
    da_x = x_c + RENDER_DA_X

    y_line = y_road + Y_LINE_OFFSET
    gx, gy = _ground_symbol_xy(x_c, y_line, half_w=RENDER_W_MB / 2 + 0.15)
    a['ground_sym'].set_data(gx, gy)
    a['contact_stem'].set_data([x_c, x_c], [y_line - Y_LINE_OFFSET + RENDER_CONTACT_STEM, y_line])
    a['contact_dot'].set_data([x_c], [y_line - Y_LINE_OFFSET])
    a['contact_dot'].set_color('#e74c3c' if on_obstacle else RENDER_C_GROUND)

    y_tire_top = y_W - RENDER_H_MW / 2
    a['tire_spring'].set_data(*_spring_xy(sp_x, y_tire_top, y_line, RENDER_SP_N, RENDER_SP_W))
    u_rod, l_rod, cyl_xy, pr = _damper_xy(da_x, y_tire_top, y_line, RENDER_DA_CYL_H_TIRE)
    a['tire_damp_rod'].set_data(*u_rod)
    a['tire_damp_lower_rod'].set_data(*l_rod)
    a['tire_damp_cyl'].set_data(*cyl_xy)
    a['tire_damp_pist'].set_xy((pr[0], pr[1]))
    a['tire_damp_pist'].set_height(pr[3])

    a['mw_patch'].set_xy((x_c - RENDER_W_MW / 2, y_W - RENDER_H_MW / 2))
    a['mw_dot'].set_data([x_c], [y_W])
    a['mw_label'].set_position((x_c - RENDER_W_MW / 2 - 0.12, y_W))

    y_susp_bot = y_W + RENDER_H_MW / 2
    y_susp_top = y_B_corner - RENDER_H_MB / 2
    if y_susp_top > y_susp_bot + 0.05:
        a['susp_spring'].set_data(*_spring_xy(sp_x, y_susp_top, y_susp_bot, RENDER_SP_N, RENDER_SP_W))
        u_rod, l_rod, cyl_xy, pr = _damper_xy(da_x, y_susp_top, y_susp_bot, RENDER_DA_CYL_H_SUSP)
        a['susp_damp_rod'].set_data(*u_rod)
        a['susp_damp_lower_rod'].set_data(*l_rod)
        a['susp_damp_cyl'].set_data(*cyl_xy)
        a['susp_damp_pist'].set_xy((pr[0], pr[1]))
        a['susp_damp_pist'].set_height(pr[3])


def _build_ts_specs(env):
    specs = []
    flags = getattr(env, "_ts_flags", {})
    if flags.get("z", False):
        specs.append(([('z_B',  r'$z_B$',     '#2c6fad', '-'),
                       ('z_Wf', r'$z_{W,f}$', '#c0392b', '--'),
                       ('z_Wr', r'$z_{W,r}$', '#27ae60', ':')],
                      r'$z\ \mathrm{[cm]}$'))
        specs.append(([('phi', r'$\varphi$', '#8e44ad', '-')],
                      r'$\varphi\ \mathrm{[deg]}$'))
    if flags.get("f", False):
        specs.append(([('F', r'$F_D$', '#27ae60', '-')], r'$F_D\ \mathrm{[N]}$'))
    if flags.get("speed", False):
        specs.append(([('s_dot', r'$v$', '#7b2d8b', '-')], r'$v\ \mathrm{[km/h]}$'))
    if flags.get("z_ddot", False):
        specs.append(([('z_B_ddot', r'$\ddot{z}_B$', '#222222', '-')],
                      r'$\ddot{z}_B\ \mathrm{[m/s^2]}$'))
    return specs


def _on_obstacle(road, s, zeta, thresh=RENDER_OBSTACLE_FLAG_M):
    for x0, A, L in (getattr(road, "_bumps", None) or []):
        if L is not None and x0 <= s <= x0 + L:
            return True
    return abs(zeta) > thresh


def _render_bumps(road):
    bumps = getattr(road, '_bumps', None) or []
    if bumps:
        return bumps
    s_bump = getattr(road, '_s_bump', None)
    if s_bump is not None:
        return [(float(s_bump), None, None)]
    return []


def _bump_signature(bumps):
    return tuple((round(x0, 3),
                  None if A is None else round(A, 4),
                  None if L is None else round(L, 3)) for x0, A, L in bumps)


def _build_bump_vlines(env, ax_r):
    bump_vlines: list[list] = []
    road   = getattr(env, '_road', None)
    bumps  = _render_bumps(road)
    labels = getattr(road, '_bump_labels', None) or []
    if not (RENDER_BUMP_MARKERS and ax_r):
        return bump_vlines, _bump_signature(bumps)

    for i, (x0, A, L) in enumerate(bumps):
        col  = _BUMP_COLOURS[i % len(_BUMP_COLOURS)]
        name = labels[i] if i < len(labels) else None
        if A is None:
            name  = name or f'Bump {i+1}'
            label = name
            ls    = '--'
        elif A < 0:
            name  = name or f'Hole {i+1}'
            label = f'{name}  ($D$ = {abs(A)*100:.1f} cm, $W$ = {L:.2f} m)'
            ls    = ':'
        else:
            name  = name or f'Bump {i+1}'
            label = f'{name}  ($H$ = {A*100:.1f} cm, $W$ = {L:.2f} m)'
            ls    = '--'
        lines = []
        for j, ax in enumerate(ax_r):
            vl = ax.axvline(x=0, color=col, lw=0.8, ls=ls, alpha=0.55,
                            label=(label if j == 0 else None), visible=False)
            lines.append(vl)
        bump_vlines.append(lines)

    if bump_vlines:
        ax_r[0].legend(fontsize=6.5, loc='upper left',
                       framealpha=0.85, edgecolor='#cccccc', borderpad=0.3)
    return bump_vlines, _bump_signature(bumps)


def render_env(env):
    if env.render_mode == 'none':
        return None
    if env._ren_hist is None:
        init_render(env)
    push_history(env)
    update_artists(env)
    jb = getattr(env, '_jukebox', None)
    if jb is not None:
        jb.tick()
    if env.render_mode == 'human':
        import matplotlib.pyplot as plt
        _ensure_play_keys(env)
        hud = getattr(env, '_speed_hud', None)
        while getattr(env, '_paused', False) and getattr(env, '_nav_request', None) is None:
            if hud is not None:
                hud.set_text('|| paused')
            env._fig.canvas.draw_idle()
            plt.pause(0.05)
        speed = float(getattr(env, '_render_speed', 1.0))
        if hud is not None:
            if speed > 1:   hud.set_text(f'>> {speed:g}x')
            elif speed < 1: hud.set_text(f'<< {speed:g}x')
            else:           hud.set_text('')
        env._frame_i = int(getattr(env, '_frame_i', 0)) + 1
        if speed >= 1.0:
            stride = max(1, int(round(speed)))
            if env._frame_i % stride == 0:
                env._fig.canvas.draw_idle()
                plt.pause(1e-3)
            else:
                env._fig.canvas.flush_events()
        else:
            env._fig.canvas.draw_idle()
            plt.pause(min(0.25, 0.02 / speed))
        if getattr(env, "_freeze_render", False):
            env._freeze_render = False
            plt.show(block=True)
        return None
    env._fig.canvas.draw()
    buf = env._fig.canvas.buffer_rgba()
    w, h = env._fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return img[..., :3].copy()


_SPEED_SEQ = (0.1, 0.25, 0.5, 1, 2, 4, 8, 16)
_ZOOM_MIN, _ZOOM_MAX, _ZOOM_STEP = 0.3, 6.0, 1.25
_CONTROLS_TEXT = ('space   pause / resume\n'
                  'f  /  d    faster / slower\n'
                  '+  /  −    zoom in / out  (or scroll)\n'
                  '0   reset zoom\n'
                  '→  /  ←    next / prev scenario\n'
                  'r   replay      q   quit')


def _grab_key_focus(fig):
    mgr = getattr(fig.canvas, 'manager', None)
    win = getattr(mgr, 'window', None)
    if win is None:
        return
    for attr in ('activateWindow', 'raise_', 'focus_force', 'focus_set', 'show'):
        try:
            getattr(win, attr)()
        except Exception:
            pass
    try:
        fig.canvas.setFocus()
    except Exception:
        pass


def _free_matplotlib_keys():
    import matplotlib as mpl
    for km, keys in (('keymap.fullscreen', ('f',)),
                     ('keymap.grid', ('g', 'G')),
                     ('keymap.save', ('s',))):
        try:
            mpl.rcParams[km] = [k for k in mpl.rcParams[km] if k not in keys]
        except Exception:
            pass


def _ensure_play_keys(env):
    if getattr(env, '_play_keys_cid', None) is not None:
        return
    fig = getattr(env, '_fig', None)
    if fig is None:
        return
    _free_matplotlib_keys()
    if not hasattr(env, '_render_speed'):
        env._render_speed = 1
    if not hasattr(env, '_nav_request'):
        env._nav_request = None
    env._paused = False
    env._speed_hud = fig.text(0.988, 0.012, '', ha='right', va='bottom',
                              fontsize=9, color='#c0392b', fontweight='bold')
    env._info_dot = fig.text(0.985, 0.965, '?', ha='center', va='center',
                             fontsize=11, color='#2c6fad', fontweight='bold',
                             bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                                       edgecolor='#2c6fad', lw=1.2))
    env._info_tip = fig.text(
        0.985, 0.945, _CONTROLS_TEXT, ha='right', va='top', fontsize=8,
        family='monospace', visible=False, zorder=20,
        bbox=dict(facecolor='white', alpha=0.95, edgecolor='#2c6fad',
                  boxstyle='round,pad=0.5', lw=0.8))

    if not hasattr(env, '_render_zoom'):
        env._render_zoom = 1.0

    def _apply_zoom():
        ax = getattr(env, '_ax_s', None)
        if ax is None:
            return
        z = float(getattr(env, '_render_zoom', 1.0))
        (x0, x1), (y0, y1) = RENDER_XLIM, RENDER_YLIM
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hx, hy = (x1 - x0) / 2.0 / z, (y1 - y0) / 2.0 / z
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)
        frac = 1.0 if z >= 1.0 else min(max((z - _ZOOM_MIN) / (1.0 - _ZOOM_MIN), 0.0), 1.0)
        env._y_scale = 1.0 + (RENDER_Y_SCALE - 1.0) * frac
        s = min(max(z, 0.3), 2.5)
        for t in ax.texts:
            if getattr(t, '_skip_zoom', False):
                continue
            if not hasattr(t, '_base_fs'):
                t._base_fs = t.get_fontsize()
            t.set_fontsize(t._base_fs * s)
        for ln in ax.lines:
            if not hasattr(ln, '_base_lw'):
                ln._base_lw, ln._base_ms = ln.get_linewidth(), ln.get_markersize()
            ln.set_linewidth(ln._base_lw * s)
            ln.set_markersize(ln._base_ms * s)
        try:
            update_artists(env)
        except Exception:
            pass
        fig.canvas.draw_idle()

    def _zoom_by(factor):
        env._render_zoom = min(max(float(getattr(env, '_render_zoom', 1.0)) * factor,
                                   _ZOOM_MIN), _ZOOM_MAX)
        _apply_zoom()

    def _on_key(ev):
        k = ev.key
        if k == ' ':
            env._paused = not getattr(env, '_paused', False)
        elif k == 'f':
            cur = float(getattr(env, '_render_speed', 1.0))
            i = _SPEED_SEQ.index(cur) if cur in _SPEED_SEQ else _SPEED_SEQ.index(1)
            env._render_speed = _SPEED_SEQ[min(i + 1, len(_SPEED_SEQ) - 1)]
        elif k == 'd':
            cur = float(getattr(env, '_render_speed', 1.0))
            i = _SPEED_SEQ.index(cur) if cur in _SPEED_SEQ else _SPEED_SEQ.index(1)
            env._render_speed = _SPEED_SEQ[max(i - 1, 0)]
        elif k in ('+', '='):
            _zoom_by(_ZOOM_STEP)
        elif k == '-':
            _zoom_by(1.0 / _ZOOM_STEP)
        elif k == '0':
            env._render_zoom = 1.0
            _apply_zoom()
        elif k == 'right':
            env._nav_request = 'next'
        elif k == 'left':
            env._nav_request = 'prev'

    def _on_scroll(ev):
        _zoom_by(_ZOOM_STEP if ev.button == 'up' else 1.0 / _ZOOM_STEP)

    def _on_move(ev):
        tip = getattr(env, '_info_tip', None)
        dot = getattr(env, '_info_dot', None)
        if tip is None or dot is None:
            return
        try:
            bb = dot.get_window_extent()
        except Exception:
            return
        over = ev.x is not None and bb.contains(ev.x, ev.y)
        if over != tip.get_visible():
            tip.set_visible(over)
            fig.canvas.draw_idle()

    env._play_keys_cid = fig.canvas.mpl_connect('key_press_event', _on_key)
    fig.canvas.mpl_connect('motion_notify_event', _on_move)
    fig.canvas.mpl_connect('scroll_event', _on_scroll)


def close_env(env):
    jb = getattr(env, '_jukebox', None)
    if jb is not None:
        jb.stop()
        env._jukebox = None
    if env._fig is not None:
        import matplotlib.pyplot as plt
        plt.close(env._fig)
        env._fig = None




def _music_label(jb) -> str:
    state = 'muted' if jb.muted else f'{jb.volume*100:.0f}%'
    return f'{jb.icon}  {jb.track.replace("_", " ")}  ·  {state}'


def _init_music(env, fig):
    jb = Jukebox(track=RENDER_MUSIC_TRACK,
                 muted=RENDER_MUSIC_MUTED,
                 volume=RENDER_MUSIC_VOLUME,
                 jukebox=RENDER_MUSIC_JUKEBOX)
    env._jukebox = jb
    if jb.ok:
        jb.start()
    else:
        import sys
        print('[render] music is on but no audio backend; install pygame to hear it '
              '(pip install pygame)', file=sys.stderr)

    if env.render_mode != 'human':
        env._music_caption = fig.text(
            0.012, 0.02, _music_label(jb), ha='left', va='bottom',
            fontsize=9, color='#7b2d8b',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc',
                      boxstyle='round,pad=0.3', lw=0.6))
        return

    if _mount_toolbar_controls(env, fig, jb):
        return

    from matplotlib.widgets import Button, Slider
    gs = env._ax_s.get_subplotspec().get_gridspec()
    gs.update(bottom=0.20)
    ax_mute = fig.add_axes([0.020, 0.05, 0.040, 0.07])
    ax_vol  = fig.add_axes([0.105, 0.075, 0.150, 0.030])
    btn_mute = Button(ax_mute, jb.icon, color='#eef2f7', hovercolor='#dde6f0')
    btn_mute.label.set_fontsize(14)
    slider   = Slider(ax_vol, 'vol', 0.0, 1.0, valinit=jb.volume,
                      color='#7b2d8b', valfmt='%.2f')
    slider.valtext.set_fontsize(7)
    slider.label.set_fontsize(7)

    def _toggle(_):
        jb.toggle_mute()
        btn_mute.label.set_text(jb.icon)
        fig.canvas.draw_idle()

    def _vol(val):
        jb.set_volume(val)
        btn_mute.label.set_text(jb.icon)

    btn_mute.on_clicked(_toggle)
    slider.on_changed(_vol)
    env._music_widgets = (btn_mute, slider)


def _mount_toolbar_controls(env, fig, jb) -> bool:
    toolbar = getattr(getattr(fig.canvas, 'manager', None), 'toolbar', None)
    if toolbar is None:
        return False
    try:
        import tkinter as tk
    except Exception:
        return False
    if not isinstance(toolbar, tk.Frame):
        return False

    def _toggle():
        jb.toggle_mute()
        mute.config(text=jb.icon)

    def _vol(v):
        jb.set_volume(int(v) / 100.0)
        mute.config(text=jb.icon)

    sep = tk.Frame(toolbar, width=2, bg='#bbbbbb')
    mute = tk.Button(toolbar, text=jb.icon, command=_toggle, relief=tk.FLAT,
                     borderwidth=0, highlightthickness=0, font=('DejaVu Sans', 15),
                     takefocus=0)
    vol = tk.Scale(toolbar, from_=0, to=100, orient=tk.HORIZONTAL, showvalue=False,
                   length=150, width=14, sliderlength=20, command=_vol,
                   relief=tk.FLAT, borderwidth=1, highlightthickness=0,
                   troughcolor='#c9c9c9', activebackground='#7b2d8b')
    vol.set(int(jb.volume * 100))
    sep.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=4)
    mute.pack(side=tk.LEFT, padx=2)
    vol.pack(side=tk.LEFT, padx=(2, 6))

    env._music_widgets = (sep, mute, vol)
    return True

def wait_for_episode_nav(env, allow_prev: bool = True) -> str:
    import matplotlib.pyplot as plt
    raw = getattr(env, 'unwrapped', env)
    fig = getattr(raw, '_fig', None)
    if fig is None:
        return 'next'

    fig.canvas.draw_idle()

    _nav = [None]

    def _on_key(event):
        if event.key in ('right', ' '):
            _nav[0] = 'next'
        elif event.key == 'left':
            _nav[0] = 'prev' if allow_prev else 'replay'
        elif event.key == 'r' and allow_prev:
            _nav[0] = 'replay'
        elif event.key in ('q', 'escape'):
            _nav[0] = 'quit'

    cid = fig.canvas.mpl_connect('key_press_event', _on_key)
    while _nav[0] is None:
        plt.pause(0.05)
        if not plt.fignum_exists(fig.number):
            _nav[0] = 'quit'
            break

    fig.canvas.mpl_disconnect(cid)
    return _nav[0]


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
        fig.suptitle(ep_title, fontsize=9, color='#444444', x=0.5, y=0.98)
        gs  = GridSpec(1, 2, figure=fig, width_ratios=[3, 2],
                       left=0.06, right=0.97, bottom=0.09, top=0.90, wspace=0.40)
        ax_s = fig.add_subplot(gs[0, 0])
        n_ts = len(ts_specs)
        gs_r = gs[0, 1].subgridspec(n_ts, 1, hspace=0.10)
        ax_r = [fig.add_subplot(gs_r[i]) for i in range(n_ts)]
    else:
        fig  = plt.figure(figsize=(9, 7))
        fig.suptitle(ep_title, fontsize=9, color='#444444', x=0.5, y=0.98)
        ax_s = fig.add_subplot(1, 1, 1)
        ax_r = []

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(gym_env_title)
    fig.set_facecolor('#fbfbfd')

    import matplotlib.ticker as ticker
    ax_s.set_facecolor('#f4f8fc')
    ax_s.set_xlim(RENDER_XLIM)
    ax_s.set_ylim(RENDER_YLIM)
    ax_s.set_xlabel('Longitudinal position [cm]', fontsize=9)
    ax_s.set_ylabel('Height [cm]', fontsize=9)
    ax_s.tick_params(labelsize=8, direction='in')
    ax_s.grid(True, axis='x', lw=0.3, alpha=0.35, color='#b8c4d0', linestyle=':')
    ax_s.set_axisbelow(True)
    ax_s.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{(x + env._s_pos) * 100:.0f}')
    )
    ax_s.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f'{y * 100:.0f}')
    )

    road_line, = ax_s.plot([], [], '-', color=RENDER_C_ROAD, lw=1.5, zorder=2,
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
                          fontsize=9, color='#333333', zorder=7)

    status_text = ax_s.text(
        0.02, 0.97, '', transform=ax_s.transAxes,
        va='top', ha='left', fontsize=8,
        bbox=dict(facecolor='white', alpha=0.92, edgecolor='#888888',
                  boxstyle='square,pad=0.45', lw=0.7),
        zorder=9,
    )
    status_text._skip_zoom = True

    ax_s.legend(fontsize=7.5, loc='upper right', frameon=False,
                handlelength=1.5, borderpad=0.4)

    ts = {}
    for i, ax in enumerate(ax_r):
        series, ylabel = ts_specs[i]
        for key, label, colour, style in series:
            ts[key], = ax.plot([], [], style, color=colour, lw=1.2, label=label)

        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7.5, direction='in', top=False, right=False)
        ax.grid(True, lw=0.25, alpha=0.35, color='gray', linestyle=':')
        ax.axhline(0, color='#bbbbbb', lw=0.6)
        ax.set_ylabel(ylabel, fontsize=8.5)

        if len(series) > 1:
            ax.legend(fontsize=7.0, loc='upper left', ncol=len(series),
                      framealpha=0.85, edgecolor='#cccccc', borderpad=0.3)

        if i < len(ax_r) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel('$t$ [s]', fontsize=8.5)

    bump_vlines, bump_sig = _build_bump_vlines(env, ax_r)

    v_line, = ax_s.plot([_ARR_X0, _ARR_X0], [_ARR_Y_V, _ARR_Y_V], '-',
                        color='#7b2d8b', lw=2.2, zorder=8)
    v_head, = ax_s.plot([_ARR_X0],          [_ARR_Y_V],            '>',
                        color='#7b2d8b', ms=7,  zorder=8)
    ax_s.text(_ARR_X0 - 0.15, _ARR_Y_V, r'$v$', ha='right', va='center',
              fontsize=8, color='#7b2d8b', zorder=8)
    ax_s.plot([_ARR_X0 + _ARR_MAX_LEN] * 2,
              [_ARR_Y_V - 0.20, _ARR_Y_V + 0.20],
              color='#e74c3c', lw=2.0, ls='-', zorder=8)
    ax_s.text(_ARR_X0 + _ARR_MAX_LEN, _ARR_Y_V - 0.32,
              f'{env._rcfg.v_max * 3.6:.0f} km/h', ha='center', va='top',
              fontsize=7.0, color='#e74c3c', fontweight='bold', zorder=8)
    vmax_tick, = ax_s.plot([_ARR_X0, _ARR_X0], [_ARR_Y_V - 0.16, _ARR_Y_V + 0.16],
                           color='#7b2d8b', lw=1.1, ls='--', alpha=0.85, zorder=7)
    vmax_label = ax_s.text(_ARR_X0, _ARR_Y_V + 0.28, '', ha='center', va='bottom',
                           fontsize=6.5, color='#7b2d8b', zorder=8)

    if env.render_mode == 'human':
        plt.ion()
        plt.show(block=False)
        _grab_key_focus(fig)

    env._fig  = fig
    env._ax_s = ax_s
    env._ax_r = ax_r
    if RENDER_MUSIC and env.render_mode != 'none':
        _init_music(env, fig)
    env._artists = {
        'road_line':           road_line,
        'axle_f':              axle_f,
        'axle_r':              axle_r,
        'mb_poly':             mb_poly,
        'mb_dot':              mb_dot,
        'mb_label':            mb_label,
        'status_text':         status_text,
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

    y_Wf   = RENDER_Y_W_NOM + RENDER_GROUND_Y + z_Wf * ys
    y_Wr   = RENDER_Y_W_NOM + RENDER_GROUND_Y + z_Wr * ys
    y_Bf   = RENDER_Y_B_NOM + RENDER_GROUND_Y + z_Bf * ys
    y_Br   = RENDER_Y_B_NOM + RENDER_GROUND_Y + z_Br * ys
    y_road_f = RENDER_GROUND_Y + zeta_f * ys
    y_road_r = RENDER_GROUND_Y + zeta_r * ys

    ax_s   = env._ax_s
    x0, x1 = ax_s.get_xlim()
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
        color='#cfc8bb', edgecolor='none', zorder=1,
    )

    h = env._ren_hist

    road = env._road
    _update_axle(art['axle_f'], RENDER_X_FRONT, y_Wf, y_Bf, y_road_f,
                 _on_obstacle(road, env._s_pos, zeta_f), ys)
    _update_axle(art['axle_r'], RENDER_X_REAR,  y_Wr, y_Br, y_road_r,
                 _on_obstacle(road, env._s_pos - L, zeta_r), ys)

    art['mb_poly'].set_xy(_chassis_corners(y_Bf, y_Br))
    x_mid = (RENDER_X_FRONT + RENDER_X_REAR) / 2
    y_mid = (y_Bf + y_Br) / 2
    art['mb_dot'].set_data([x_mid], [y_mid])
    art['mb_label'].set_position((x_mid, max(y_Bf, y_Br) + RENDER_H_MB / 2 + 0.12))

    obst_str = f'{int(env._bumps_passed)}/{env._bumps_total}' if env._bumps_total else '-'
    if getattr(env, '_holes_total', 0):
        obst_str += f' H{int(env._holes_passed)}/{env._holes_total}'
    art['status_text'].set_text(
        f'$t$ = {env._t:.2f} s'
        f'    $s$ = {env._s_pos:.1f} m'
        f'    $z_B$ = {float(x[9])*1000:+.2f} mm'
        f'    $\\varphi$ = {np.degrees(phi):+.2f}°'
        f'    $\\zeta_f$ = {zeta_f*1000:.3f} mm\n'
        f'$v$ = {env._v*3.6:.1f} km/h'
        f'    obst {obst_str}'
        f'    $R$ = {env._episode_reward:.2f}'
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
    bumps   = _render_bumps(getattr(env, '_road', None))
    cur_sig = _bump_signature(bumps)
    if cur_sig != art.get('bump_sig'):
        for lines in bump_vlines:
            for vl in lines:
                vl.remove()
        bump_vlines, _ = _build_bump_vlines(env, env._ax_r)
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
