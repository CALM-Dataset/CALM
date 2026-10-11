import os
import collections
from typing import Tuple

import numpy as np

from .jukebox import Jukebox

from gym_env.quarter_car.config.render_params import (
    RENDER_Y_W_NOM, RENDER_Y_B_NOM,
    RENDER_H_MW, RENDER_W_MW,
    RENDER_H_MB, RENDER_W_MB,
    RENDER_XLIM, RENDER_YLIM,
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
)

_ROAD_X = np.linspace(-RENDER_ROAD_HALF, RENDER_ROAD_HALF, RENDER_ROAD_N)
_MAX_EP_STEPS = 4000

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


def _build_ts_specs(env):
    specs = []
    flags = getattr(env, "_ts_flags", {})
    if flags.get("z", False):
        specs.append(('z_B', 'z_W', r'$z\ \mathrm{[cm]}$', '#2c6fad', '#c0392b'))
    if flags.get("f", False):
        specs.append(('F', None, r'$F_D\ \mathrm{[N]}$', '#27ae60', None))
    if flags.get("speed", False):
        specs.append(('s_dot', None, r'$v\ \mathrm{[km/h]}$', '#7b2d8b', None))
    if flags.get("z_ddot", False):
        specs.append(('z_B_ddot', None, r'$\ddot{z}_B\ \mathrm{[m/s^2]}$', '#222222', None))
    return specs


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


def _render_env(env):
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
        while (getattr(env, '_paused', False) and getattr(env, '_nav_request', None) is None
               and not _window_gone(env)):
            if hud is not None:
                hud.set_text('|| paused')
            env._fig.canvas.draw_idle()
            plt.pause(0.05)
        speed = int(getattr(env, '_render_speed', 1))
        if hud is not None:
            hud.set_text(f'>> {speed}x' if speed > 1 else '')
        env._frame_i = int(getattr(env, '_frame_i', 0)) + 1
        if speed <= 1 or env._frame_i % speed == 0:
            env._fig.canvas.draw_idle()
            plt.pause(max(1e-4, 1e-3 / speed))
        else:
            env._fig.canvas.flush_events()
        if getattr(env, "_freeze_render", False):
            env._freeze_render = False
            plt.show(block=True)
        return None
    env._fig.canvas.draw()
    buf = env._fig.canvas.buffer_rgba()
    w, h = env._fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return img[..., :3].copy()


_SPEED_SEQ = (1, 2, 4, 8, 16)
_CONTROLS_TEXT = ('space   pause / resume\n'
                  'f  /  d    faster / slower\n'
                  '→  /  ←    next / prev scenario\n'
                  'r   replay      q   quit')


def _free_matplotlib_keys():
    import matplotlib as mpl
    for name in [k for k in mpl.rcParams if k.startswith('keymap.')]:
        mpl.rcParams[name] = []


def _window_gone(env) -> bool:
    import matplotlib.pyplot as plt
    fig = getattr(env, '_fig', None)
    if fig is None:
        return False
    return (getattr(env, '_quit_request', False) or getattr(env, '_window_closed', False)
            or not plt.fignum_exists(fig.number))


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

    def _guarded(fn):
        def wrapper(ev):
            try:
                fn(ev)
            except Exception as exc:
                import sys
                print(f'[render] ignored {type(exc).__name__}: {exc}', file=sys.stderr)
        return wrapper

    def _on_key(ev):
        k = ev.key
        if k in ('q', 'escape'):
            env._quit_request = True
        elif k == ' ':
            env._paused = not getattr(env, '_paused', False)
        elif k == 'f':
            cur = int(getattr(env, '_render_speed', 1))
            i = _SPEED_SEQ.index(cur) if cur in _SPEED_SEQ else 0
            env._render_speed = _SPEED_SEQ[min(i + 1, len(_SPEED_SEQ) - 1)]
        elif k == 'd':
            cur = int(getattr(env, '_render_speed', 1))
            i = _SPEED_SEQ.index(cur) if cur in _SPEED_SEQ else 0
            env._render_speed = _SPEED_SEQ[max(i - 1, 0)]
        elif k == 'r':
            env._nav_request = 'replay'
        elif k == 'right':
            env._nav_request = 'next'
        elif k == 'left':
            env._nav_request = 'prev'

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

    env._play_keys_cid = fig.canvas.mpl_connect('key_press_event', _guarded(_on_key))
    fig.canvas.mpl_connect('motion_notify_event', _guarded(_on_move))
    fig.canvas.mpl_connect('close_event', lambda ev: setattr(env, '_window_closed', True))


def render_env(env):
    if env.render_mode != 'human':
        return _render_env(env)
    if _window_gone(env):
        raise KeyboardInterrupt('render window closed')
    try:
        return _render_env(env)
    except Exception:
        if _window_gone(env):
            raise KeyboardInterrupt('render window closed') from None
        raise


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
                or f'Quarter-Car Env  ·  Episode {getattr(env, "_episode_count", "?")}')

    gym_env_title = 'Quarter-Car gym env'

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
                          fontsize=9, color='white', zorder=7)

    susp_spring,          = ax_s.plot([], [], '-', color=RENDER_C_SPRING, lw=2.0, zorder=4)
    susp_damp_rod,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
    susp_damp_lower_rod,  = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=1.5, zorder=4)
    susp_damp_cyl,        = ax_s.plot([], [], '-', color=RENDER_C_DAMPER, lw=2.0, zorder=4)
    susp_damp_pist = Rectangle(
        (RENDER_DA_X - _hw + 0.01, 0), RENDER_DA_W - 0.02, RENDER_DA_PIST_H,
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
                          fontsize=9, color='#333333', zorder=7)

    status_text = ax_s.text(
        0.02, 0.97, '', transform=ax_s.transAxes,
        va='top', ha='left', fontsize=8,
        bbox=dict(facecolor='white', alpha=0.92, edgecolor='#888888',
                  boxstyle='square,pad=0.45', lw=0.7),
        zorder=9,
    )

    ax_s.legend(fontsize=7.5, loc='upper right', frameon=False,
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
        ax.tick_params(labelsize=7.5, direction='in', top=False, right=False)
        ax.grid(True, lw=0.25, alpha=0.35, color='gray', linestyle=':')
        ax.axhline(0, color='#bbbbbb', lw=0.6)
        ax.set_ylabel(ylabel, fontsize=8.5)

        if k1 in ('z_B', 's_dot'):
            ax.legend(fontsize=7.5, loc='upper left',
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

    env._fig  = fig
    env._ax_s = ax_s
    env._ax_r = ax_r
    if RENDER_MUSIC and env.render_mode != 'none':
        _init_music(env, fig)
    env._artists = {
        'road_line':           road_line,
        'contact_stem':        contact_stem,
        'contact_dot':         contact_dot,
        'ground_sym':          ground_sym,
        'tire_spring':         tire_spring,
        'tire_damp_rod':       tire_damp_rod,
        'tire_damp_lower_rod': tire_damp_lower_rod,
        'tire_damp_cyl':       tire_damp_cyl,
        'tire_damp_pist':      tire_damp_pist,
        'mw_patch':            mw_patch,
        'mw_dot':              mw_dot,
        'mw_label':            mw_label,
        'susp_spring':         susp_spring,
        'susp_damp_rod':       susp_damp_rod,
        'susp_damp_lower_rod': susp_damp_lower_rod,
        'susp_damp_cyl':       susp_damp_cyl,
        'susp_damp_pist':      susp_damp_pist,
        'mb_patch':            mb_patch,
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

    y_W      = RENDER_Y_W_NOM + RENDER_GROUND_Y + z_W * ys
    y_B      = RENDER_Y_B_NOM + RENDER_GROUND_Y + z_B * ys
    y_road_0 = RENDER_GROUND_Y + zeta_0 * ys

    s_q    = env._s_pos + _ROAD_X
    road_h = env._road.get_height_array_pos(s_q) * ys + RENDER_GROUND_Y
    art['road_line'].set_data(_ROAD_X, road_h)

    ax_s = env._ax_s
    if art.get('road_fill') is not None:
        art['road_fill'].remove()
    art['road_fill'] = ax_s.fill_between(
        _ROAD_X, road_h, RENDER_YLIM[0],
        color='#cfc8bb', edgecolor='none', zorder=1,
    )

    h = env._ren_hist

    y_road_0 += Y_LINE_OFFSET
    gx, gy = _ground_symbol_xy(0.0, y_road_0, half_w=RENDER_W_MB / 2 + 0.15)
    art['ground_sym'].set_data(gx, gy)
    art['contact_stem'].set_data([0.0, 0.0], [y_road_0 - Y_LINE_OFFSET + RENDER_CONTACT_STEM, y_road_0])
    art['contact_dot'].set_data([0.0], [y_road_0 - Y_LINE_OFFSET])
    on_obstacle = abs(zeta_0) > 1e-6
    art['contact_dot'].set_color('#e74c3c' if on_obstacle else RENDER_C_GROUND)

    y_tire_top = y_W - RENDER_H_MW / 2
    art['tire_spring'].set_data(
        *_spring_xy(RENDER_SP_X, y_tire_top, y_road_0, RENDER_SP_N, RENDER_SP_W))
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

    y_susp_bot = y_W + RENDER_H_MW / 2
    y_susp_top = y_B - RENDER_H_MB / 2
    if y_susp_top > y_susp_bot + 0.05:
        art['susp_spring'].set_data(
            *_spring_xy(RENDER_SP_X, y_susp_top, y_susp_bot, RENDER_SP_N, RENDER_SP_W))
        u_rod, l_rod, cyl_xy, pr = _damper_xy(RENDER_DA_X, y_susp_top, y_susp_bot,
                                              RENDER_DA_CYL_H_SUSP)
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
    art['status_text'].set_text(
        f'$t$ = {env._t:.2f} s'
        f'    $s$ = {env._s_pos:.1f} m'
        f'    $z_B$ = {z_B*1000:+.2f} mm'
        f'    $z_W$ = {z_W*1000:+.2f} mm'
        f'    $\\zeta$ = {zeta_0*1000:.3f} mm\n'
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
