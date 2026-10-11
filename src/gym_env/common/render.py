import time
from typing import Tuple

import numpy as np

from gym_env.common.jukebox import Jukebox
from gym_env.common.render_params import (
    RENDER_BUMP_MARKERS, RENDER_MUSIC_JUKEBOX, RENDER_MUSIC_MUTED, RENDER_MUSIC_TRACK,
    RENDER_MUSIC_VOLUME, RENDER_Y_SCALE,
)

import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'stix'

_BUMP_COLOURS = ['#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#d35400']
_SPEED_SEQ = (0.1, 0.25, 0.5, 1, 2, 4, 8, 16)
_CONTROLS = (
    ('Space', 'pause / resume'),
    ('F / S', 'faster / slower'),
    ('+ / −', 'zoom (or scroll)'),
    ('0', 'reset zoom'),
    ('← / →', 'previous / next'),
    ('R', 'replay'),
    ('Q', 'quit'),
)
_CONTROLS_TEXT = '\n'.join(f'{key}    {text}' for key, text in _CONTROLS)
_ZOOM_MIN, _ZOOM_MAX, _ZOOM_STEP = 0.3, 6.0, 1.25
_NAV_KEYS = {'right': 'next', 'left': 'prev', 'r': 'replay'}
_NAV_DEBOUNCE = 0.6


def spring_xy(x_c: float, y_top: float, y_bot: float,
               n: int = 8, w: float = 0.18) -> Tuple[np.ndarray, np.ndarray]:
    n_pts = 2 * n + 2
    ys = np.linspace(y_top, y_bot, n_pts)
    xs = np.full(n_pts, x_c)
    idx = np.arange(1, n_pts - 1)
    xs[1:-1] = x_c + w * np.where(idx % 2 == 1, 1.0, -1.0)
    return xs, ys


def damper_xy(x_c: float, y_top: float, y_bot: float, cyl_h: float,
              w: float, pist_h: float, pist_frac: float, lower_stem: float):
    hw      = w / 2
    gap     = max(y_top - y_bot, 0.05)
    cyl_bot = y_bot + lower_stem
    cyl_top = cyl_bot + cyl_h
    pist_h  = pist_h
    pist_top = float(np.clip(
        y_bot + gap * pist_frac,
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


def ground_symbol_xy(x_c: float, y: float, half_w: float = 0.55):
    return np.array([x_c - half_w, x_c + half_w]), np.array([y, y])


def render_bumps(road):
    bumps = getattr(road, '_bumps', None) or []
    if bumps:
        return bumps
    s_bump = getattr(road, '_s_bump', None)
    if s_bump is not None:
        return [(float(s_bump), None, None)]
    return []


def bump_signature(bumps):
    return tuple((round(x0, 3),
                  None if A is None else round(A, 4),
                  None if L is None else round(L, 3)) for x0, A, L in bumps)


def build_bump_vlines(env, ax_r):
    bump_vlines: list[list] = []
    road   = getattr(env, '_road', None)
    bumps  = render_bumps(road)
    labels = getattr(road, '_bump_labels', None) or []
    if not (RENDER_BUMP_MARKERS and ax_r):
        return bump_vlines, bump_signature(bumps)

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
            vl = ax.axvline(x=0, color=col, lw=1.0, ls=ls, alpha=0.8,
                            label=(label if j == 0 else None), visible=False)
            lines.append(vl)
        bump_vlines.append(lines)

    if bump_vlines:
        ax_r[0].legend(fontsize=9, loc='lower left', bbox_to_anchor=(0.0, 1.03), ncol=2,
                       frameon=False, labelcolor='#1b1f24', columnspacing=1.2, handlelength=1.8)
    return bump_vlines, bump_signature(bumps)


class Hud:
    def __init__(self, ax, fields):
        from matplotlib.patches import Rectangle
        n = len(fields)
        h, y0 = 0.075, 1.02
        w = 1.0 / n
        ax.add_patch(Rectangle((0, y0), 1.0, h, transform=ax.transAxes, clip_on=False,
                               fc='white', ec='#555555', lw=0.8, zorder=9))
        self._values = []
        for i, (label, key) in enumerate(fields):
            x0 = i * w
            if i:
                ax.plot([x0, x0], [y0 + 0.012, y0 + h - 0.012], transform=ax.transAxes,
                        color='#bbbbbb', lw=0.7, clip_on=False, zorder=10)
            ax.text(x0 + w / 2, y0 + h * 0.76, label, transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, style='italic', color='#3a444e',
                    zorder=11, clip_on=False)
            t = ax.text(x0 + w / 2, y0 + h * 0.31, '', transform=ax.transAxes,
                        ha='center', va='center', fontsize=11, color='#111111',
                        zorder=11, clip_on=False)
            t._skip_zoom = True
            self._values.append((key, t))

    def update(self, **values):
        for key, t in self._values:
            t.set_text(values[key])


class RenderQuit(KeyboardInterrupt):
    pass


_DRAW_PERIOD = 1.0 / 30.0
_MAX_STALE = 0.25


def _sleep_until(plt, deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        plt.pause(min(remaining, 0.02))


def _render_env(env):
    if env.render_mode == 'none':
        return None
    mod = env._vehicle.render()
    if env._ren_hist is None:
        mod.init_render(env)
    mod.push_history(env)
    jb = getattr(env, '_jukebox', None)
    if jb is not None:
        jb.tick()
    if env.render_mode == 'human':
        import matplotlib.pyplot as plt
        _ensure_play_keys(env)
        hud = getattr(env, '_speed_hud', None)
        if getattr(env, '_paused', False):
            mod.update_artists(env)
            env._pace = None
        while (getattr(env, '_paused', False) and getattr(env, '_nav_request', None) is None
               and not _window_gone(env)):
            if hud is not None:
                hud.set_text('|| paused')
            env._fig.canvas.draw_idle()
            plt.pause(0.05)
        speed = float(getattr(env, '_render_speed', 1.0))
        if hud is not None:
            if speed > 1:   hud.set_text(f'>> {speed:g}x')
            elif speed < 1: hud.set_text(f'<< {speed:g}x')
            else:           hud.set_text('')
        now = time.perf_counter()
        pace = getattr(env, '_pace', None)
        if pace is None or pace[2] != speed or pace[3] != env._episode_count or env._t < pace[1]:
            pace = env._pace = (now, env._t, speed, env._episode_count)
        due = pace[0] + (env._t - pace[1]) / speed
        stale = now - getattr(env, '_last_draw', -1.0)
        if (stale >= _DRAW_PERIOD and now <= due) or stale >= _MAX_STALE or env._t < pace[1] + 1e-9:
            mod.update_artists(env)
            env._fig.canvas.draw_idle()
            plt.pause(1e-3)
            env._last_draw = time.perf_counter()
        else:
            env._fig.canvas.flush_events()
        _sleep_until(plt, due)
        if getattr(env, "_freeze_render", False):
            mod.update_artists(env)
            env._freeze_render = False
            plt.show(block=True)
        return None
    mod.update_artists(env)
    env._fig.canvas.draw()
    buf = env._fig.canvas.buffer_rgba()
    w, h = env._fig.canvas.get_width_height()
    img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return img[..., :3].copy()


def grab_key_focus(fig):
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
        env._render_speed = 0.5
    if not hasattr(env, '_nav_request'):
        env._nav_request = None
    env._paused = False
    env._speed_hud = fig.text(0.988, 0.012, '', ha='right', va='bottom',
                              fontsize=11, color='#a33', style='italic')
    env._info_dot = fig.text(0.985, 0.965, '?', ha='center', va='center',
                             fontsize=12, color='#444444', style='italic',
                             bbox=dict(boxstyle='circle,pad=0.3', facecolor='white',
                                       edgecolor='#777777', lw=0.9))
    env._info_tip = fig.text(
        0.985, 0.945, _CONTROLS_TEXT, ha='right', va='top', fontsize=10,
        multialignment='left', color='#222222', visible=False, zorder=20, linespacing=1.5,
        bbox=dict(facecolor='white', alpha=0.97, edgecolor='#999999',
                  boxstyle='square,pad=0.7', lw=0.7))

    if not hasattr(env, '_render_zoom'):
        env._render_zoom = 1.0

    def _apply_zoom():
        ax = getattr(env, '_ax_s', None)
        if ax is None:
            return
        z = float(getattr(env, '_render_zoom', 1.0))
        (x0, x1), (y0, y1) = env._vehicle.render().RENDER_XLIM, env._vehicle.render().RENDER_YLIM
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
            env._vehicle.render().update_artists(env)
        except Exception:
            pass
        fig.canvas.draw_idle()

    def _zoom_by(factor):
        env._render_zoom = min(max(float(getattr(env, '_render_zoom', 1.0)) * factor,
                                   _ZOOM_MIN), _ZOOM_MAX)
        _apply_zoom()

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
            cur = float(getattr(env, '_render_speed', 1.0))
            i = _SPEED_SEQ.index(cur) if cur in _SPEED_SEQ else _SPEED_SEQ.index(1)
            env._render_speed = _SPEED_SEQ[min(i + 1, len(_SPEED_SEQ) - 1)]
        elif k == 's':
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
        elif k in _NAV_KEYS:
            action = _NAV_KEYS[k]
            allowed = getattr(env, '_nav_allowed', None)
            if allowed is not None and action in allowed and not allowed[action]:
                return
            now = time.perf_counter()
            if now - getattr(env, '_nav_time', -1.0) >= _NAV_DEBOUNCE:
                env._nav_time = now
                env._nav_request = action

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

    env._play_keys_cid = fig.canvas.mpl_connect('key_press_event', _guarded(_on_key))
    fig.canvas.mpl_connect('motion_notify_event', _guarded(_on_move))
    fig.canvas.mpl_connect('scroll_event', _guarded(_on_scroll))
    fig.canvas.mpl_connect('close_event', lambda ev: setattr(env, '_window_closed', True))


def render_env(env):
    if env.render_mode != 'human':
        return _render_env(env)
    if _window_gone(env):
        raise RenderQuit('render window closed')
    try:
        return _render_env(env)
    except Exception:
        if _window_gone(env):
            raise RenderQuit('render window closed') from None
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


def init_music(env, fig):
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

    raw._vehicle.render().update_artists(raw)
    fig.canvas.draw_idle()
    raw._nav_request = None
    was_paused = getattr(raw, '_paused', False)

    while True:
        plt.pause(0.05)
        if _window_gone(raw):
            return 'quit'
        if getattr(raw, '_paused', False) != was_paused:
            raw._paused = False
            return 'next'
        nav = raw._nav_request
        if nav is not None:
            raw._nav_request = None
            return nav if allow_prev or nav == 'next' else 'replay'
