import os

import numpy as np

_SR = 44100

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          *[os.pardir] * 4))
_MUSIC_DIR = os.environ.get('QC_MUSIC_DIR',
                            os.path.join(_REPO_ROOT, 'assets', 'music'))
_EXTS = ('.ogg', '.wav', '.mp3')

_A4 = 440.0
_SEMITONES = {
    'C': -9, 'C#': -8, 'D': -7, 'D#': -6, 'E': -5, 'F': -4,
    'F#': -3, 'G': -2, 'G#': -1, 'A': 0, 'A#': 1, 'B': 2,
}


def _freq(note: str) -> float:
    if note == 'R':
        return 0.0
    name, octave = note[:-1], int(note[-1])
    semis = _SEMITONES[name] + 12 * (octave - 4)
    return _A4 * 2.0 ** (semis / 12.0)


SONGS = {
    'cruise': (120, [
        ('C4', 1), ('E4', 1), ('G4', 1), ('C5', 1),
        ('G4', 0.5), ('E4', 0.5), ('C4', 1), ('R', 1),
    ]),
    'drive': (128, [
        ('A4', 0.5), ('A4', 0.5), ('C5', 1), ('B4', 0.5), ('A4', 0.5),
        ('G4', 1), ('E4', 1), ('R', 0.5),
    ]),
    'open_road': (110, [
        ('D4', 1), ('F#4', 1), ('A4', 1), ('D5', 0.5), ('A4', 0.5),
        ('F#4', 1), ('D4', 1),
    ]),
}

DEFAULT_TRACK = 'cruise'


def _find_file(track: str):
    for ext in _EXTS:
        path = os.path.join(_MUSIC_DIR, track + ext)
        if os.path.isfile(path):
            return path
    return None


def _synth(song) -> "np.ndarray":
    bpm, notes = song
    beat = 60.0 / bpm
    chunks = []
    for note, beats in notes:
        n = int(_SR * beat * beats)
        t = np.arange(n) / _SR
        f = _freq(note)
        if f == 0.0:
            chunks.append(np.zeros(n))
            continue
        wave = (np.sin(2 * np.pi * f * t)
                + 0.15 * np.sin(2 * np.pi * 2 * f * t))
        env = np.ones(n)
        attack = max(int(0.01 * _SR), 1)
        decay = max(int(0.06 * _SR), 1)
        env[:attack] = np.linspace(0, 1, attack)
        env[attack:attack + decay] = np.linspace(1, 0.7, min(decay, n - attack))
        release = min(int(0.03 * _SR), n)
        env[-release:] *= np.linspace(1, 0, release)
        chunks.append(wave * env)
    mono = np.concatenate(chunks) if chunks else np.zeros(1)
    mono /= max(np.max(np.abs(mono)), 1e-9)
    mono = (mono * 0.4 * 32767).astype(np.int16)
    return np.column_stack([mono, mono])


class Jukebox:

    def __init__(self, track: str = DEFAULT_TRACK,
                 muted: bool = True, volume: float = 0.6,
                 jukebox: bool = False):
        self.track = track if track in SONGS else DEFAULT_TRACK
        self.muted = bool(muted)
        self.volume = float(np.clip(volume, 0.0, 1.0))
        self.jukebox = bool(jukebox)
        self.ok = False
        self._sound = None
        self._channel = None
        self._mode = None

        if os.environ.get('QC_NO_AUDIO'):
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=_SR, size=-16, channels=2)
            self._pygame = pygame
            self.ok = True
        except Exception:
            self._pygame = None
            self.ok = False

    def _load(self):
        if not self.ok:
            return
        path = _find_file(self.track)
        if path:
            self._pygame.mixer.music.load(path)
            self._mode = 'file'
        else:
            buf = _synth(SONGS[self.track])
            self._sound = self._pygame.sndarray.make_sound(np.ascontiguousarray(buf))
            self._mode = 'synth'
        self._apply_volume()

    def _apply_volume(self):
        vol = 0.0 if self.muted else self.volume
        if self._mode == 'file':
            self._pygame.mixer.music.set_volume(vol)
        elif self._sound is not None:
            self._sound.set_volume(vol)

    def start(self):
        if not self.ok:
            return
        if self._mode is None:
            self._load()
        loops = 0 if self.jukebox else -1
        if self._mode == 'file':
            self._pygame.mixer.music.play(loops=loops)
        else:
            self._channel = self._sound.play(loops=loops)
        self._apply_volume()

    def tick(self):
        if not self.ok or not self.jukebox:
            return
        if not self._busy():
            self.next_track()

    def _busy(self) -> bool:
        if self._mode == 'file':
            return bool(self._pygame.mixer.music.get_busy())
        return self._channel is not None and bool(self._channel.get_busy())

    def stop(self):
        if not self.ok:
            return
        if self._mode == 'file':
            self._pygame.mixer.music.stop()
        elif self._sound is not None:
            self._sound.stop()

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        self._apply_volume()
        return self.muted

    def set_volume(self, vol: float):
        self.volume = float(np.clip(vol, 0.0, 1.0))
        if self.volume > 0:
            self.muted = False
        self._apply_volume()

    def set_track(self, track: str):
        if track not in SONGS:
            return
        playing = self._mode is not None
        self.stop()
        self.track = track
        self._sound = None
        self._mode = None
        if playing:
            self.start()

    def next_track(self) -> str:
        keys = list(SONGS)
        i = (keys.index(self.track) + 1) % len(keys)
        self.set_track(keys[i])
        return self.track

    @property
    def icon(self) -> str:
        return '⊘' if self.muted else '♫'
