import subprocess
import sys

from utils.paths import ROOT


def calm(*args):
    return subprocess.run([sys.executable, "-m", "calm", *args], cwd=ROOT, capture_output=True, text=True)


def test_help_lists_the_commands():
    out = calm("--help")
    assert out.returncode == 0
    for command in ("train", "eval", "mpc", "doctor"):
        assert command in out.stdout


def test_an_unknown_command_fails():
    assert calm("no-such-command").returncode != 0
