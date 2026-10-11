#!/usr/bin/env python3

from __future__ import annotations

import sys


from train.train import main as _train_main


def main() -> None:
    forwarded = sys.argv[1:]
    argv = ["--algo", "dummy_agent"]
    if "--timesteps" not in forwarded:
        argv += ["--timesteps", "100"]
    sys.argv = [sys.argv[0]] + argv + forwarded
    _train_main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
