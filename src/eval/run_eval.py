#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from utils.paths import ROOT as _ROOT
_PY         = sys.executable
_MODEL_ROOT = _ROOT / "logs" / "models" / "PPO"

_ALL = ["PPO", "MPC", "Reactive", "TrainCurve"]

silent = True


def _norm_exp(tag: str) -> str:
    return f"exp_{tag}" if tag.isdigit() else tag


def _model_path(exp: str) -> Path:
    p = _MODEL_ROOT / exp / "PPO_final.zip"
    if not p.is_file():
        raise FileNotFoundError(f"model not found: {p}")
    if not (p.parent / "vecnormalize.pkl").is_file():
        raise FileNotFoundError(f"vecnormalize.pkl missing in {p.parent}")
    return p


def _run(label: str, cmd: list[str]) -> bool:
    print(f"\n{'='*70}\n  {label}\n  $ {' '.join(str(c) for c in cmd)}\n{'='*70}", flush=True)
    if silent:
        res = subprocess.run(cmd, cwd=str(_ROOT), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
        ok = res.returncode == 0
        if not ok:
            tail = "".join((res.stdout or "").splitlines(keepends=True)[-25:])
            print(tail, flush=True)
    else:
        res = subprocess.run(cmd, cwd=str(_ROOT))
        ok = res.returncode == 0
    print(f"  -> {label}: {'OK' if ok else f'FAILED (exit {res.returncode})'}", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="PPO experiment tag, e.g. exp_6 or 6")
    ap.add_argument("--data-dir", default="data", help="output root (default: data)")
    ap.add_argument("--scenario", default=None,
                    help="run a single scenario instead of all enabled suites")
    ap.add_argument("--only", nargs="+", choices=_ALL, default=None,
                    help="run only these methods")
    ap.add_argument("--skip", nargs="+", choices=_ALL, default=None,
                    help="skip these methods")
    ap.add_argument("--formats", default="mat",
                    help="output file formats, comma list: mat,npz,json (default: mat)")
    args = ap.parse_args()

    os.environ["EVAL_SAVE_FORMATS"] = args.formats

    exp     = _norm_exp(args.exp)
    methods = args.only if args.only else list(_ALL)
    if args.skip:
        methods = [m for m in methods if m not in args.skip]

    scen = ["--scenario", args.scenario] if args.scenario else []
    log  = ["--log-data", args.data_dir]

    model = None
    if "PPO" in methods:
        model = _model_path(exp)

    plan: list[tuple[str, list[str]]] = []
    for m in methods:
        if m == "PPO":
            plan.append(("PPO", [_PY, "-m", "eval.eval", "--algo", "PPO",
                                 "--model_path", str(model), *scen, *log]))
        elif m == "MPC":
            plan.append(("MPC", [_PY, "-m", "baseline.mpc.mpc", *scen, *log]))
        elif m == "Reactive":
            plan.append(("Reactive", [_PY, "-m", "baseline.reactive_driver.driver_eval", *scen, *log]))
        elif m == "TrainCurve":
            plan.append(("TrainCurve", [_PY, "-m", "utils.export_training_curve", exp]))

    print(f"\nEXP = {exp}  methods = {[p[0] for p in plan]}")

    results = {label: _run(label, cmd) for label, cmd in plan}

    print(f"\n{'='*70}\n  SUMMARY  (exp={exp})\n{'='*70}")
    for label, ok in results.items():
        print(f"  {'[OK]' if ok else '[FAIL]':7s} {label}")
    print(f"  data → {_ROOT / args.data_dir}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
