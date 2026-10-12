#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
: "${MATLAB_FIGURES:?set MATLAB_FIGURES in .env (copy .env.example)}"

for f in comfort_scores mpc_frontier ppo_runs ppo_runs_scenarios scenarios; do
    [ -f logs/comfort/$f.csv ] && cp logs/comfort/$f.csv "$MATLAB_FIGURES/$f.csv" && echo "copied logs/comfort/$f.csv -> $MATLAB_FIGURES"
done
if [ -d logs/comfort/profiles ]; then
    mkdir -p "$MATLAB_FIGURES/comfort_profiles"
    cp logs/comfort/profiles/*.csv "$MATLAB_FIGURES/comfort_profiles/"
    echo "copied $(ls logs/comfort/profiles | wc -l) profile files -> $MATLAB_FIGURES/comfort_profiles"
fi
