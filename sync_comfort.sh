#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
: "${MATLAB_FIGURES:?set MATLAB_FIGURES in .env (copy .env.example)}"

what=${1:-all}
files=(); dirs=()
case "$what" in
    real) files=(comfort_scores mpc_frontier ppo_runs ppo_runs_scenarios scenarios); dirs=(comfort_profiles:profiles) ;;
    synthetic) files=(synthetic_scores); dirs=(synthetic_profiles:synthetic_profiles) ;;
    all) files=(comfort_scores mpc_frontier ppo_runs ppo_runs_scenarios scenarios synthetic_scores)
         dirs=(comfort_profiles:profiles synthetic_profiles:synthetic_profiles) ;;
    *) echo "usage: $0 [all|real|synthetic]"; exit 2 ;;
esac

for f in "${files[@]}"; do
    [ -f logs/comfort/$f.csv ] && cp logs/comfort/$f.csv "$MATLAB_FIGURES/$f.csv" && echo "copied logs/comfort/$f.csv -> $MATLAB_FIGURES"
done
for d in "${dirs[@]}"; do
    dest=${d%%:*}; src=logs/comfort/${d##*:}
    if [ -d "$src" ]; then
        rm -rf "${MATLAB_FIGURES:?}/$dest"
        mkdir -p "$MATLAB_FIGURES/$dest"
        cp "$src"/*.csv "$MATLAB_FIGURES/$dest/"
        echo "copied $(ls "$src" | wc -l) profile files -> $MATLAB_FIGURES/$dest"
    fi
done
