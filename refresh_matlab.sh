#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a
: "${MATLAB_FIGURES:?set MATLAB_FIGURES in .env (copy .env.example)}"

PY=${PYTHON:-.venv/bin/python3}
RUNS=${RUNS:-"tq26_s68 tq26_s69 tq26_s70 tq26_s71 tq26_s72"}
REFERENCE=${REFERENCE:-tq26_s69}
MODELS=${MODELS:-logs/models/PPO}
SYNTHETIC_SUITES="01_single_bump 02_two_bumps 03_three_bumps 04_four_bumps 05_one_bump_one_hole 06_two_bumps_one_hole 07_two_bumps_two_holes flat"
export PYTHONPATH=$PWD/src MPLBACKEND=Agg SDL_AUDIODRIVER=dummy

model() { echo "$MODELS/$1/PPO_final.zip"; }
final() { echo "final_s${1##*_s}"; }

real() {
    mkdir -p logs/comfort/runs
    local yaml=logs/comfort/runs_synced.yaml
    echo "runs:" > "$yaml"
    for run in $RUNS; do
        local name; name=$(final "$run")
        local extra=()
        [ "$run" = "$REFERENCE" ] && extra=(--profiles-dir logs/comfort/profiles)
        $PY -m eval.comfort_score --ppo-model "$(model "$run")" --suites real_all --mpc "${extra[@]}" \
            --out "logs/comfort/runs/h_$name.csv"
        echo "  $name: {csv: h_$name, group: seed, seed: ${run##*_s}}" >> "$yaml"
    done
    cp "logs/comfort/runs/h_$(final "$REFERENCE").csv" logs/comfort/comfort_scores.csv
    $PY -m eval.export_runs --runs "$yaml" --runs-dir logs/comfort/runs --reference "$(final "$REFERENCE")" --out-dir logs/comfort
    $PY -m eval.mpc_frontier build --out logs/comfort/mpc_frontier.csv
}

synthetic() {
    rm -rf logs/comfort/synthetic_profiles
    $PY -m eval.comfort_score --ppo-model "$(model "$REFERENCE")" --suites $SYNTHETIC_SUITES --mpc \
        --out logs/comfort/synthetic_scores.csv --profiles-dir logs/comfort/synthetic_profiles
}

case "${1:-all}" in
    real) real; ./sync_comfort.sh real ;;
    synthetic) synthetic; ./sync_comfort.sh synthetic ;;
    all) real; synthetic; ./sync_comfort.sh all ;;
    sync) ./sync_comfort.sh all ;;
    *) echo "usage: $0 [all|real|synthetic|sync]"; exit 2 ;;
esac
