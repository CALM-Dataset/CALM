set shell := ["bash", "-c"]
set positional-arguments := true

venv := ".venv/bin/python3"
pip  := ".venv/bin/pip3"

export PYTHONPATH := "src"
export ACADOS_SOURCE_DIR := justfile_directory() + "/acados"
export LD_LIBRARY_PATH := justfile_directory() + "/acados/lib:" + env("LD_LIBRARY_PATH", "")

default:
    @just --list

# create .venv, then install CPU PyTorch, the dependencies and the package
[group('setup')]
install:
    [ -x .venv/bin/python3 ] || python3 -m venv .venv
    {{pip}} install torch --index-url https://download.pytorch.org/whl/cpu
    {{pip}} install -r requirements.txt
    {{pip}} install -e ".[dev]"

# static check, as in CI
[group('setup')]
lint:
    {{venv}} -m pyflakes src

# build the acados submodule (only the MPC needs it)
[group('setup')]
build-acados:
    git submodule update --init --recursive acados
    mkdir -p acados/build
    cd acados/build && cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_INSTALL_PREFIX=$(pwd)/.. .. && make -j$(nproc) && make install
    {{pip}} install vcs-versioning setuptools_scm
    {{pip}} install -e acados/interfaces/acados_template

# download the real scenarios from Hugging Face (gated: needs access and `hf auth login`)
[group('data')]
fetch-scenarios:
    {{venv}} -c "from utils.dataset import ensure_local; ensure_local()"

# metadata table over every scenario (config/dataset/index.json)
[group('data')]
build-index:
    {{venv}} -m utils.dataset

# procedural training scenarios
[group('data')]
gen-scene *args:
    {{venv}} -m utils.generate_train_scenarios "$@"

# PPO training through the curriculum; logs/models/PPO/<run-name>/
[group('rl')]
train *args:
    {{venv}} -m train.train --algo PPO "$@"

# evaluate logs/models/<algo>/<run>/<algo>_final.zip, e.g. just eval PPO exp_3
[group('rl')]
eval algo run *args:
    shift 2; {{venv}} -m eval.eval --algo {{algo}} --model_path logs/models/{{algo}}/{{run}}/{{algo}}_final.zip "$@"

# Optuna hyperparameter search
[group('rl')]
tune *args:
    {{venv}} -m tune.tune --storage sqlite:///tune.db --study-name ppo_study "$@"

# TensorBoard on logs/tensorboard
[group('rl')]
tb:
    {{venv}} -m tensorboard.main --logdir logs/tensorboard

# ISO 2631 comfort of PPO, MPC and the recorded drive; set CALM_SEQUENCES for the human
[group('comfort')]
compare run ckpt="PPO_final.zip" *args:
    @case "{{ckpt}}" in -*) echo "usage: just compare <run> [checkpoint.zip] [options]" >&2; exit 2;; esac
    shift 2; {{venv}} -m eval.comfort_score --ppo-model logs/models/PPO/{{run}}/{{ckpt}} --mpc --profiles-dir logs/comfort/profiles "$@"

# MPC baseline (needs acados); pass --scenario <name> or --render
[group('baselines')]
mpc *args:
    {{venv}} -m baseline.mpc.mpc "$@"

# rule-based driver baseline; pass --scenario <name> or --render
[group('baselines')]
reactive-driver *args:
    {{venv}} -m baseline.reactive_driver.driver_eval "$@"

# zero-action baseline run through the training pipeline (smoke test)
[group('baselines')]
dummy *args:
    {{venv}} -m baseline.dummy_agent.train "$@"
