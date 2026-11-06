set dotenv-load := true
set shell := ["bash", "-c"]
set positional-arguments := true
set quiet := true

venv := env("CALM_PYTHON", ".venv/bin/python3")
pip  := env("CALM_PIP", ".venv/bin/pip3")
calm := venv + " -m calm"

export PYTHONPATH := "src"
export ACADOS_SOURCE_DIR := justfile_directory() + "/acados"
export LD_LIBRARY_PATH := justfile_directory() + "/acados/lib:" + env("LD_LIBRARY_PATH", "")

default:
    @just --list

# check Python, packages, acados, data and login, with hints
[group('setup')]
doctor:
    {{calm}} doctor

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
    {{venv}} -m pyflakes src tests

# fast tests, as in CI (no data or GPU needed)
[group('setup')]
test *args:
    MPLBACKEND=Agg SDL_AUDIODRIVER=dummy {{venv}} -m pytest -q "$@"

# build the acados submodule (only the MPC needs it)
[group('setup')]
build-acados:
    git submodule update --init --recursive acados
    mkdir -p acados/build
    cd acados/build && cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_INSTALL_PREFIX=$(pwd)/.. .. && make -j$(python3 -c "import os; print(os.cpu_count())") && make install
    {{pip}} install vcs-versioning setuptools_scm
    {{pip}} install -e acados/interfaces/acados_template

# download the real scenarios from Hugging Face (gated: needs access and `hf auth login`)
[group('data')]
fetch-scenarios:
    {{calm}} fetch

# procedural training scenarios
[group('data')]
gen-scene *args:
    {{calm}} gen-scenes "$@"

# PPO training through the curriculum; logs/models/PPO/<run-name>/
[group('RL')]
train *args:
    {{calm}} train "$@"

# evaluate logs/models/PPO/<run>/PPO_final.zip, e.g. just eval exp_3
[group('RL')]
eval *args:
    {{calm}} eval "$@"

# Optuna hyperparameter search
[group('RL')]
tune *args:
    {{calm}} tune --storage sqlite:///tune.db --study-name ppo_study "$@"

# TensorBoard on logs/tensorboard
[group('RL')]
tb:
    {{calm}} tb

# ISO 2631 comfort of PPO, MPC and the recorded drive; set CALM_SEQUENCES for the human
[group('comfort')]
compare *args:
    {{calm}} compare "$@"

# train several seeds and score each one against the MPC and the human drive
[group('comfort')]
reproduce *args:
    {{calm}} reproduce "$@"

# MPC baseline (needs acados); pass --scenario <name> or --render
[group('baselines')]
mpc *args:
    {{calm}} mpc "$@"

# rule-based driver baseline; pass --scenario <name> or --render
[group('baselines')]
reactive-driver *args:
    {{calm}} reactive "$@"

