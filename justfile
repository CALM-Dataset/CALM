set shell := ["bash", "-c"]
set positional-arguments := true

venv := ".venv/bin/python3"
pip  := ".venv/bin/pip3"

export PYTHONPATH := "src"
export ACADOS_SOURCE_DIR := justfile_directory() + "/acados"
export LD_LIBRARY_PATH := justfile_directory() + "/acados/lib:" + env("LD_LIBRARY_PATH", "")

default:
    @just --list

[group('setup')]
install:
    {{pip}} install -r requirements.txt
    {{pip}} install -e .

[group('setup')]
install-torch-cpu:
    {{pip}} install torch --index-url https://download.pytorch.org/whl/cpu

# build the acados submodule (only the MPC needs it)
[group('setup')]
build-acados:
    git submodule update --init --recursive acados
    mkdir -p acados/build
    cd acados/build && cmake -DACADOS_WITH_QPOASES=ON -DCMAKE_INSTALL_PREFIX=$(pwd)/.. .. && make -j$(nproc) && make install
    {{pip}} install -e acados/interfaces/acados_template

# download the real scenarios from Hugging Face (CALM-Germany CC BY 4.0, CALM-Italy CC BY-NC 4.0)
[group('data')]
fetch-scenarios:
    {{venv}} -c "from utils.dataset import ensure_local; ensure_local()"

[group('data')]
build-index:
    {{venv}} -m utils.dataset

# procedural training scenarios
[group('data')]
gen-scene *args:
    {{venv}} -m utils.generate_train_scenarios "$@"

[group('rl')]
train *args:
    {{venv}} -m train.train --algo PPO "$@"

# evaluate logs/models/<algo>/<run>/<algo>_final.zip, e.g. just eval PPO exp_3
[group('rl')]
eval algo run *args:
    shift 2; {{venv}} -m eval.eval --algo {{algo}} --model_path logs/models/{{algo}}/{{run}}/{{algo}}_final.zip "$@"

[group('rl')]
tune *args:
    {{venv}} -m tune.tune --storage sqlite:///tune.db --study-name ppo_study "$@"

[group('rl')]
tb:
    {{venv}} -m tensorboard.main --logdir logs/tensorboard

# ISO 2631 comfort of PPO, MPC and the recorded drive; set CALM_SEQUENCES for the human
[group('comfort')]
compare run ckpt="PPO_final.zip" *args:
    shift 2; {{venv}} -m eval.comfort_score --ppo-model logs/models/PPO/{{run}}/{{ckpt}} --mpc --profiles-dir logs/comfort/profiles "$@"

[group('baselines')]
mpc *args:
    {{venv}} -m baseline.mpc.mpc "$@"

[group('baselines')]
reactive-driver *args:
    {{venv}} -m baseline.reactive_driver.driver_eval "$@"

[group('baselines')]
dummy *args:
    {{venv}} -m baseline.dummy_agent.train "$@"
