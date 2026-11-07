# Getting started

Install CALM, download the data, run a first training and evaluation. You need Python 3.10 or newer and [just](https://github.com/casey/just). No GPU is needed.

## Install

```bash
git clone --recurse-submodules https://github.com/CALM-Dataset/CALM
cd CALM
just install
.venv/bin/calm doctor
```

1. `just install` creates `.venv`, installs CPU PyTorch, `requirements.txt` and the package that provides the `calm` command.
2. `calm doctor` prints `ok`, `warn` or `FAIL` per item with a hint. Only `FAIL` stops you.
3. Only for the MPC: `just build-acados` builds the acados submodule with qpOASES into `acados/` and installs `acados_template`.

<details>
<summary>Linux</summary>

Without `DISPLAY` or `WAYLAND_DISPLAY` the renderer uses the headless Agg backend. Use `--save-gif` or `--save-plots` instead of `--render`.
</details>

<details>
<summary>macOS</summary>

The simulator window opens full screen. The MPC needs a compiler and cmake.

```bash
xcode-select --install
brew install cmake
```
</details>

<details>
<summary>Windows</summary>

Run `just` from Git Bash. The command is `.venv/Scripts/calm`. Training runs in one process because Windows has no `fork`. The MPC needs cmake and the Visual Studio C++ build tools; acados is built with the generic BLAS target.
</details>

<details>
<summary>Docker</summary>

The image has everything except acados and renders headless.

```bash
mkdir -p logs
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose build
docker compose run --rm planner
```
</details>

<details>
<summary>conda or another environment</summary>

Set the interpreter in a `.env` file in the repository root:

```bash
CALM_PYTHON=/opt/anaconda3/envs/calm/bin/python
CALM_PIP=/opt/anaconda3/envs/calm/bin/pip
```
</details>

> [!NOTE]
> The first MPC run downloads the acados template renderer from GitHub, so it needs network once.

## Data

The real crossings are in the gated [CALM-Scenarios](https://huggingface.co/datasets/CALM-Dataset/CALM-Scenarios). Request access on its page, then:

```bash
hf auth login
calm fetch
calm gen-scenes
```

- `calm fetch` skips crossings marked `low_confidence` and the one under `exclude` in `split.yaml`.
- It fills `config/train/scenarios/real/` (90 crossings), `config/eval/scenarios/real_train/` (a copy) and `config/eval/scenarios/real/` (14 held-out crossings).
- `calm gen-scenes` writes the generated training roads, 200 per difficulty level.

## First run

```bash
calm train --run-name demo --timesteps 2000
calm eval demo --scenario c --render
```

- `--timesteps`: decisions, one per bump. Without the real crossings, training uses only generated roads.
- `--scenario c`: the raised crosswalk, 5 episodes by default.
- `--render`: opens the [Simulator](Simulator) window.

To run a policy from [CALM-PPO](https://huggingface.co/CALM-Dataset/CALM-PPO), copy `PPO_final.zip` and `vecnormalize.pkl` into `logs/models/PPO/calm_ppo/` and run `calm eval calm_ppo --scenario c --render`. For the pickle-free files copy `policy.safetensors` and `vecnormalize.json` and run `calm eval calm_ppo policy.safetensors --scenario c --render`.

## Outputs

- `calm train`: `logs/models/PPO/<run>/` (`PPO_final.zip`, `vecnormalize.pkl`, `summary.json`, `best/`), `logs/tensorboard/<run>/`, `logs/monitor/<run>/`.
- `calm eval`: `logs/eval/results/eval_PPO_<scenario>_run/` or `eval_PPO_suite_run/`.
- `calm mpc`, `calm reactive`: `logs/eval/results/mpc/<scenario>/`, `logs/eval/results/reactive_driver/<scenario>/`.
- `calm compare`: `logs/comfort/comfort_scores.csv`.

Without `--run-name`, runs are numbered `exp_1`, `exp_2` and so on.

See also: [Command line](Command-Line), [Troubleshooting](Troubleshooting), [PPO training](PPO-Training)
