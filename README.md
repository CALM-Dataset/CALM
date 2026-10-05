# Comfort-Aware Speed Planning over Speed Bumps

Recommended-speed baselines for passenger comfort over speed bumps. A PPO agent
and a model-predictive controller each read the road ahead and recommend a speed
for crossing every bump; the result is scored with ISO 2631-1 against the speed the
human driver actually drove. The two planners are independent of each other.

![Vehicle environment](assets/gym_env.png)

## Overview

The planner is trained with PPO on a nonlinear vehicle simulation wrapped in a
Gymnasium environment. Two plant models are available:

- **Half-car** (default): 4 degrees of freedom, heave and pitch, with separate
  front and rear axles. The rear axle rides the road one wheelbase behind the
  front.
- **Quarter-car**: 2 degrees of freedom, single-axle heave only.

The vehicle dynamics use a nonlinear suspension (progressive spring with
bumpstops, asymmetric compression/rebound damping) integrated with RK4. The
reward puts ride comfort first: it penalises body vertical and pitch
acceleration, with secondary terms for longitudinal jerk and throttle
smoothness. The policy outputs a recommended speed between an 8 km/h floor and the scenario
speed, which a follower tracks, so it learns how much to slow for each obstacle
from the comfort terms and a controlled average speed, without using the MPC.

Training uses a four-level performance-gated curriculum over randomised roads.
A model-based MPC planner and rule-based drivers are included as baselines.

## Installation

```bash
just install          # create .venv, install dependencies and the package (pip install -e .)
```

Python dependencies are listed in `requirements.txt` (Gymnasium,
Stable-Baselines3, PyTorch, NumPy, SciPy, Numba, Matplotlib, TensorBoard).

### acados (MPC baseline only)

The MPC baseline uses [acados](https://github.com/acados/acados) for the optimal
control problem. It is a separate C library and is not bundled here. To build
and install it:

```bash
git clone https://github.com/acados/acados.git
cd acados && git submodule update --init --recursive
mkdir build && cd build && cmake -DACADOS_WITH_QPOASES=ON .. && make install -j4
pip install -e ../interfaces/acados_template
export LD_LIBRARY_PATH=$PWD/../lib:$LD_LIBRARY_PATH
```

The RL agent, the rule-based driver and evaluation do not require acados.

## Docker

Run the whole project in a container without installing anything locally.
The container runs as a non-root user matching your host UID/GID, so files it
writes to the mounted `logs/` and `data/` folders stay owned by you:

```bash
mkdir -p logs data
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose build
docker compose run --rm planner            # opens a shell with everything installed
```

Inside the container:

```bash
just train
just eval PPO <run-name>
just mpc                                    # requires acados (see below)
```

Training scenarios are generated during the image build. Logs and evaluation
data are written to `./logs` and `./data` on the host through mounted volumes.
Rendering is headless in the container (`MPLBACKEND=Agg`); use `--save-plots`
or `--save-gif` to capture output. The acados MPC baseline needs the acados
library, which is not included in the image.

## Scenarios

The measured (real) training roads and the evaluation suites extend the
[CARD dataset](https://huggingface.co/CARD-Data) with comfort labels, hosted on
HuggingFace as [CALM-Dataset/CALM-Germany](https://huggingface.co/datasets/CALM-Dataset/CALM-Germany)
(CC BY 4.0) and [CALM-Dataset/CALM-Italy](https://huggingface.co/datasets/CALM-Dataset/CALM-Italy)
(CC BY-NC 4.0), not committed to the repository.

Both are gated: request access on each dataset page (agreeing to its license
and usage terms), then authenticate locally before downloading:

```bash
hf auth login                   # once, with a token from huggingface.co/settings/tokens
just fetch-scenarios            # writes config/train/scenarios/real and config/eval/scenarios
```

The procedural training scenarios are generated locally, not stored anywhere:

```bash
just gen-scene                  # writes config/train/scenarios/{curriculum,normal}
```

`just build-index` writes `config/dataset/index.json`, a metadata table over
every scenario (source, size, bump count, recommended speeds).

## Usage

```bash
just train                      # PPO through the 4-level curriculum (half-car)
just train --vehicle quarter    # quarter-car plant

just eval PPO <run-name> --render     # evaluate a trained model
just mpc --render                           # MPC baseline
just reactive-driver --render               # rule-based driver baseline
just dummy                                  # zero-action baseline

just tb                         # TensorBoard
just tune --trials 50           # Optuna hyperparameter search
```

The vehicle at evaluation time must match the one the model was trained on.

### Render controls

- `space` pause, `f` / `d` faster / slower playback (down to 0.1x)
- `+` / `-` or scroll wheel zoom, `0` reset zoom
- Left / right arrow: previous / next scenario

## Repository layout

```
src/gym_env/half_car/     half-car Gymnasium environment (4-DOF)
src/gym_env/quarter_car/  quarter-car Gymnasium environment (2-DOF)
src/train/        PPO training, curriculum, callbacks
src/eval/         evaluation and plotting
src/baseline/     MPC and rule-based baselines
src/road/         road profile generation
src/utils/        scenario generation and helpers
config/           reward, curriculum, preview and scenario configuration
```

## Vehicle dynamics

The quarter-car body/wheel model:

```
m_B z̈_B = -F_spring(z_W - z_B) - F_damp(ż_W - ż_B)
m_W z̈_W =  F_spring + F_damp - F_tyre(z_W - ζ)
```

The half-car adds a pitch degree of freedom about the centre of mass and a
second axle, with load transfer coupling the front and rear suspension.

## License

See `LICENSE`.
