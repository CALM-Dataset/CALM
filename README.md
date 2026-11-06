# CALM

**Comfort-Aware Longitudinal Motion**

A car is approaching a speed bump. How much should it slow down so that the people inside barely feel it, without wasting time?

![The CALM simulator](assets/gym_env.png)

*The half-car simulator on a real bump from Augsburg, with body motion, pitch, speed and acceleration traced on the right.*

## What it is

Every driver makes this choice by feel: brake too little and the bump hits the passengers, brake too much and the trip gets slower for nothing. CALM turns the choice into a planning problem. A planner reads the road ahead and recommends a crossing speed for each bump, and the result is judged by the ISO 2631-1 weighted acceleration that the passengers would feel.

The repository holds a half-car and a quarter-car simulator with nonlinear suspension, both as Gymnasium environments. On top of them sit a PPO policy trained with a curriculum, an MPC reference built on acados, and a rule-based driver. An evaluation tool measures every planner with ISO 2631-1. The real roads come from a dataset of 159 recorded bump crossings in Germany and Italy, built on CARD.

## Get started

You need Python 3.10 or newer and [just](https://github.com/casey/just). Then:

```bash
git clone https://github.com/CALM-Dataset/CALM.git && cd CALM
just install        # creates .venv with CPU PyTorch, the dependencies and the package
just doctor         # checks Python, packages, acados, data and login
```

To use a conda environment instead of `.venv`, set `CALM_PYTHON` and `CALM_PIP` to its `python` and `pip` (a `.env` file works too). acados is only needed for the MPC: `just build-acados` builds the submodule. Docker users can run `docker compose run --rm planner` for a shell with everything but acados installed.

A first run needs no downloaded data:

```bash
just gen-scene                  # procedural training roads
just train --run-name demo      # PPO through the curriculum, half-car by default
just eval demo --scenario c --render
```

The real roads live on Hugging Face and are gated. Request access on the [CALM-Scenarios](https://huggingface.co/datasets/CALM-Dataset/CALM-Scenarios) page, then:

```bash
hf auth login
just fetch-scenarios
```

`just compare` and `just reproduce` run the MPC, so they need acados. They also evaluate the recorded human drive: point `CALM_SEQUENCES` at the `sequences` folders of CALM-Italy and CALM-Germany (separated by `:`). `just compare` accepts `--allow-missing-human` to skip it.

## Documentation

The full documentation is in the [wiki](https://github.com/CALM-Dataset/CALM/wiki). `just --list`, or `calm --help` inside the environment, shows every command.

| Page | What you find there |
| --- | --- |
| **Start here** | |
| [Home](https://github.com/CALM-Dataset/CALM/wiki/Home) | The project at a glance and a map of the wiki |
| [Getting started](https://github.com/CALM-Dataset/CALM/wiki/Getting-Started) | Install, data, and a first training run step by step |
| [Command line](https://github.com/CALM-Dataset/CALM/wiki/Command-Line) | Every `calm` and `just` command with its options |
| [Architecture](https://github.com/CALM-Dataset/CALM/wiki/Architecture) | How the packages fit together |
| **The simulator** | |
| [Simulator and buttons](https://github.com/CALM-Dataset/CALM/wiki/Simulator-and-Buttons) | What the render window shows and the keys that control it |
| [Vehicle models](https://github.com/CALM-Dataset/CALM/wiki/Vehicle-Models) | The half-car and quarter-car equations and parameters |
| [Road and bumps](https://github.com/CALM-Dataset/CALM/wiki/Road-and-Bumps) | How bump profiles and road sections are built |
| [Gymnasium environment](https://github.com/CALM-Dataset/CALM/wiki/Gymnasium-Environment) | The environment loop, actions and episodes |
| **The learning side** | |
| [Observation](https://github.com/CALM-Dataset/CALM/wiki/Observation) | What the policy sees before each bump |
| [Reward](https://github.com/CALM-Dataset/CALM/wiki/Reward) | How comfort and time are traded in the reward |
| [Training the policy](https://github.com/CALM-Dataset/CALM/wiki/Training-the-Policy) | PPO settings, the curriculum and tuning |
| **The planners** | |
| [MPC](https://github.com/CALM-Dataset/CALM/wiki/MPC) | The acados reference planner and its objectives |
| [Reactive driver](https://github.com/CALM-Dataset/CALM/wiki/Reactive-Driver) | The rule-based driver |
| **Data and evaluation** | |
| [Scenarios and data](https://github.com/CALM-Dataset/CALM/wiki/Scenarios-and-Data) | Synthetic and real scenarios, and the Hugging Face datasets |
| [Comfort metric](https://github.com/CALM-Dataset/CALM/wiki/Comfort-Metric) | ISO 2631-1 weighting and how comfort is measured |
| [Evaluation](https://github.com/CALM-Dataset/CALM/wiki/Evaluation) | Comparing PPO, the MPC and the human drive |
| **Help** | |
| [Troubleshooting](https://github.com/CALM-Dataset/CALM/wiki/Troubleshooting) | Common errors and how to fix them |

## Models and data

Everything below is on Hugging Face under CC BY-NC 4.0, with access on request.

| Name | Contents |
| --- | --- |
| [CALM-PPO](https://huggingface.co/CALM-Dataset/CALM-PPO) | The trained PPO policy |
| [CALM-Scenarios](https://huggingface.co/datasets/CALM-Dataset/CALM-Scenarios) | The real road scenarios used for training and evaluation |
| [CALM-Germany](https://huggingface.co/datasets/CALM-Dataset/CALM-Germany) | Recorded bump crossings from Germany |
| [CALM-Italy](https://huggingface.co/datasets/CALM-Dataset/CALM-Italy) | Recorded bump crossings from Italy |

The same ride models in MATLAB and Simulink are in a separate package, [ride-models-matlab](https://github.com/CALM-Dataset/ride-models-matlab).

## Repository layout

```
src/gym_env/   Gymnasium environment, half-car and quarter-car plants, reward, renderer
src/train/     PPO training and the curriculum
src/tune/      Optuna hyperparameter search
src/baseline/  MPC and rule-based driver
src/eval/      evaluation and ISO 2631-1 comfort
src/road/      road profile generation
src/utils/     scenario generation, dataset index and helpers
src/calm/      the calm command line
config/        environment, reward, curriculum, planner and evaluation settings
tests/         fast tests that need no data or GPU
```

The first quarter-car implementation is kept at the git tag `thesis-quarter-car`.

## Citation

```bibtex
@misc{calm_dataset,
  title  = {CALM: Comfort-Aware Longitudinal Motion Planning},
  author = {Mohammed El Said Azab Abdelazim},
  year   = {2026},
  url    = {https://huggingface.co/CALM-Dataset}
}
```

The real roads are built on the CARD recordings, so please also cite:

```bibtex
@inproceedings{elazab2025card,
  title     = {CARD: A Multi-Modal Automotive Dataset for Dense 3D Reconstruction in Challenging Road Topography},
  author    = {Elazab, Gasser and Neuhaus, Frank and Ko{\ss}, Tilman and Splietker, Malte and Date, Aditya and Unterreiner, Michael and Jansen, Maximilian and Hellwich, Olaf},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## License

The code is under the MIT License (see `LICENSE`). The data and models on Hugging Face are under CC BY-NC 4.0.
