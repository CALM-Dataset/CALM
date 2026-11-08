# CALM: Comfort-Aware Longitudinal Motion Planning

![Code: MIT](https://img.shields.io/badge/code-MIT-blue)
![Data: CC BY-NC 4.0](https://img.shields.io/badge/data-CC%20BY--NC%204.0-lightgrey)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-CALM--Dataset-ffd21e)](https://huggingface.co/CALM-Dataset)

CALM recommends how much a car should slow down for a speed bump so that the people inside barely feel it, without wasting time. It is a half-car and quarter-car simulator with nonlinear suspension, wrapped as Gymnasium environments, with a PPO policy, an acados MPC and a rule-based driver on top. Comfort is measured with the ISO 2631-1 weighted acceleration, on 159 recorded bump crossings from Germany and Italy built on CARD.

![The CALM simulator](assets/gym_env.png)

*The half-car simulator on a real bump from Augsburg, with body motion, pitch, speed and acceleration traced on the right.*

## Installation

You need Python 3.10 or newer and [just](https://github.com/casey/just).

```bash
git clone https://github.com/CALM-Dataset/CALM.git && cd CALM
just install        # .venv with CPU PyTorch, the dependencies and the package
just doctor         # checks Python, packages, acados, data and login
```

- Conda instead of `.venv`: set `CALM_PYTHON` and `CALM_PIP` to its `python` and `pip` (a `.env` file works too).
- MPC only: `git submodule update --init --recursive` then `just build-acados` (Linux, macOS, Windows; needs cmake and a C compiler).
- Docker: `docker compose -f docker/docker-compose.yml run --rm planner` opens a shell with everything but acados installed.
- Windows: run `just` from Git Bash.

## Getting started

```bash
just gen-scene                        # procedural training roads
just train --run-name demo            # PPO through the curriculum, half-car
just eval demo --scenario c --render  # watch it cross a raised crosswalk
```

Real roads and the trained policy:

```bash
hf auth login                         # once, after requesting access to CALM-Scenarios
just fetch-scenarios                  # real roads and the perception model
just eval calm_ppo --scenario c       # after copying the CALM-PPO files to logs/models/PPO/calm_ppo/
```

Comparing PPO, MPC and the recorded human drive:

```bash
just compare demo --allow-missing-human
```

Point `CALM_SEQUENCES` at the `sequences` folders of CALM-Italy and CALM-Germany (separated by `:`) to include the human drive. `just --list` shows every recipe.

## Documentation

Reference pages live in the [wiki](https://github.com/CALM-Dataset/CALM/wiki):

- Start: [Getting started](https://github.com/CALM-Dataset/CALM/wiki/Getting-Started), [Command line](https://github.com/CALM-Dataset/CALM/wiki/Command-Line), [Simulator](https://github.com/CALM-Dataset/CALM/wiki/Simulator)
- Environment: [Gymnasium environment](https://github.com/CALM-Dataset/CALM/wiki/Gymnasium-Environment) with [action space](https://github.com/CALM-Dataset/CALM/wiki/Action-Space), [observation space](https://github.com/CALM-Dataset/CALM/wiki/Observation-Space), [reward](https://github.com/CALM-Dataset/CALM/wiki/Reward-Function), [episodes](https://github.com/CALM-Dataset/CALM/wiki/Episodes-and-Termination) and [roads](https://github.com/CALM-Dataset/CALM/wiki/Roads-and-Bumps)
- Vehicles: [Vehicle models](https://github.com/CALM-Dataset/CALM/wiki/Vehicle-Models), [half-car](https://github.com/CALM-Dataset/CALM/wiki/Half-Car-Model), [quarter-car](https://github.com/CALM-Dataset/CALM/wiki/Quarter-Car-Model)
- Policy and planners: [PPO training](https://github.com/CALM-Dataset/CALM/wiki/PPO-Training), [curriculum](https://github.com/CALM-Dataset/CALM/wiki/Curriculum), [MPC](https://github.com/CALM-Dataset/CALM/wiki/MPC-Planner), [reactive driver](https://github.com/CALM-Dataset/CALM/wiki/Reactive-Driver)
- Data and evaluation: [Scenarios](https://github.com/CALM-Dataset/CALM/wiki/Scenarios), [dataset format](https://github.com/CALM-Dataset/CALM/wiki/Dataset-Format), [Evaluation](https://github.com/CALM-Dataset/CALM/wiki/Evaluation), [comfort metric](https://github.com/CALM-Dataset/CALM/wiki/Comfort-Metric)
- Help: [Troubleshooting](https://github.com/CALM-Dataset/CALM/wiki/Troubleshooting)

## Models and data

Everything below is on Hugging Face under CC BY-NC 4.0, with access on request.

| Name | Contents |
| --- | --- |
| [CALM-PPO](https://huggingface.co/CALM-Dataset/CALM-PPO) | The trained PPO policy, as Stable-Baselines3 files and `policy.safetensors` |
| [CALM-Scenarios](https://huggingface.co/datasets/CALM-Dataset/CALM-Scenarios) | The real road scenarios and the train and evaluation split |
| [CALM-Germany](https://huggingface.co/datasets/CALM-Dataset/CALM-Germany) | Recorded bump crossings from Germany |
| [CALM-Italy](https://huggingface.co/datasets/CALM-Dataset/CALM-Italy) | Recorded bump crossings from Italy |

The same ride models in MATLAB and Simulink are in [ride-models-matlab](https://github.com/CALM-Dataset/ride-models-matlab).

## Repository layout

```
src/gym_env/   environments, half-car and quarter-car plants, reward, renderer
src/train/     PPO training and the curriculum
src/tune/      Optuna hyperparameter search
src/baseline/  MPC and rule-based driver
src/eval/      evaluation and ISO 2631-1 comfort
src/road/      road profile generation
src/utils/     scenario generation, dataset download, helpers
src/calm/      the calm command line
config/        environment, reward, curriculum, planner and evaluation settings
tests/         fast tests, no data or GPU needed
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

## Acknowledgment

- [CARD](https://huggingface.co/CARD-Data) for the recordings the real roads are built on
- [acados](https://github.com/acados/acados) for the MPC solver
- [Gymnasium](https://gymnasium.farama.org) and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the environment interface and PPO

## License

The code is under the MIT License (see `LICENSE`). The data and models on Hugging Face are under CC BY-NC 4.0.
