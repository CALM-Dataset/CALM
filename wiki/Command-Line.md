# Command line

Every `calm` command and `just` recipe. `calm <command> --help` lists all options. Options after a command name go to that command; `just` recipes pass extra arguments through (`just eval demo --scenario c --render`).

| Command | What it does | Main options |
|---|---|---|
| `calm doctor`, `just doctor` | checks Python, packages, acados, data and login, with hints | `--data-only` |
| `calm fetch`, `just fetch-scenarios` | downloads the gated real crossings, then checks the data | needs `hf auth login` |
| `calm gen-scenes`, `just gen-scene` | writes the generated training roads | `--out-dir`, `--seed`, `--mode bumps\|potholes\|both` |
| `calm train`, `just train` | PPO training through the curriculum | `--run-name`, `--timesteps`, `--n-envs`, `--seed` (69), `--vehicle half\|quarter`, `--resume <zip>`, `--start-level`, `-m`, `--sparse-reward` |
| `calm tune`, `just tune` | Optuna search over the PPO settings; the recipe adds `--storage sqlite:///tune.db --study-name ppo_study` | `--trials`, `--timesteps`, `--n-jobs`, `--study-name`, `--storage`, `--no-curriculum`, `--config` |
| `calm tb`, `just tb` | TensorBoard on `logs/tensorboard` | `--port` |
| `calm eval <run> [checkpoint]`, `just eval` | evaluates `logs/models/PPO/<run>/`, default checkpoint `PPO_final.zip` | `--scenario`, `--suites`, `--n-episodes` (5), `--seed` (0), `--vehicle`, `--render`, `--save-gif`, `--save-plots` |
| `calm mpc`, `just mpc` | MPC baseline, needs acados | `--scenario`, `--suites`, `--vehicle`, `--no-perception`, `--horizon`, `--seed`, `--render`, `--save-gif`, `--save-plots` |
| `calm reactive`, `just reactive-driver` | reactive driver baseline | as `calm mpc`, with `--preview` (metres) instead of `--horizon` |
| `calm compare <run> [checkpoint]`, `just compare` | comfort of PPO, MPC and the recorded drive on the same roads | `--suites`, `--mpc-objective comfort_map\|frontier`, `--horizon`, `--real-root`, `--uncalibrated`, `--allow-missing-human`, `--seed`, `--out` |
| `calm reproduce`, `just reproduce` | trains several seeds and compares each | `--seeds`, `--timesteps`, `--prefix`, `--suites`, `--skip-train`, `--dry-run` |
| `just install` | creates `.venv`, installs CPU PyTorch, `requirements.txt`, the package | |
| `just build-acados` | builds the acados submodule and installs `acados_template`; separate recipe on Windows | |
| `just test` | the fast tests with `MPLBACKEND=Agg` | |
| `just lint` | pyflakes on `src` and `tests` | |

`calm eval` and `calm compare` need the run name first, otherwise they print their usage line. `calm train` also takes `--algo PPO`, the only choice.

## Scenario selection

- `--scenario <name or path>`: one scenario, a file name without extension (`c`, `nardo_sb01`) or a path to a `.yaml` or `.json` file.
- `--suites <folder> ...`: a folder in `config/eval/scenarios/` (`01_single_bump`, `10_potholes`), or `real` (held out), `real_train`, `real_all`, `flat`.
- Neither: the suites enabled in `config/eval/eval_suites.yaml`.

With `--render`, `calm mpc` and `calm reactive` step through the scenarios when `--scenario` is not given.

## Environment variables

Put the first three in a `.env` file in the repository root, `just` reads it.

- `CALM_PYTHON`: interpreter for the recipes [default `.venv/bin/python3`, `.venv/Scripts/python.exe` on Windows].
- `CALM_PIP`: pip for `just install` and `just build-acados`.
- `CALM_SEQUENCES`: folders with the recorded drives (`sequences` of CALM-Germany and CALM-Italy), separated by `:`. Read by `calm compare`, `calm reproduce` and `calm doctor`.
- `ACADOS_SOURCE_DIR`: acados folder. The MPC uses `acados/` in the repository when `acados/lib` exists, otherwise this variable.
- `MPLBACKEND`: `Agg` forces the headless backend.
- `NUMBA_CACHE_DIR`: set by the Dockerfile to `/tmp/numba_cache`.

See also: [Getting started](Getting-Started), [PPO training](PPO-Training), [Evaluation](Evaluation), [Troubleshooting](Troubleshooting)
