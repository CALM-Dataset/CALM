# Troubleshooting

Start with `calm doctor`. Only `FAIL` lines are required, `warn` lines are parts you may not need.

## Problems

### Install

```
python ... (need 3.10 or newer)
```
Install a newer Python and run `just install` again.

```
<package> ... run: just install
```
The package is missing in the environment you run from. Run `just install`, or install into that environment.

`just` looks for a `.venv` that does not exist.

You use conda. Set `CALM_PYTHON` and `CALM_PIP` in `.env`.

```
bash: calm: command not found
```
The environment is not active. Use `.venv/bin/calm` or `python -m calm`.

### MPC and acados

```
acados: submodule not checked out
```
```bash
git submodule update --init --recursive
```

```
acados library: not built
```
Run `just build-acados` with the same interpreter (needs cmake and a C compiler). A build elsewhere works with `ACADOS_SOURCE_DIR`.

```
dlopen ... libqpOASES_e.dylib: not found
```
macOS only. Run `just build-acados` again, it adds `@loader_path`. No compiler: `xcode-select --install`, and install cmake separately.

Windows build fails: install cmake and the Visual Studio C++ build tools, then `just build-acados`. The first MPC run downloads `t_renderer` from GitHub, so it fails offline.

### Data

```
cannot download CALM-Dataset/CALM-Scenarios
```
The dataset is gated. Request access on the [CALM-Scenarios page](https://huggingface.co/datasets/CALM-Dataset/CALM-Scenarios), run `hf auth login` with a read token, then `calm fetch`.

```
No scenarios found. Enable a suite in config/eval/eval_suites.yaml ...
```
The real folders are empty and no suite is enabled. Run `calm fetch`, or pass `--suites 01_single_bump`.

```
recorded drive not found for N real scenarios
```
`CALM_SEQUENCES` is unset. Point it at the `sequences` folders, or pass `--allow-missing-human`.

```bash
export CALM_SEQUENCES=/data/CALM-Germany/sequences:/data/CALM-Italy/sequences
```

`calm fetch` does nothing when the three local folders already hold JSON files. After the dataset changed, start clean:

```bash
rm -rf config/train/scenarios/real config/eval/scenarios/real config/eval/scenarios/real_train
rm -f config/perception/perception_model.yaml
calm fetch
```

### Models

```
usage: calm eval <run> [checkpoint] [eval options]
```
The run name is missing: `calm eval demo --scenario c`.

```
Model not found: ...
```
The folder is `logs/models/PPO/<run>/`. Check the run and checkpoint names.

```
Could not find vecnormalize.pkl for ...
```
Put `vecnormalize.pkl` next to the model `.zip`. For `policy.safetensors` the file is `vecnormalize.json`.

> [!WARNING]
> `PPO_final.zip` and `vecnormalize.pkl` are pickles. Load them only from trusted sources. The vehicle must match: a `--vehicle quarter` model is evaluated with `--vehicle quarter`.

### Windows and Docker

```
ValueError: cannot find context for 'fork'
```
Fixed: training falls back to one process. Update the repository.

```
cannot cache function ... no locator available
```
Numba has no writable cache folder, for example in a container with another user id. Use the Dockerfile, which sets `NUMBA_CACHE_DIR`, or set it to a writable folder.

### Rendering

- No window on Linux: no `DISPLAY` or `WAYLAND_DISPLAY`, so Agg is used. Use a desktop session, X11 forwarding or `--save-gif`.
- No window on macOS or desktop Linux: `MPLBACKEND=Agg` is set in the shell or `.env`. Unset it.
- Keys do nothing: click into the window.
- Window covers the screen: set `RENDER_FULLSCREEN: false` in `config/gym_env/render.yaml`.
- No GIF written: drop `--render`. `calm mpc` and `calm reactive` need a single `--scenario`.

```
Pillow not installed - cannot save GIF
```
Run `pip install Pillow`.

```
[render] music is on but no audio backend
```
Install pygame or set `RENDER_MUSIC: false`.

### Speed

- Playback is slow: it starts at 0.5x. Press `f`, or drop `--render`.
- The first run is slow: Numba compiles the vehicle dynamics once, then caches.
- Training is slow (CPU only): shorten with `--timesteps`, match `--n-envs` to the cores, run one training at a time.
- `calm compare` takes long because the MPC runs on every scenario. `calm reproduce --dry-run` lists the steps, `--skip-train` reuses runs.

See also: [Getting started](Getting-Started), [Command line](Command-Line), [Simulator](Simulator)
