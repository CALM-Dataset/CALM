import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ParamsFiles:

    gym_env = "env.yaml"
    gym_env_dir = "gym_env"

    render  = "render.yaml"
    render_dir = "gym_env"

    reward  = "reward.yaml"
    reward_dir = "reward"

    preview = "preview.yaml"
    preview_dir = "preview"

    road    = "random_road.yaml"
    road_dir = "catalog"

def _find_config_file(filename: str) -> Path:

    _known = [ParamsFiles.gym_env, ParamsFiles.render, ParamsFiles.reward,
              ParamsFiles.preview, ParamsFiles.road]
    if filename not in _known:
        raise ValueError(f"Unknown config file: {filename}")

    if filename == ParamsFiles.gym_env:
        config_dir = ParamsFiles.gym_env_dir
    elif filename == ParamsFiles.render:
        config_dir = ParamsFiles.render_dir
    elif filename == ParamsFiles.reward:
        config_dir = ParamsFiles.reward_dir
    elif filename == ParamsFiles.preview:
        config_dir = ParamsFiles.preview_dir
    else:
        config_dir = ParamsFiles.road_dir

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        candidate = parent / "config" / config_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Config file not found: {filename}")


def _load_yaml(filename: str) -> dict:
    path = _find_config_file(filename)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}")
    return data