import yaml

from utils.paths import CONFIG

_DIRS = {
    "env.yaml": "gym_env",
    "render.yaml": "gym_env",
    "reward.yaml": "reward",
    "preview.yaml": "preview",
}


def _find_config_file(filename: str):
    if filename not in _DIRS:
        raise ValueError(f"Unknown config file: {filename}")
    path = CONFIG / _DIRS[filename] / filename
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def _load_yaml(filename: str) -> dict:
    path = _find_config_file(filename)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format in {path}")
    return data
