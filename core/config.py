"""Settings: config.json, re-read whenever it is needed so edits take effect without a restart.

    from core.config import current_config, get, require

    cfg = current_config()
    url = require(cfg, "models.local1.url")         # raises ConfigError with a plain message if unset
    target = get(cfg, "translation.target_new_chars", 50)

If config.json is missing or half-saved, current_config() keeps the last good copy.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE / "config.json"

_last_good = {}


class ConfigError(RuntimeError):
    """A settings problem, worded so it can be shown to the user as is."""


def load_config(path=None):
    """Read config.json strictly. Raises ConfigError if it is missing or not valid JSON."""
    global _last_good
    path = Path(path) if path else CONFIG_PATH
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"Could not read {path.name}: {e}")
    except ValueError as e:
        raise ConfigError(f"{path.name} is not valid JSON: {e}")
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path.name} must hold a JSON object.")
    _last_good = cfg
    return cfg


def current_config(path=None):
    """Re-read config.json; if that fails, keep using the last good copy (and say so)."""
    try:
        return load_config(path)
    except ConfigError as e:
        if _last_good:
            print(f"  {e}; using the previous settings")
            return _last_good
        raise


def get(cfg, dotted, default=None):
    """cfg["a"]["b"] written as get(cfg, "a.b"). Returns default if any step is missing."""
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def require(cfg, dotted):
    """Like get(), but a missing, null or empty value raises ConfigError naming the key."""
    value = get(cfg, dotted)
    if value is None or value == "":
        raise ConfigError(f'Set "{dotted}" in config.json.')
    return value


def resolve_path(cfg, key):
    """A folder from the "paths" section, as an absolute path under the project folder."""
    rel = get(cfg, f"paths.{key}")
    if not rel:
        raise ConfigError(f'Set "paths.{key}" in config.json.')
    p = Path(rel).expanduser()
    return p if p.is_absolute() else BASE / p
