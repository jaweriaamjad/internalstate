"""
Shared config loader for internalstate_code.
Loads paths and paper settings from config/*.json.
Paths are resolved relative to the repo root (this directory) if not absolute.
"""

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def _load_json(filename, required=True):
    path = _REPO_ROOT / "config" / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Copy from config/{filename}.example and edit."
            )
        return {}
    with open(path) as f:
        return json.load(f)


def _resolve_path(p):
    """If relative, resolve against repo root."""
    path = Path(p)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def get_paths():
    """
    Load config/paths.json and return data_dir, results_dir, figures_dir, models_dir
    as absolute Paths. Relative paths are resolved against the repo root.
    """
    cfg = _load_json("paths.json")
    out = {
        "data_dir": _resolve_path(cfg["data_dir"]),
        "results_dir": _resolve_path(cfg["results_dir"]),
        "models_dir": _resolve_path(cfg["models_dir"]),
    }
    if "figures_dir" in cfg:
        out["figures_dir"] = _resolve_path(cfg["figures_dir"])
    else:
        out["figures_dir"] = _REPO_ROOT / "figures"
    return out


def get_paper_config():
    """Load config/paper_config.json (tag, vae_run_folder, glmhmm_K, etc.)."""
    return _load_json("paper_config.json")


def get_repo_root():
    """Return the repo root directory as a Path."""
    return _REPO_ROOT
