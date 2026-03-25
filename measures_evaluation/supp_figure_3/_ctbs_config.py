from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CTBS_CONFIG_PATH = REPO_ROOT / "ctbs_config.json"


def load_ctbs_runtime_config() -> dict:
    with open(CTBS_CONFIG_PATH, "r") as f:
        return json.load(f)


def configured_cnp2cnp_file() -> Path:
    return Path(load_ctbs_runtime_config()["cnp2cnp_FILE"]).expanduser().resolve()


def configured_cnp2cnp_module_dir() -> Path:
    return configured_cnp2cnp_file().parent
