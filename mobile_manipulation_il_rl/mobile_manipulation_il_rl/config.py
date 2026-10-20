from __future__ import annotations

import copy
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    source = Path(path) if path else ROOT / "configs" / "default.yaml"
    with source.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    sim = cfg["simulation"]
    if sim["physics_hz"] % sim["control_hz"]:
        raise ValueError("physics_hz must be divisible by control_hz")
    if cfg["task"]["grasp_mode"] not in {"physical", "assisted"}:
        raise ValueError("grasp_mode must be physical or assisted")
    if len(cfg["robot"]["arm_link_lengths"]) != 3:
        raise ValueError("the arm requires exactly three links")


def make_run_dir(kind: str, base: str | Path | None = None) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = Path(base or ROOT / "outputs") / f"{kind}_{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def save_run_metadata(out: Path, cfg: dict, command: str, seed: int) -> None:
    with (out / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    (out / "metadata.json").write_text(
        json.dumps({"command": command, "seed": seed, "created_at": dt.datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def with_seed(cfg: dict, seed: int) -> dict:
    result = copy.deepcopy(cfg)
    result["seed"] = seed
    return result

