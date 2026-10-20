from __future__ import annotations

import json
from pathlib import Path

import numpy as np


FIELDS = ("observations", "actions", "next_observations", "rewards", "phases", "terminated", "truncated")


def collect_episode(env, teacher, seed: int, path: Path) -> dict:
    obs, reset_info = env.reset(seed=seed)
    teacher.reset()
    buffers = {k: [] for k in FIELDS}
    for _ in range(env.max_steps):
        action = teacher.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        values = (obs, action, next_obs, reward, info["phase_id"], terminated, truncated)
        for key, value in zip(FIELDS, values):
            buffers[key].append(value)
        obs = next_obs
        if terminated or truncated:
            break
    arrays = {k: np.asarray(v) for k, v in buffers.items()}
    arrays["success"] = np.asarray(info["success"])
    arrays["initial_state"] = np.asarray(json.dumps(reset_info["initial_state"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {"steps": len(buffers["actions"]), "success": bool(info["success"]), "phase_counts": dict(zip(*np.unique(arrays["phases"], return_counts=True)))}


def split_for_episode(index: int, seed: int) -> str:
    # Split before collection by episode identity, never by transitions.
    value = np.random.default_rng(seed + index).random()
    return "train" if value < 0.8 else "validation" if value < 0.9 else "test"


class DemonstrationDataset:
    def __init__(self, root: str | Path, split: str):
        files = sorted((Path(root) / split).glob("episode_*.npz"))
        if not files:
            raise FileNotFoundError(f"no demonstrations in {Path(root) / split}")
        chunks = [np.load(f) for f in files]
        self.observations = np.concatenate([x["observations"] for x in chunks]).astype(np.float32)
        self.actions = np.concatenate([x["actions"] for x in chunks]).astype(np.float32)
        self.phases = np.concatenate([x["phases"] for x in chunks]).astype(np.int64)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return self.observations[index], self.actions[index], self.phases[index]


def dataset_statistics(root: str | Path) -> dict:
    result = {}
    for split in ("train", "validation", "test"):
        files = sorted((Path(root) / split).glob("episode_*.npz"))
        phases, successes, count = [], [], 0
        for f in files:
            data = np.load(f)
            phases.extend(data["phases"].tolist()); successes.append(bool(data["success"])); count += len(data["actions"])
        keys, vals = np.unique(phases, return_counts=True) if phases else ([], [])
        result[split] = {"episodes": len(files), "transitions": count,
                         "success_rate": sum(successes) / len(successes) if successes else None,
                         "phase_counts": {str(k): int(v) for k, v in zip(keys, vals)}}
    return result

