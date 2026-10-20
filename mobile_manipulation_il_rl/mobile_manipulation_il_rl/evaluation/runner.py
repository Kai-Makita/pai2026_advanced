from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def evaluate_policy(env_factory, policy, episodes: int, seed: int = 0, output: Path | None = None,
                    randomization: str = "train") -> dict:
    records, failures, phase_visits = [], Counter(), Counter()
    started = time.perf_counter()
    for episode in range(episodes):
        env = env_factory()
        obs, _ = env.reset(seed=seed + episode, options={"randomization": randomization})
        agent = policy(env) if callable(policy) and not hasattr(policy, "act") and not hasattr(policy, "predict") else policy
        if hasattr(agent, "reset"):
            agent.reset()
        phase_seen = set()
        total_reward = 0.0
        for steps in range(env.max_steps):
            action = agent.act(obs) if hasattr(agent, "act") else agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            phase_seen.add(info["phase"])
            if terminated or truncated:
                break
        success = bool(info["success"])
        if not success:
            failures[info["failure_reason"] or "unknown"] += 1
        phase_visits.update(phase_seen)
        records.append({"episode": episode, "seed": seed + episode, "success": success,
                        "steps": steps + 1, "completion_time_s": (steps + 1) * env.dt,
                        "reward": total_reward, "failure_reason": info["failure_reason"],
                        "reward_sums": info["reward_sums"]})
        env.close()
    elapsed = time.perf_counter() - started
    successes = [r for r in records if r["success"]]
    summary = {
        "episodes": episodes, "success_rate": len(successes) / episodes,
        "mean_completion_time_s": float(np.mean([r["completion_time_s"] for r in successes])) if successes else None,
        "object_drop_rate": failures["object_drop"] / episodes, "failure_reasons": dict(failures),
        "phase_reach_rate": {k: v / episodes for k, v in phase_visits.items()},
        "wall_time_s": elapsed, "randomization": randomization, "records": records,
    }
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

