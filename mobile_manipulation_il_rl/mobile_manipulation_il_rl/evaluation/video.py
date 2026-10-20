from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio


def record_episode(env, policy, output: Path, seed: int = 0) -> dict:
    obs, _ = env.reset(seed=seed); frames = [env.render()]
    if hasattr(policy, "reset"): policy.reset()
    for step in range(env.max_steps):
        action = policy.act(obs) if hasattr(policy, "act") else policy.predict(obs, deterministic=True)[0]
        obs, _, terminated, truncated, info = env.step(action); frames.append(env.render())
        if terminated or truncated: break
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=env.metadata["render_fps"], codec="libx264")
    return {"success": info["success"], "steps": step + 1, "grasp_mode": info["grasp_mode"]}

