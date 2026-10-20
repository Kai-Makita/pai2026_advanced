from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

from .bc import Actor


POLICY_KWARGS = {"activation_fn": torch.nn.ReLU, "net_arch": {"pi": [256, 256], "vf": [256, 256]}}


def make_ppo(env_fn, cfg: dict, seed: int, tensorboard_log: str | None = None) -> PPO:
    vec = make_vec_env(env_fn, n_envs=cfg["n_envs"], seed=seed)
    return PPO("MlpPolicy", vec, learning_rate=cfg["learning_rate"], gamma=cfg["gamma"],
               gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"], policy_kwargs=POLICY_KWARGS,
               n_steps=cfg.get("n_steps", 2048), batch_size=cfg.get("batch_size", 64),
               tensorboard_log=tensorboard_log, seed=seed, verbose=1)


def transfer_bc_actor(ppo: PPO, bc_state: dict, log_std_init: float = -1.0) -> float:
    """Copy only actor mean layers; the value network remains independently initialized."""
    bc = Actor(); bc.load_state_dict(bc_state); bc.eval()
    target = ppo.policy
    src_linear = [bc.net[0], bc.net[2]]
    dst_linear = [target.mlp_extractor.policy_net[0], target.mlp_extractor.policy_net[2]]
    with torch.no_grad():
        for src, dst in zip(src_linear, dst_linear):
            dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
        target.action_net.weight.copy_(bc.net[4].weight); target.action_net.bias.copy_(bc.net[4].bias)
        target.log_std.fill_(log_std_init)
        probes = torch.linspace(-1, 1, 40 * 8, device=target.device).reshape(8, 40)
        bc_mean = bc(probes.cpu()).to(target.device)
        ppo_mean = target.action_net(target.mlp_extractor.forward_actor(target.extract_features(probes)))
        max_error = float(torch.max(torch.abs(bc_mean - ppo_mean)).cpu())
    if max_error > 1e-6:
        raise RuntimeError(f"BC actor transfer mismatch: {max_error}")
    return max_error


def train_ppo(env_fn, output: Path, cfg: dict, seed: int, bc_path: Path | None = None,
              resume: Path | None = None) -> PPO:
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        vec = make_vec_env(env_fn, n_envs=cfg["n_envs"], seed=seed)
        model = PPO.load(resume, env=vec, tensorboard_log=str(output / "tensorboard"))
    else:
        model = make_ppo(env_fn, cfg, seed, str(output / "tensorboard"))
        if bc_path:
            state = torch.load(bc_path, map_location="cpu", weights_only=True)
            error = transfer_bc_actor(model, state, cfg["log_std_init"])
            (output / "transfer_verification.txt").write_text(f"max_action_mean_error={error:.9g}\n", encoding="utf-8")
            from ..evaluation.runner import evaluate_policy
            evaluate_policy(env_fn, model, cfg.get("pre_finetune_eval_episodes", 20), seed,
                            output / "pre_finetune_evaluation")
    callback = CheckpointCallback(save_freq=max(10_000 // cfg["n_envs"], 1), save_path=str(output / "checkpoints"), name_prefix="ppo")
    model.learn(total_timesteps=cfg["total_timesteps"], callback=callback, reset_num_timesteps=resume is None)
    model.save(output / "model_final")
    return model

