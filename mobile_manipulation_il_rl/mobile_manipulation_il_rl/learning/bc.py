from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import DemonstrationDataset


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(40, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 7))

    def forward(self, x):
        return self.net(x)


class BCPolicy:
    def __init__(self, actor: Actor, device: str = "cpu"):
        self.actor, self.device = actor.eval().to(device), device

    def act(self, obs):
        with torch.no_grad():
            return self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).clamp(-1, 1).cpu().numpy()


def load_bc(path: str | Path, device="cpu") -> BCPolicy:
    actor = Actor(); actor.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return BCPolicy(actor, device)


def train_bc(dataset_root: Path, output: Path, cfg: dict, device: str = "cpu") -> dict:
    train, val = DemonstrationDataset(dataset_root, "train"), DemonstrationDataset(dataset_root, "validation")
    actor = Actor().to(device)
    counts = np.bincount(train.phases, minlength=6)
    weights = 1 / np.maximum(counts[train.phases], 1)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True) if cfg["phase_balancing"] else None
    loader = DataLoader(train, batch_size=cfg["batch_size"], sampler=sampler, shuffle=sampler is None)
    val_loader = DataLoader(val, batch_size=cfg["batch_size"])
    optimizer, loss_fn = torch.optim.Adam(actor.parameters(), lr=cfg["learning_rate"]), nn.MSELoss()
    output.mkdir(parents=True, exist_ok=True); history, best = [], float("inf")
    for epoch in range(cfg["epochs"]):
        actor.train(); train_losses = []
        for obs, actions, _ in loader:
            loss = loss_fn(actor(obs.to(device)), actions.to(device)); optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_losses.append(loss.item())
        actor.eval(); val_losses = []
        with torch.no_grad():
            for obs, actions, _ in val_loader:
                val_losses.append(loss_fn(actor(obs.to(device)), actions.to(device)).item())
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "validation_loss": float(np.mean(val_losses))}
        history.append(row)
        if row["validation_loss"] < best:
            best = row["validation_loss"]; torch.save(actor.state_dict(), output / "bc_best.pt")
    torch.save(actor.state_dict(), output / "bc_final.pt")
    with (output / "bc_history.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
    return {"best_validation_loss": best, "epochs": cfg["epochs"]}

