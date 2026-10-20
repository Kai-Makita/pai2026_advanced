from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import pandas as pd


def comparison_plot(csv_path: Path, output: Path) -> None:
    df = pd.read_csv(csv_path)
    metrics = ["success_rate", "mean_completion_time_s", "object_drop_rate"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, metrics):
        df.groupby("method")[metric].agg(["mean", "std"]).plot.bar(y="mean", yerr="std", ax=ax, legend=False, capsize=3)
        ax.set_title(metric); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def flatten_evaluation(method: str, seed: int, condition: str, result: dict, costs: dict | None = None) -> dict:
    return {"method": method, "seed": seed, "condition": condition,
            "success_rate": result["success_rate"], "mean_completion_time_s": result["mean_completion_time_s"],
            "object_drop_rate": result["object_drop_rate"], "wall_time_s": result["wall_time_s"], **(costs or {})}

