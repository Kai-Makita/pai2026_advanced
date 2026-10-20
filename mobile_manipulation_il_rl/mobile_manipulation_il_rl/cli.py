from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

from .config import ROOT, load_config, make_run_dir, save_run_metadata, seed_everything
from .controllers import AnalyticTeacher
from .envs import MobilePickPlaceEnv
from .evaluation.runner import evaluate_policy


def env_factory(cfg, render_mode=None, evaluation_mode="train"):
    return lambda: MobilePickPlaceEnv(cfg, render_mode=render_mode, evaluation_mode=evaluation_mode)


def policy_from_args(kind: str, model: str | None, env=None):
    if kind == "teacher":
        if env is None: return lambda e: AnalyticTeacher(e)
        return AnalyticTeacher(env)
    if not model: raise ValueError(f"--model is required for {kind}")
    if kind == "bc":
        from .learning.bc import load_bc
        return load_bc(model)
    from stable_baselines3 import PPO
    return PPO.load(model)


def common_start(args, kind):
    cfg = load_config(args.config); seed_everything(args.seed)
    if args.grasp_mode:
        cfg["task"]["grasp_mode"] = args.grasp_mode
    out = make_run_dir(kind, args.output_root); save_run_metadata(out, cfg, " ".join(args.argv), args.seed)
    return cfg, out


def cmd_teacher_test(args):
    cfg, out = common_start(args, "teacher_test")
    result = evaluate_policy(env_factory(cfg), lambda env: AnalyticTeacher(env), args.episodes, args.seed, out)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


def cmd_collect(args):
    from .learning.dataset import collect_episode, dataset_statistics, split_for_episode
    cfg, out = common_start(args, "collection")
    root = Path(args.dataset); started = time.perf_counter(); summaries = []
    env = MobilePickPlaceEnv(cfg); teacher = AnalyticTeacher(env)
    for i in range(args.episodes):
        split = split_for_episode(i, args.seed)
        summaries.append(collect_episode(env, teacher, args.seed + i, root / split / f"episode_{i:05d}.npz"))
    env.close()
    stats = dataset_statistics(root); stats["collection_wall_time_s"] = time.perf_counter() - started
    (out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def cmd_bc(args):
    from .learning.bc import train_bc
    cfg, out = common_start(args, "bc")
    started = time.perf_counter(); result = train_bc(Path(args.dataset), out, cfg["learning"]["bc"], args.device)
    result["training_wall_time_s"] = time.perf_counter() - started
    (out / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); print(out)


def cmd_ppo(args, from_bc=False):
    from .learning.ppo import train_ppo
    cfg, out = common_start(args, "bc_ppo" if from_bc else "ppo")
    if args.timesteps: cfg["learning"]["ppo"]["total_timesteps"] = args.timesteps
    model = train_ppo(env_factory(cfg), out, cfg["learning"]["ppo"], args.seed,
                      Path(args.bc_model) if from_bc else None, Path(args.resume) if args.resume else None)
    # Required measurement immediately after transfer is recorded before learning by a separate command;
    # here the exact numerical transfer check is always stored in the run directory.
    print(out)


def cmd_evaluate(args):
    cfg, out = common_start(args, "evaluation")
    policy = policy_from_args(args.policy, args.model)
    all_results = {}
    for condition in args.conditions:
        all_results[condition] = evaluate_policy(env_factory(cfg, evaluation_mode=condition), policy, args.episodes,
                                                 args.seed, out / condition, condition)
    (out / "all_evaluations.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8"); print(out)


def cmd_video(args):
    from .evaluation.video import record_episode
    cfg, out = common_start(args, "video")
    env = MobilePickPlaceEnv(cfg, render_mode="rgb_array")
    policy = policy_from_args(args.policy, args.model, env)
    result = record_episode(env, policy, out / f"{args.policy}_{cfg['task']['grasp_mode']}.mp4", args.seed)
    env.close(); (out / "video.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); print(out)


def cmd_viewer(args):
    import time as timer
    cfg = load_config(args.config)
    if args.grasp_mode: cfg["task"]["grasp_mode"] = args.grasp_mode
    env = MobilePickPlaceEnv(cfg, render_mode="human")
    policy = policy_from_args(args.policy, args.model, env); obs, _ = env.reset(seed=args.seed)
    try:
        while True:
            started = timer.perf_counter(); action = policy.act(obs) if hasattr(policy, "act") else policy.predict(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(action)
            if term or trunc: obs, _ = env.reset(); getattr(policy, "reset", lambda: None)()
            timer.sleep(max(0, env.dt / args.speed - (timer.perf_counter() - started)))
    except (KeyboardInterrupt, RuntimeError): pass
    finally: env.close()


def cmd_compare(args):
    import pandas as pd
    from .evaluation.plots import comparison_plot, flatten_evaluation
    cfg, out = common_start(args, "comparison"); rows = []
    models = {"bc": args.bc_model, "ppo": args.ppo_model, "bc_ppo": args.bc_ppo_model}
    for seed in args.seeds:
        for method, path in models.items():
            policy = policy_from_args("bc" if method == "bc" else "ppo", path)
            for condition in ("train", "ood"):
                result = evaluate_policy(env_factory(cfg, evaluation_mode=condition), policy, args.episodes, seed,
                                         out / method / f"seed_{seed}" / condition, condition)
                rows.append(flatten_evaluation(method, seed, condition, result))
    csv_path = out / "comparison.csv"; pd.DataFrame(rows).to_csv(csv_path, index=False)
    comparison_plot(csv_path, out / "comparison.png")
    (out / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8"); print(out)


def cmd_physical_debug(args):
    from .evaluation.physical_debug import run_physical_stages
    cfg, out = common_start(args, "physical_debug")
    cfg["task"]["grasp_mode"] = "physical"
    results = run_physical_stages(env_factory(cfg), out, args.episodes, args.seed)
    print(json.dumps({k: {x: y for x, y in v.items() if x != "records"} for k, v in results.items()}, indent=2))


def cmd_benchmark(args):
    import numpy as np
    cfg, out = common_start(args, "benchmark")
    env = MobilePickPlaceEnv(cfg); env.reset(seed=args.seed); action = np.zeros(7, np.float32)
    started = time.perf_counter()
    for _ in range(args.steps): env.step(action)
    elapsed = time.perf_counter() - started; env.close()
    result = {"control_steps": args.steps, "wall_time_s": elapsed, "control_steps_per_s": args.steps / elapsed,
              "estimated_100k_env_seconds": 100000 * elapsed / args.steps,
              "note": "environment only; neural-network updates and vectorization are excluded"}
    (out / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


def parser():
    p = argparse.ArgumentParser(description="Independent mobile manipulation IL/RL benchmark")
    p.add_argument("--config", default=str(ROOT / "configs/default.yaml")); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grasp-mode", choices=["physical", "assisted"], help="explicit debug override; physical is the default")
    p.add_argument("--output-root", default=str(ROOT / "outputs")); sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("teacher-test"); t.add_argument("--episodes", type=int, default=100); t.set_defaults(func=cmd_teacher_test)
    c = sub.add_parser("collect"); c.add_argument("--episodes", type=int, default=400); c.add_argument("--dataset", default=str(ROOT / "datasets/default")); c.set_defaults(func=cmd_collect)
    b = sub.add_parser("train-bc"); b.add_argument("--dataset", default=str(ROOT / "datasets/default")); b.add_argument("--device", default="cpu"); b.set_defaults(func=cmd_bc)
    for name, from_bc in (("train-ppo", False), ("train-bc-ppo", True)):
        x = sub.add_parser(name); x.add_argument("--timesteps", type=int); x.add_argument("--resume")
        if from_bc: x.add_argument("--bc-model", required=True)
        x.set_defaults(func=lambda a, flag=from_bc: cmd_ppo(a, flag))
    e = sub.add_parser("evaluate"); e.add_argument("--policy", choices=["teacher", "bc", "ppo", "bc_ppo"], required=True); e.add_argument("--model"); e.add_argument("--episodes", type=int, default=100); e.add_argument("--conditions", nargs="+", choices=["train", "ood"], default=["train", "ood"]); e.set_defaults(func=cmd_evaluate)
    v = sub.add_parser("video"); v.add_argument("--policy", choices=["teacher", "bc", "ppo", "bc_ppo"], required=True); v.add_argument("--model"); v.set_defaults(func=cmd_video)
    w = sub.add_parser("viewer"); w.add_argument("--policy", choices=["teacher", "bc", "ppo", "bc_ppo"], required=True); w.add_argument("--model"); w.add_argument("--speed", type=float, default=1); w.set_defaults(func=cmd_viewer)
    q = sub.add_parser("compare"); q.add_argument("--bc-model", required=True); q.add_argument("--ppo-model", required=True); q.add_argument("--bc-ppo-model", required=True); q.add_argument("--episodes", type=int, default=100); q.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33]); q.set_defaults(func=cmd_compare)
    d = sub.add_parser("physical-debug"); d.add_argument("--episodes", type=int, default=5); d.set_defaults(func=cmd_physical_debug)
    z = sub.add_parser("benchmark"); z.add_argument("--steps", type=int, default=1000); z.set_defaults(func=cmd_benchmark)
    return p


def main():
    p = parser(); args = p.parse_args(); args.argv = ["python", "-m", "mobile_manipulation_il_rl.cli", *([] if args.command is None else [args.command])]; args.func(args)


if __name__ == "__main__": main()

