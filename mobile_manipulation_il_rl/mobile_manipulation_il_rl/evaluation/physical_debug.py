from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from ..controllers import AnalyticTeacher
from ..envs.fsm import Phase


STAGES = ("reach", "grasp", "grasp_lift", "fixed_pick_place", "move_pick", "transport", "full")


def _fixed_pick_setup(env) -> None:
    pick = np.asarray(env.cfg["task"]["pick_xy"], float)
    bq = env.model.jnt_qposadr[env.base_joint]
    env.data.qpos[bq:bq + 7] = [pick[0] - env.cfg["task"]["stand_off"], pick[1],
                                env.cfg["robot"]["wheel_radius"], 1, 0, 0, 0]
    oq = env.model.jnt_qposadr[env.obj_joint]
    env.data.qpos[oq:oq + 3] = [*pick, env.cfg["task"]["table_size"][2] + env.cfg["task"]["object_size"][2] + .002]
    env.data.qvel[:] = 0
    env.fsm.phase = Phase.REACH; env.fsm.phase_time = 0
    env._episode_targets["pick_xy"] = pick.tolist(); env._episode_targets["active_xy"] = pick.tolist()
    mujoco.mj_forward(env.model, env.data)


def _stage_success(stage: str, phase: Phase) -> bool:
    targets = {"reach": Phase.GRASP, "grasp": Phase.LIFT, "grasp_lift": Phase.NAVIGATE_TO_PLACE,
               "move_pick": Phase.LIFT, "transport": Phase.PLACE}
    return phase.value >= targets[stage].value and phase != Phase.FAILURE if stage in targets else phase == Phase.SUCCESS


def run_physical_stages(env_factory, output: Path, episodes: int, seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=True); summaries = {}
    for stage in STAGES:
        records, failures = [], Counter()
        for episode in range(episodes):
            env = env_factory(); obs, _ = env.reset(seed=seed + episode, options={"randomization": "fixed"})
            if stage in {"reach", "grasp", "grasp_lift", "fixed_pick_place", "transport"}: _fixed_pick_setup(env)
            teacher = AnalyticTeacher(env); teacher.reset(); trace, events, previous = [], [], env.fsm.phase
            max_steps = env.max_steps if stage in {"fixed_pick_place", "transport", "full"} else int(25 / env.dt)
            for step in range(max_steps):
                action = teacher.act(obs); obs, _, terminated, truncated, info = env.step(action)
                state = env.diagnostic_state(teacher); state.update({"episode": episode, "step": step})
                trace.append(state)
                if env.fsm.phase != previous:
                    events.append({"from": previous.name, "to": env.fsm.phase.name, "time_s": state["time_s"],
                                   "reason": info["failure_reason"] or "condition_met"}); previous = env.fsm.phase
                if _stage_success(stage, env.fsm.phase) or terminated or truncated: break
            success = _stage_success(stage, env.fsm.phase)
            reason = "success" if success else info["failure_reason"] or f"stage_timeout:{env.fsm.phase.name}"
            failures[reason] += int(not success)
            records.append({"episode": episode, "success": success, "duration_s": (step + 1) * env.dt,
                            "final_phase": env.fsm.phase.name, "reason": reason,
                            "min_ee_error": min((x["ee_error"] for x in trace if x["ee_error"] is not None), default=None),
                            "max_contact_count": max(x["contact_count"] for x in trace),
                            "bilateral_contact_steps": sum(x["left_contact"] and x["right_contact"] for x in trace),
                            "min_joint_margin": min(min(x["joint_limit_margins"]) for x in trace), "events": events})
            ep = output / stage / f"episode_{episode:03d}"; ep.mkdir(parents=True, exist_ok=True)
            (ep / "trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
            (ep / "summary.json").write_text(json.dumps(records[-1], indent=2), encoding="utf-8")
            env.close()
        summaries[stage] = {"episodes": episodes, "success_rate": sum(x["success"] for x in records) / episodes,
                            "mean_duration_s": float(np.mean([x["duration_s"] for x in records if x["success"]])) if any(x["success"] for x in records) else None,
                            "failures": dict(failures), "records": records}
    (output / "physical_stages.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (output / "physical_stages.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stage", "success_rate", "mean_duration_s"]); writer.writeheader()
        writer.writerows({"stage": k, "success_rate": v["success_rate"], "mean_duration_s": v["mean_duration_s"]} for k, v in summaries.items())
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(list(summaries), [summaries[x]["success_rate"] for x in summaries], color="#315f85")
        ax.set_ylim(0, 1); ax.set_ylabel("success rate"); ax.set_title("Physical-mode staged verification")
        ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(output / "physical_stages.png", dpi=160); plt.close(fig)
    except (ImportError, OSError):
        pass
    return summaries
