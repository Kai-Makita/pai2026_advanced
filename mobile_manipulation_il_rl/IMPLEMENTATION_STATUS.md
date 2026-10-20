# Implementation and verification status

Date: 2026-10-21. This file reports commands actually run during development; it does not claim benchmark performance.

| Phase | Implemented | Verification performed |
|---|---|---|
| 1: environment | Parameterized original MJCF, two tables, free object, differential wheels, passive supports, 4-DoF arm, two fingers, SE(2) velocity-controlled base, Gymnasium API, viewer | MJCF loaded by MuJoCo 3.16; `check_env`; observation/action and finite-value tests; gravity test |
| 2: teacher | Proportional navigation, DLS IK, phase controller, failures and timeouts, physical and explicit assisted grasp branches | Three assisted episodes at seed 77: 3/3 success, mean 8.0 s. This is a debug-mode result. Physical mode did not achieve reliable success in the short tuning run. |
| 3: BC | Episode NPZ collection, leak-resistant episode split, statistics, balanced sampler, 40-256-256-7 Actor, checkpoints | 21 assisted debug episodes collected; 2-epoch smoke training completed; checkpoint reloaded and returned shape `(7,)` |
| 4: PPO | SB3 vector environments, checkpoints, resume, TensorBoard, common actor layout, BC actor transfer, independent critic, log-std initialization | 64-step PPO smoke update completed; saved model reloaded. Exact BC→PPO mean test passed. Full 100k+ run not executed. |
| 5: evaluation | train/OOD evaluation, phase and reward metrics, JSON/CSV, comparison plot, MP4 recording, viewer, Streamlit loader | Evaluation JSON paths exercised by teacher runs. Long comparison and video rendering were not executed in this Windows session. |

Final automated test result during development: `7 passed`. The suite covers MJCF construction, Gymnasium API, dimensions, finite observations/rewards, deterministic reset, gravity stability, FSM timeout/order, BC→PPO action-mean equality, PPO save/reload.

## Known incomplete work

The standard `physical` mode uses bilateral collision contact and friction. Staged debugging found and fixed table/base penetration, palm/object interference, insufficient finger preload, premature top-edge grasp detection, and 20 Hz base pose jumps. In the latest retained 3-episode staged configuration, fixed REACH, GRASP, and GRASP→LIFT each achieved 3/3. Contact was lost during base transport, so fixed Pick & Place and full-task success remain incomplete. No 100-episode success rate is claimed, and Physical training data was not collected. The `assisted` branch remains separate and uses explicit kinematic attachment solely to validate learning plumbing.

The 100-teacher-episode gate, 300–500 episode final dataset, 100k–500k PPO runs, three-seed benchmark, and final videos remain compute jobs for the target Ubuntu environment. Their commands are in `README.md`; generated outputs are intentionally excluded from Git.

