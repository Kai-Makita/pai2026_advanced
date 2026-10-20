# Architecture

```text
YAML -> parameterized MJCF -> MobilePickPlaceEnv -> {teacher, BC, PPO}
                              |             |          |
                              v             v          v
                         episode NPZ -> Actor 40-256-256-7
                              |             |
                              +------ evaluation ------+
                                           |
                                  JSON / CSV / PNG / MP4
                                           |
                                      Streamlit
```

`envs` owns physics, observation/action contracts, rewards, and the FSM. `controllers` may inspect MuJoCo state only to generate demonstrations; learned policies receive only the 40-dimensional observation. `learning` owns datasets and models. `evaluation` treats all policies through the same episode loop.

