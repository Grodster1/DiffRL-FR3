# DiffRL-Panda

Comparison of Reinforcement Learning and Diffusion Policy in manipulator trajectory generation.

Engineering thesis project (Control Engineering and Robotics, Wrocław University of Science
and Technology, 2026/2027). The repository holds a complete simulation stack for a Franka
FR3 arm - Gazebo Harmonic scene, ROS 2 control chain, a Gymnasium environment, a scripted
expert for demonstration collection, and the training code for both families of methods.

## Goal

Train and systematically compare two approaches to generating manipulator trajectories on
the same pick & place task, under identical observation and action spaces:

- **Reinforcement Learning** - SAC and PPO (stable-baselines3).
- **Diffusion Policy** - CNN variant (1D U-Net + FiLM), trained per-task from scratch on demonstrations collected in this environment (Chi et al., 2023).

Comparison metrics: success rate, sample efficiency, trajectory smoothness (jerk integral,
EE path length), inference latency, and robustness to disturbances.

### Task

Pick a 5 cm cube off the main table and place it on one of two side tables, in three
difficulty levels:

| Level | Scene |
|---|---|
| L1 | Fixed cube position, fixed goal (left table) |
| L2 | Randomized cube position, goal drawn 50/50 between left and right table |
| L3 | L2 plus dynamic disturbances during motion |

The side tables are separated from the main one by a ~7.5 cm gap, so the cube cannot be
pushed to the goal - it has to be lifted. This removes the usual "pick & place degenerates
into push" failure mode without any reward engineering.

## Design decisions

A single control path is shared by RL, Diffusion Policy and the scripted expert - this is
the fairness condition of the whole comparison.

```
policy (ΔEE @ 20 Hz)
  -> action clip + workspace box clip
  -> damped least-squares IK on a 6x7 Jacobian (pinocchio, frame fr3_hand_tcp)
  -> joint_trajectory_controller (position)
  -> gz_ros2_control -> Gazebo Harmonic
  -> observations back to the policy
```

- **Action space:** 4D `(dx, dy, dz, g)` in `[-1, 1]`, orientation frozen (gripper pointing
  down), max 5 cm per step. Gripper command is continuous in the network and thresholded
  to binary.
- **Observation space:** 17D state-based oracle, normalized to `[-1, 1]` - EE position, 6D
  orientation, gripper aperture, EE-to-cube vector, cube-to-goal vector, grasp flag. Vision
  is explicitly out of scope (future work).
- **Reward:** shaped - reach distance penalty, grasp bonus, transport distance penalty,
  success bonus, small `||a||^2` penalty. No low-pass filtering on actions anywhere, so
  smoothness is reported from raw trajectories.
- **IK:** own DLS implementation rather than MoveIt Servo in the training loop -
  deterministic, unit-testable, no extra nodes.

Full rationale for every decision, including the rejected alternatives, is in
`docs/thesis-project-context.md`.

## Stack

| Component | Technology |
|---|---|
| Simulation | Gazebo Harmonic (Gz Sim 8), headless |
| Robotics framework | ROS 2 Jazzy (Ubuntu 24.04) |
| Manipulator | Franka FR3 |
| Control | ros2_control + gz_ros2_control |
| Kinematics | pinocchio |
| RL | stable-baselines3 (SAC, PPO) |
| Diffusion Policy | PyTorch + diffusers |
| Tracking | Weights & Biases |
| Containerization | Docker + Docker Compose |

Everything runs inside the container - the host has no ROS installation.

## Repository layout

```
DiffRL-Panda/
├── docker/                        # Dockerfile, compose, entrypoint
├── src/                           # ROS 2 workspace (colcon)
│   ├── franka_sim/                # ament_cmake: Gazebo scene, URDF, controllers, launch
│   │   ├── urdf/fr3_gazebo.urdf.xacro
│   │   ├── worlds/fr3_world.sdf
│   │   ├── config/controllers.yaml
│   │   └── launch/bringup.launch.py
│   └── franka_rl/                 # ament_python: Gym environment + RL
│       ├── config.py              # constants (thresholds, reward weights, geometry) - no ROS
│       ├── kinematics.py          # FK + Jacobian (pinocchio) - no ROS
│       ├── inverse_kinematics.py  # DLS + workspace/joint-limit clipping - no ROS
│       ├── observations.py        # observation dict + normalization - no ROS
│       ├── expert.py              # scripted expert, 8-phase state machine - no ROS
│       ├── ros_bridge.py          # the only module importing rclpy
│       ├── gym_env.py             # Gymnasium <-> ROS 2 wrapper
│       ├── collect_demos.py       # expert execution loop, demo recording
│       ├── train_sac.py
│       └── test/                  # pytest, runs without Gazebo
├── data/                          # demos, checkpoints, results (gitignored)
├── evaluation/                    # evaluation protocol
└── docs/                          # cheatsheet, design context, constant derivations
```

The ROS-free / ROS boundary is deliberate: unit tests for kinematics, IK and the expert run
without a live simulation, and Diffusion Policy training will read finished files from
`data/demos/` rather than driving the robot.

## Quick start

```bash
# 1. Container
cd docker && docker compose up -d && cd ..
docker ps                                  # expect a running 'franka_sim'

# 2. Build
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash; \
  cd /ws && colcon build --packages-select franka_sim franka_rl && source install/setup.bash"

# 3. Bring up the simulation (Gazebo headless + controllers)
docker exec -it franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /opt/franka_ws/install/setup.bash && \
  source /ws/install/setup.bash; ros2 launch franka_sim bringup.launch.py"

# 4. In a second shell: collect demonstrations
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; \
  cd /ws && ros2 run franka_rl collect_demos --level L2 --seed 0 --episodes 100 --save"

# 5. Or train SAC
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash; \
  cd /ws && ros2 run franka_rl train_sac --level L1 --timesteps 100000"

# Unit tests (no Gazebo, no rebuild required)
docker exec franka_sim bash -c "source /opt/ros/jazzy/setup.bash; cd /ws && python3 -m pytest src/franka_rl/test -v"
```

The full command reference is in `docs/cheatsheet.md`.

## Status

Done and verified in simulation:

- Docker image, container, GPU passthrough, Gazebo GUI as an optional debug client.
- `franka_sim`: FR3 spawns in the ready pose with three active controllers
  (`joint_state_broadcaster`, arm JTC, gripper JTC), base anchored to the world.
- Pick & place scene: main table, two goal tables, cube with pose publishing.
- Gripper control: both fingers commanded explicitly (mimic joints are unsupported by DART,
  and `DetachableJoint` was tested and rejected).
- `franka_rl`: Gymnasium environment, DLS-IK, observations, reward, unit test suite.
- Scripted expert: 100/100 successful L2 episodes, `drop_rate` and `ik_failure_rate` both
  0.000; two demonstration sets recorded.
- SAC training script with resume support (checkpoint plus replay buffer).

## Planned

| Item | Notes |
|---|---|
| SAC training runs on L1/L2 | In progress; L2 with curriculum from L1 weights |
| PPO (`train_ppo.py`) | Not written yet; expected to be far less sample-efficient than SAC - that is itself a result |
| L3 disturbances | Level accepted as an argument, perturbation logic not implemented |
| `franka_diffusion` package | Dataset loader, 1D U-Net + FiLM, DDPM training / DDIM inference (T_o=2, T_p=16, T_a=8) |
| Gazebo stepping, phase 2 | Replace free-run + `/clock` timing with `multi_step` via `ControlWorld`; done when the `check_env` xfail starts passing |
| `eval.py` and evaluation protocol | >=50 test episodes x >=3 training seeds, identical scene seeds across methods |
| Smoothness metrics | Jerk integral / sum of squared joint accelerations + EE path length |
| SAC + HER | Optional additional experiment (off-policy only) |
| Ablations | Demonstration set size vs Diffusion Policy quality; T_a vs smoothness and reactivity under L3 |

Known open issues: an occasional unresponsive joint after long sessions in the same
container (fixed by `docker compose restart sim`, cause unidentified), and a possible race
between trajectory publication and world stepping once phase 2 lands.

## Documentation

- `docs/cheatsheet.md` - operational commands ("how").
- `docs/thesis-project-context.md` - design decisions and their rationale ("why").
- `docs/constants-derivations.md` - where each numeric constant comes from.
- `docs/plan-gym-wrapper-dls-ik.md` - execution plan for the Gym + DLS-IK stage.
- `CLAUDE.md` - repository working rules.

## References

1. Chi et al. (2023) - *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS.
2. Haarnoja et al. (2018) - *Soft Actor-Critic*, ICML.
3. Schulman et al. (2017) - *Proximal Policy Optimization Algorithms*.
4. Ho et al. (2020) - *Denoising Diffusion Probabilistic Models*, NeurIPS.
5. Mandlekar et al. (2021) - *What Matters in Learning from Offline Human Demonstrations*, CoRL.
6. Gallouédec et al. (2021) - *panda-gym: Open-source goal-conditioned environments for robotic learning*.
7. Andrychowicz et al. (2017) - *Hindsight Experience Replay*, NeurIPS.
8. Zhou et al. (2019) - *On the Continuity of Rotation Representations in Neural Networks*, CVPR.
9. Heo et al. (2023) - *FurnitureBench*, RSS.
10. Ren et al. (2024) - *Diffusion Policy Policy Optimization*, ICLR 2025.

## License

MIT - see [LICENSE](LICENSE).
