"""Tests for `gym_env.py`.

Sections 1-6 are pure: no rclpy, no Gazebo. `FrankaPickPlaceEnv.__init__` is bypassed
(`object.__new__`); section 5 swaps in `FakeSim`, so `step()` runs offline with real DLS-IK.
Section 7 needs a live simulation (`@pytest.mark.sim`) and is skipped when `/clock` is silent.
"""

import numpy as np
import pytest

import franka_rl.config as cfg
from franka_rl.gym_env import FrankaPickPlaceEnv
from franka_rl.kinematics import FrankaKinematics

URDF_PATH = "/tmp/fr3.urdf"

# Cube heights relative to the table, derived from the latch thresholds in config.py.
Z_TABLE = cfg.CUBE_POS_DEFAULT[2]
Z_ENTER = Z_TABLE + cfg.LIFT_MARGIN_ENTER + 0.005   # safely above the enter threshold
Z_BAND = Z_TABLE + 0.5 * (cfg.LIFT_MARGIN_EXIT + cfg.LIFT_MARGIN_ENTER)  # inside the hysteresis band
Z_EXIT = Z_TABLE                                     # safely below the exit threshold

GRIP_HOLD = 0.5 * (cfg.GRASP_OPENING_MIN + cfg.GRASP_OPENING_MAX)   # opening that counts as holding
GRIP_OPEN = cfg.GRIPPER_RANGE[1]                                    # fully open

NEAR = np.array([0.0, 0.0, 0.5 * cfg.GRASP_DIST_THRESH])   # EE at the cube
FAR = np.array([0.0, 0.0, 2.0 * cfg.GRASP_DIST_THRESH])    # EE away from the cube


def make_env(level="L1", goal=cfg.GOAL_LEFT, seed=0):
    """Env without `__init__`: only the fields the tested logic uses."""
    env = object.__new__(FrankaPickPlaceEnv)
    env.dt = 0.05
    env.max_episode_steps = 200
    env.level = level
    env.goal_pos = np.asarray(goal, dtype=float).copy()
    env._grasped = False
    env._grasped_streak = 0
    env._just_grasped = False
    env._ever_grasped = False
    env._success = False
    env._dropped = False
    env._min_ee_to_cube = np.inf
    env._min_cube_to_goal = np.inf
    env._drop_xy = np.full(2, np.nan)
    env._step_count = 0
    env._ik_failures = 0
    env._reset_control_period_stats()
    env.np_random = np.random.default_rng(seed)
    return env


def set_obs(env, ee_to_cube=NEAR, cube_to_goal=(0.0, 0.0, 0.0), is_grasped=False,
            gripper=GRIP_OPEN, cube_pos=None):
    """`ee_pos` is derived from the cube position so `cube_z` in `_state_cost` stays
    consistent. Defaults: cube on the table, gripper open (`s3 = s4 = 0`)."""
    ee_to_cube = np.asarray(ee_to_cube, dtype=float)
    cube_pos = np.asarray(cfg.CUBE_POS_DEFAULT if cube_pos is None else cube_pos, dtype=float)
    env._last_obs_dict = {
        "ee_pos": cube_pos - ee_to_cube,
        "ee_to_cube": ee_to_cube,
        "cube_to_goal": np.asarray(cube_to_goal, dtype=float),
        "gripper": float(gripper),
        "is_grasped": is_grasped,
    }
    return env._last_obs_dict


def latch(env, n, ee_to_cube=NEAR, z=Z_ENTER, gripper=GRIP_HOLD):
    """Calls `_update_grasped` n times with the same state; returns the last result."""
    result = None
    for _ in range(n):
        result = env._update_grasped(np.asarray(ee_to_cube, dtype=float),
                                     np.array([0.0, 0.0, z]), gripper)
    return result


# ---------------------------------------------------------------------------
# 1. Grasp latch (`_update_grasped`) - a state machine, not a per-frame check
# ---------------------------------------------------------------------------

def test_grasp_needs_all_three_conditions():
    """Closed gripper, proximity and lift - missing any one means no grasp."""
    assert latch(make_env(), cfg.N_GRASP_CONFIRM, gripper=GRIP_OPEN) is False
    assert latch(make_env(), cfg.N_GRASP_CONFIRM, ee_to_cube=FAR) is False
    assert latch(make_env(), cfg.N_GRASP_CONFIRM, z=Z_EXIT) is False


def test_grasp_requires_consecutive_confirmations():
    """A single frame is not enough (`N_GRASP_CONFIRM`)."""
    env = make_env()

    assert latch(env, cfg.N_GRASP_CONFIRM - 1) is False
    assert latch(env, 1) is True


def test_grasp_streak_resets_on_interruption():
    """An interruption resets the streak instead of pausing it."""
    env = make_env()

    latch(env, cfg.N_GRASP_CONFIRM - 1)
    latch(env, 1, gripper=GRIP_OPEN)          # one frame without holding
    assert env._grasped_streak == 0
    assert latch(env, cfg.N_GRASP_CONFIRM - 1) is False   # counts from zero again


def test_just_grasped_is_a_rising_edge():
    """`R_GRASP` is paid once; a sticky flag would reward hovering."""
    env = make_env()

    latch(env, cfg.N_GRASP_CONFIRM - 1)
    assert env._just_grasped is False

    latch(env, 1)
    assert env._just_grasped is True

    latch(env, 1)
    assert env._just_grasped is False


def test_grasp_hysteresis_band_holds_but_does_not_start():
    """Between the exit and enter thresholds the latch holds but does not engage."""
    latched = make_env()
    latch(latched, cfg.N_GRASP_CONFIRM)
    assert latch(latched, 3, z=Z_BAND) is True

    fresh = make_env()
    assert latch(fresh, 10, z=Z_BAND) is False


def test_grasp_released_below_exit_margin():
    env = make_env()
    latch(env, cfg.N_GRASP_CONFIRM)

    assert latch(env, 1, z=Z_EXIT) is False
    assert env._grasped_streak == 0


def test_grasp_released_when_gripper_opens():
    env = make_env()
    latch(env, cfg.N_GRASP_CONFIRM)

    assert latch(env, 1, gripper=GRIP_OPEN) is False


def test_ever_grasped_survives_a_release():
    """`is_grasped` in info is an episode statistic, not the current frame state."""
    env = make_env()
    latch(env, cfg.N_GRASP_CONFIRM)
    latch(env, 1, gripper=GRIP_OPEN)

    assert env._grasped is False
    assert env._ever_grasped is True


# ---------------------------------------------------------------------------
# 2. Success predicate (`_is_success`)
# ---------------------------------------------------------------------------

AT_REST = np.zeros(3)


def on_goal(env, dxy=0.0, dz=0.0):
    return env.goal_pos + np.array([dxy, 0.0, dz])


def test_success_nominal():
    env = make_env()
    assert env._is_success(on_goal(env), AT_REST, grasped=False)


def test_success_rejects_held_cube():
    """A cube held above the table is not a placement."""
    env = make_env()
    assert not env._is_success(on_goal(env), AT_REST, grasped=True)


def test_success_rejects_moving_cube():
    env = make_env()
    moving = np.array([cfg.SUCCESS_VEL_EPS * 2.0, 0.0, 0.0])
    assert not env._is_success(on_goal(env), moving, grasped=False)


def test_success_rejects_wrong_height():
    """Checking Z separately keeps a cube in flight from counting as success."""
    env = make_env()
    too_high = on_goal(env, dz=cfg.SUCCESS_Z_TOL * 2.0)
    assert not env._is_success(too_high, AT_REST, grasped=False)


def test_success_rejects_wrong_xy():
    env = make_env()
    off = on_goal(env, dxy=cfg.SUCCESS_XY_TOL * 2.0)
    assert not env._is_success(off, AT_REST, grasped=False)


def test_success_tolerances_are_strict():
    """Exactly on a threshold is not success (`<`, not `<=`)."""
    env = make_env()

    assert not env._is_success(on_goal(env, dxy=cfg.SUCCESS_XY_TOL), AT_REST, False)
    assert not env._is_success(on_goal(env, dz=cfg.SUCCESS_Z_TOL), AT_REST, False)

    vel = np.array([cfg.SUCCESS_VEL_EPS, 0.0, 0.0])
    assert not env._is_success(on_goal(env), vel, False)


def test_success_uses_the_episode_goal_not_a_constant():
    """L2 samples the goal, so the predicate must use `self.goal_pos`."""
    env = make_env(goal=cfg.GOAL_RIGHT)

    assert env._is_success(cfg.GOAL_RIGHT.copy(), AT_REST, False)
    assert not env._is_success(cfg.GOAL_LEFT.copy(), AT_REST, False)


# ---------------------------------------------------------------------------
# 3. Reward (`_compute_reward`) and state cost (`_state_cost`)
# ---------------------------------------------------------------------------

def test_state_cost_is_negative_distance_to_go():
    """The cost is minus the remaining distance plus the full `W_G` (EE outside `R_XY`)."""
    env = make_env()
    set_obs(env, ee_to_cube=[0.2, 0.0, 0.0], cube_to_goal=[0.7, 0.0, 0.0])

    assert env._state_cost() == pytest.approx(
        -(cfg.W_REACH * 0.2 + cfg.W_TRANSPORT * 0.7 + cfg.W_G))


def test_reward_is_the_cost_of_the_resulting_state():
    """Every step pays the cost of the state it ends in. That per-step price is what
    makes proximity valuable to the critic; the earlier potential-based variant had it
    cancel out and the policy learned to drive away from the cube (09.2026)."""
    env = make_env()
    set_obs(env, ee_to_cube=[0.2, 0.0, 0.0], cube_to_goal=[0.7, 0.0, 0.0])

    reward = env._compute_reward(np.zeros(4), just_grasped=False, success=False)

    assert reward == pytest.approx(env._state_cost())


def test_state_cost_drops_reach_term_once_grasped():
    """Once grasped, reach and grasp progress are no longer remaining cost."""
    env = make_env()

    set_obs(env, ee_to_cube=[0.2, 0.0, 0.0], cube_to_goal=[0.7, 0.0, 0.0])
    free = env._state_cost()

    set_obs(env, ee_to_cube=[0.2, 0.0, 0.0], cube_to_goal=[0.7, 0.0, 0.0], is_grasped=True)
    held = env._state_cost()

    assert held == pytest.approx(-cfg.W_TRANSPORT * 0.7)
    assert held - free == pytest.approx(cfg.W_REACH * 0.2 + cfg.W_G)


# States along the expert sequence (EE above the cube centre), each reaching one more
# stage of the chain: `(name, ee_to_cube, gripper, cube_lift)`.
Z_T = cfg.CUBE_POS_DEFAULT[2]
GRIP_ON_CUBE = 0.5 * (cfg.GRASP_OPENING_MIN + cfg.GRASP_OPENING_MAX)
GRASP_STAGES = [
    ("far",                 [0.20, 0.0, 0.00], GRIP_OPEN,    0.0),
    ("above, high",         [0.00, 0.0, -0.12], GRIP_OPEN,   0.0),
    ("at grasp height",     [0.00, 0.0, 0.00], GRIP_OPEN,    0.0),
    ("clamped",             [0.00, 0.0, 0.00], GRIP_ON_CUBE, 0.0),
    ("lifted 1 cm",         [0.00, 0.0, 0.00], GRIP_ON_CUBE, 0.5 * cfg.LIFT_MARGIN_ENTER),
    ("lifted 2 cm",         [0.00, 0.0, 0.00], GRIP_ON_CUBE, cfg.LIFT_MARGIN_ENTER),
]


def stage_phi(env, ee_to_cube, gripper, lift):
    cube = cfg.CUBE_POS_DEFAULT + np.array([0.0, 0.0, lift])
    set_obs(env, ee_to_cube=ee_to_cube, cube_to_goal=cfg.GOAL_LEFT - cube,
            gripper=gripper, cube_pos=cube)
    return env._state_cost()


def test_grasp_progress_rises_strictly_along_the_expert_sequence():
    """Every expert stage must raise `Phi`, including closing and the 2 cm lift."""
    env = make_env()
    phis = [stage_phi(env, e, g, l) for _, e, g, l in GRASP_STAGES]

    for (name_a, *_), (name_b, *_), a, b in zip(GRASP_STAGES, GRASP_STAGES[1:], phis, phis[1:]):
        assert b > a, f"{name_a} -> {name_b}: Phi {a:.4f} -> {b:.4f} does not rise"


def test_grasp_progress_skipping_a_stage_earns_nothing():
    """Stages multiply: clamping on air or while misaligned is worth nothing."""
    env = make_env()
    at_height_open = stage_phi(env, [0.0, 0.0, 0.0], GRIP_OPEN, 0.0)

    closed_on_air = stage_phi(env, [0.0, 0.0, 0.0], cfg.GRIPPER_RANGE[0], 0.0)
    assert closed_on_air == pytest.approx(at_height_open)

    misaligned = [cfg.R_XY + 0.01, 0.0, 0.0]
    open_far = stage_phi(env, misaligned, GRIP_OPEN, 0.0)
    clamped_far = stage_phi(env, misaligned, GRIP_ON_CUBE, 0.0)
    assert clamped_far == pytest.approx(open_far)


def test_grasp_progress_band_is_the_latch_band():
    """`Phi` uses the same clamp band as `_update_grasped`."""
    env = make_env()
    at = lambda g: stage_phi(env, [0.0, 0.0, 0.0], g, 0.0)
    eps = 1e-6

    assert at(cfg.GRASP_OPENING_MIN + eps) > at(cfg.GRASP_OPENING_MIN - eps)
    assert at(cfg.GRASP_OPENING_MAX - eps) > at(cfg.GRASP_OPENING_MAX + eps)


def test_state_cost_is_bounded():
    """`Phi <= 0`; its floor bounds the termination payout that `R_DROP` must outweigh."""
    env = make_env()
    rng = np.random.default_rng(3)

    for _ in range(500):
        ee_to_cube = rng.uniform(-0.4, 0.4, 3)
        cube_to_goal = rng.uniform(-0.8, 0.8, 3)
        gripper = rng.uniform(*cfg.GRIPPER_RANGE)
        lift = rng.uniform(0.0, 0.1)
        cube = cfg.CUBE_POS_DEFAULT + np.array([0.0, 0.0, lift])
        set_obs(env, ee_to_cube, cube_to_goal, gripper=gripper, cube_pos=cube)

        floor = -(cfg.W_REACH * np.linalg.norm(ee_to_cube)
                  + cfg.W_TRANSPORT * np.linalg.norm(cube_to_goal) + cfg.W_G)
        assert floor - 1e-9 <= env._state_cost() <= 0.0


def test_being_closer_to_the_cube_costs_less():
    """States nearer the cube must be cheaper, otherwise nothing pulls the policy in."""
    env = make_env()
    set_obs(env, ee_to_cube=[0.30, 0.0, 0.0], cube_to_goal=[0.7, 0.0, 0.0])
    far = env._compute_reward(np.zeros(4), just_grasped=False, success=False)

    set_obs(env, ee_to_cube=[0.10, 0.0, 0.0], cube_to_goal=[0.7, 0.0, 0.0])
    near = env._compute_reward(np.zeros(4), just_grasped=False, success=False)

    assert near > far


def test_entering_grasp_cannot_lower_reward():
    """Grasping must never be locally unprofitable: holding switches off the reach and
    grasp-progress terms, so the state can only get cheaper."""
    env = make_env()
    rng = np.random.default_rng(7)

    for _ in range(200):
        ee_to_cube = rng.uniform(-0.4, 0.4, 3)
        cube_to_goal = rng.uniform(-0.8, 0.8, 3)
        action = rng.uniform(-1.0, 1.0, 4)

        set_obs(env, ee_to_cube, cube_to_goal, is_grasped=False)
        free = env._compute_reward(action, just_grasped=False, success=False)

        set_obs(env, ee_to_cube, cube_to_goal, is_grasped=True)
        held = env._compute_reward(action, just_grasped=True, success=False)

        assert held >= free


def test_grasp_bonus_paid_once_per_episode():
    """`R_GRASP` rewards discovering the grasp, not repeating it. Paid per latch edge it
    would be farmable: the latch can be re-armed every few steps by bobbing the cube."""
    env = make_env()
    set_obs(env)

    steady = env._compute_reward(np.zeros(4), just_grasped=False, success=False)
    edge = env._compute_reward(np.zeros(4), just_grasped=True, success=False)

    assert edge - steady == pytest.approx(cfg.R_GRASP)


def test_success_bonus_is_added():
    env = make_env()
    set_obs(env)

    plain = env._compute_reward(np.zeros(4), just_grasped=False, success=False)
    won = env._compute_reward(np.zeros(4), just_grasped=False, success=True)

    assert won - plain == pytest.approx(cfg.R_SUCCESS)


def test_energy_penalty_scales_with_squared_action():
    env = make_env()
    set_obs(env)

    idle = env._compute_reward(np.zeros(4), just_grasped=False, success=False)
    full = env._compute_reward(np.ones(4), just_grasped=False, success=False)

    assert idle - full == pytest.approx(cfg.W_ENERGY * 4.0)


def test_successful_terminal_step_is_net_positive():
    """`R_SUCCESS` outweighs the costs of the final step."""
    env = make_env()
    set_obs(env, ee_to_cube=[0.35, 0.0, 0.0], cube_to_goal=[0.0, 0.0, 0.0])

    reward = env._compute_reward(np.ones(4), just_grasped=False, success=True)

    assert reward > 0.0


# ---------------------------------------------------------------------------
# 4. Scene sampling and seed determinism
# ---------------------------------------------------------------------------

def test_l1_goal_is_fixed():
    env = make_env(level="L1")
    for _ in range(20):
        np.testing.assert_allclose(env._sample_goal_cube_pos(), cfg.GOAL_LEFT)


def test_l2_goal_is_bimodal():
    """Two goal tables are a design decision; sampling only one collapses L2 into L1."""
    env = make_env(level="L2")
    goals = [env._sample_goal_cube_pos() for _ in range(200)]

    seen_left = any(np.allclose(g, cfg.GOAL_LEFT) for g in goals)
    seen_right = any(np.allclose(g, cfg.GOAL_RIGHT) for g in goals)

    assert seen_left and seen_right
    assert all(np.allclose(g, cfg.GOAL_LEFT) or np.allclose(g, cfg.GOAL_RIGHT) for g in goals)


def test_sampled_goal_is_a_copy():
    """Returning the config constant by reference would let an episode overwrite it."""
    env = make_env(level="L1")
    goal = env._sample_goal_cube_pos()
    goal += 1.0

    np.testing.assert_allclose(cfg.GOAL_LEFT, [0.0, 0.525, 0.425])


def test_sampled_cube_stays_in_spawn_zone():
    env = make_env(level="L2")

    for _ in range(500):
        pos = env._sample_cube_pos()
        assert 0.4 <= pos[0] <= 0.6
        assert -0.2 <= pos[1] <= 0.2
        assert pos[2] == pytest.approx(cfg.CUBE_POS_DEFAULT[2])


def test_sampling_is_reproducible_for_a_seed():
    """Evaluation protocol: identical scene seeds for every method."""
    a = make_env(level="L2", seed=42)
    b = make_env(level="L2", seed=42)

    for _ in range(10):
        np.testing.assert_allclose(a._sample_cube_pos(), b._sample_cube_pos())
        np.testing.assert_allclose(a._sample_goal_cube_pos(), b._sample_goal_cube_pos())


# ---------------------------------------------------------------------------
# 5. `step()` on a fake simulation - real IK, no ROS
# ---------------------------------------------------------------------------

class FakeSim:
    """Minimal `SimInterface` stand-in: returns a set state, records commands."""

    def __init__(self, q_arm=None, cube_pos=None, cube_vel=None, gripper_opening=GRIP_OPEN):
        self.state = {
            "q_arm": np.asarray(cfg.Q_READY if q_arm is None else q_arm, dtype=float),
            "dq_arm": np.zeros(7),
            "cube_pos": np.asarray(cfg.CUBE_POS_DEFAULT if cube_pos is None else cube_pos, dtype=float),
            "cube_vel": np.zeros(3) if cube_vel is None else np.asarray(cube_vel, dtype=float),
            "gripper_opening": gripper_opening,
        }
        self.arm_commands = []
        self.arm_dts = []
        self.gripper_commands = []
        self.gripper_dts = []
        self.cube_poses = []
        self.reserved = []
        self.destroyed = 0
        self.clock = 0.0
        self.overshoot = 0.0

    def get_state(self):
        return self.state

    def wait_for_state(self, timeout=10.0):
        return self.state

    def publish_arm_command(self, q, dt):
        self.arm_commands.append(np.asarray(q, dtype=float))
        self.arm_dts.append(dt)

    def publish_gripper_command(self, opening, dt):
        self.gripper_commands.append(opening)
        self.gripper_dts.append(dt)

    def destroy_node(self):
        self.destroyed += 1

    def set_cube_pose(self, cube_pos, timeout=5.0):
        self.cube_poses.append(np.asarray(cube_pos, dtype=float))

    def sim_time(self):
        return self.clock

    def reserve_t(self, dt):
        self.reserved.append(dt)
        self.clock += dt + self.overshoot

        return dt + self.overshoot


class ColdStartSim(FakeSim):
    """`get_state()` returns empty fields until `wait_for_state()` is called, as
    `SimInterface` does right after startup, before DDS delivers `/joint_states`."""

    def __init__(self, **sim_kwargs):
        super().__init__(**sim_kwargs)
        self.warm = False

    def get_state(self):
        if not self.warm:
            return {key: None for key in self.state}
        return self.state

    def wait_for_state(self, timeout=10.0):
        self.warm = True
        return self.state


@pytest.fixture(scope="module")
def kin():
    return FrankaKinematics(URDF_PATH)


def make_stepping_env(kin, sim_cls=FakeSim, **sim_kwargs):
    env = make_env()
    env.kin = kin
    env.sim_interface = sim_cls(**sim_kwargs)
    _, env.R_frozen = kin.fk(cfg.Q_READY)
    set_obs(env)
    return env


def test_step_returns_the_gym_five_tuple(kin):
    env = make_stepping_env(kin)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4))

    assert obs.shape == (cfg.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert set(info) == {"cube_to_goal", "ik_failures", "is_success", "is_grasped",
                         "dropped", "min_ee_to_cube", "sim_dt_mean", "sim_dt_max",
                         "min_cube_to_goal", "drop_x", "drop_y"}


def test_step_commands_the_arm_towards_the_requested_delta(kin, monkeypatch):
    """A unit Z action moves the TCP by `ACTION_SCALE` (scaling -> IK -> command,
    including the sign in `pose_error`). The joint rate limit is lifted here: at
    `MAX_JOINT_VEL` a full action needs ~0.16 rad of joint travel and would be cut
    to less than half - that cut has its own test."""
    monkeypatch.setattr(cfg, "MAX_JOINT_VEL", 1e3)
    env = make_stepping_env(kin)
    p_before, _ = kin.fk(cfg.Q_READY)

    env.step(np.array([0.0, 0.0, 1.0, 0.0]))

    q_cmd = env.sim_interface.arm_commands[-1]
    p_after, R_after = kin.fk(q_cmd)

    np.testing.assert_allclose(p_after, p_before + [0.0, 0.0, cfg.ACTION_SCALE], atol=1e-3)
    # orientation stays frozen
    assert np.degrees(np.arccos(np.clip(R_after[:, 2] @ np.array([0.0, 0.0, -1.0]), -1, 1))) < 1.0


def test_step_limits_the_joint_move_without_changing_its_direction(kin, monkeypatch):
    """Scaling the whole joint step, not clipping per joint: the direction IK chose has
    to survive, otherwise the frozen gripper orientation drifts."""
    monkeypatch.setattr(cfg, "MAX_JOINT_VEL", 1e3)
    free = make_stepping_env(kin)
    free.step(np.array([1.0, 0.0, 1.0, 0.0]))
    dq_free = free.sim_interface.arm_commands[-1] - cfg.Q_READY

    monkeypatch.setattr(cfg, "MAX_JOINT_VEL", 0.2)
    limited = make_stepping_env(kin)
    limited.step(np.array([1.0, 0.0, 1.0, 0.0]))
    dq = limited.sim_interface.arm_commands[-1] - cfg.Q_READY

    assert np.abs(dq).max() == pytest.approx(0.2 * limited.dt)
    assert np.abs(dq_free).max() > 0.2 * limited.dt          # the limit really bit
    scale = dq[np.argmax(np.abs(dq_free))] / dq_free[np.argmax(np.abs(dq_free))]
    np.testing.assert_allclose(dq, dq_free * scale, atol=1e-9)


def test_step_clips_target_to_workspace(kin):
    """Repeated max upward actions cannot leave the workspace box."""
    env = make_stepping_env(kin)

    for _ in range(40):
        env.step(np.array([0.0, 0.0, 1.0, 0.0]))
        env.sim_interface.state["q_arm"] = env.sim_interface.arm_commands[-1]

    p, _ = kin.fk(env.sim_interface.arm_commands[-1])
    assert p[2] <= cfg.WORKSPACE_BOX[1][2] + 1e-3


def test_gripper_command_is_binary(kin):
    """A threshold instead of a continuous value, so the policy cannot flutter the gripper."""
    env = make_stepping_env(kin)

    for a3 in (1.0, 0.3, 0.001, -0.001, -0.3, -1.0):
        env.step(np.array([0.0, 0.0, 0.0, a3]))

    assert set(env.sim_interface.gripper_commands) == set(cfg.GRIPPER_RANGE)
    assert env.sim_interface.gripper_commands[:3] == [cfg.GRIPPER_RANGE[1]] * 3
    assert env.sim_interface.gripper_commands[3:] == [cfg.GRIPPER_RANGE[0]] * 3


def test_step_advances_simulation_exactly_once(kin):
    env = make_stepping_env(kin)
    env.step(np.zeros(4))

    assert env.sim_interface.reserved == [env.dt]


def test_episode_truncates_at_step_limit(kin):
    env = make_stepping_env(kin)
    env.max_episode_steps = 3

    flags = [env.step(np.zeros(4))[3] for _ in range(3)]

    assert flags == [False, False, True]


def test_episode_terminates_on_success(kin):
    env = make_stepping_env(kin, cube_pos=cfg.GOAL_LEFT)
    _, reward, terminated, _, _ = env.step(np.zeros(4))

    assert terminated is True
    assert reward > 0.0


def test_a_drop_does_not_end_the_episode(kin):
    """Regression (09.2026): with the dense cost, ending on a drop cut the running cost
    and the policy learned to shove the cube off the table (56% `dropped`). The episode
    must run on, so a drop keeps paying."""
    env = make_stepping_env(kin, cube_pos=[0.16, 0.0, cfg.CUBE_DROP_Z - 0.01])

    _, r1, terminated, truncated, info = env.step(np.zeros(4))
    _, r2, terminated2, _, _ = env.step(np.zeros(4))

    assert (terminated, truncated, terminated2) == (False, False, False)
    assert info["dropped"] is True
    assert r1 < 0.0 and r2 < 0.0


def test_drop_position_records_the_first_drop(kin):
    """Where the cube left the gripper, not where it came to rest: the cube keeps
    rolling on the floor for the rest of the episode."""
    env = make_stepping_env(kin, cube_pos=[0.16, 0.37, cfg.CUBE_DROP_Z - 0.01])

    _, _, _, _, first = env.step(np.zeros(4))
    env.sim_interface.state["cube_pos"] = np.array([0.9, -0.5, 0.05])
    _, _, _, _, later = env.step(np.zeros(4))

    assert (first["drop_x"], first["drop_y"]) == pytest.approx((0.16, 0.37))
    assert (later["drop_x"], later["drop_y"]) == pytest.approx((0.16, 0.37))


def test_drop_position_is_nan_without_a_drop(kin):
    """Monitor reads the key on every episode, so it has to exist even when nothing fell."""
    env = make_stepping_env(kin)

    _, _, _, _, info = env.step(np.zeros(4))

    assert np.isnan(info["drop_x"]) and np.isnan(info["drop_y"])


def test_info_reports_the_success_verdict(kin):
    env = make_stepping_env(kin, cube_pos=cfg.GOAL_LEFT)
    _, _, _, _, info = env.step(np.zeros(4))

    assert info["is_success"] is True
    assert info["dropped"] is False


def test_info_reports_a_dropped_cube(kin):
    env = make_stepping_env(kin, cube_pos=[0.16, 0.0, cfg.CUBE_DROP_Z - 0.01])
    _, _, _, _, info = env.step(np.zeros(4))

    assert info["dropped"] is True
    assert info["is_success"] is False


def test_min_ee_to_cube_never_grows(kin):
    """Episode minimum, not the last frame - it shows whether the policy reaches the cube at all."""
    env = make_stepping_env(kin)
    minima = []
    for _ in range(5):
        _, _, _, _, info = env.step(np.array([0.0, 0.0, -1.0, 0.0]))
        minima.append(info["min_ee_to_cube"])

    assert minima == sorted(minima, reverse=True)
    assert minima[-1] <= np.linalg.norm(env._last_obs_dict["ee_to_cube"]) + 1e-9


def test_min_cube_to_goal_tracks_the_transport_frontier(kin):
    """Between the first grasp and the first success this is the only signal that the
    policy carries the cube anywhere: the episode minimum, seeded by `reset()`."""
    env = make_stepping_env(kin)
    _, _, _, _, start = env.step(np.zeros(4))

    env.sim_interface.state["cube_pos"] = cfg.GOAL_LEFT + np.array([0.1, 0.0, 0.0])
    _, _, _, _, closer = env.step(np.zeros(4))

    env.sim_interface.state["cube_pos"] = cfg.CUBE_POS_DEFAULT.copy()
    _, _, _, _, back = env.step(np.zeros(4))

    assert start["min_cube_to_goal"] == pytest.approx(
        np.linalg.norm(cfg.GOAL_LEFT - cfg.CUBE_POS_DEFAULT))
    assert closer["min_cube_to_goal"] == pytest.approx(0.1)
    assert back["min_cube_to_goal"] == pytest.approx(0.1)     # the minimum, not the current value


def test_ik_succeeds_inside_the_reachable_workspace(kin):
    """Steps from ready into the workspace must not fail IK."""
    env = make_stepping_env(kin)

    for _ in range(10):
        _, _, _, _, info = env.step(np.array([0.0, 0.0, -0.2, 0.0]))
        env.sim_interface.state["q_arm"] = env.sim_interface.arm_commands[-1]

    assert info["ik_failures"] == 0


def test_ik_failures_are_counted_at_the_singular_corner(kin):
    """The far-high corner of `WORKSPACE_BOX` is singular; `ik_failures` must count it."""
    env = make_stepping_env(kin)

    for _ in range(30):
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 1.0, 0.0]))
        env.sim_interface.state["q_arm"] = env.sim_interface.arm_commands[-1]

    assert info["ik_failures"] > 0
    assert info["ik_failures"] == env._ik_failures


def test_reset_moves_the_arm_before_placing_the_cube(kin):
    """A cube placed inside the finger geometry gets ejected by the contact solver."""
    env = make_stepping_env(kin)
    env.step(np.zeros(4))
    env._ik_failures = 5
    env._grasped = True

    order = []
    sim = env.sim_interface
    sim.publish_arm_command = lambda q, dt: order.append("arm")
    sim.set_cube_pose = lambda pos, timeout=5.0: order.append("cube")

    obs, info = env.reset(seed=1)

    assert order == ["arm", "cube"]
    assert env._step_count == 0
    assert env._ik_failures == 0
    assert env._grasped is False and env._just_grasped is False
    assert obs.shape == (cfg.OBS_DIM,)
    assert info["ik_failures"] == 0


def test_reset_opens_the_gripper_before_moving_the_arm(kin):
    """JTC holds the last finger command, so reset must open the gripper first."""
    env = make_stepping_env(kin)
    env.step(np.array([0.0, 0.0, 0.0, -1.0]))    # close the gripper
    assert env.sim_interface.gripper_commands[-1] == cfg.GRIPPER_RANGE[0]

    order = []
    sim = env.sim_interface
    sim.publish_gripper_command = lambda opening, dt: order.append(("gripper", opening))
    sim.publish_arm_command = lambda q, dt: order.append(("arm", None))

    env.reset(seed=1)

    assert order[0] == ("gripper", cfg.GRIPPER_RANGE[1])
    assert order[1][0] == "arm"


def test_reset_homes_the_arm_slower_than_a_policy_step(kin):
    """Homing crosses the whole workspace; `time_from_start = dt` would jerk JTC."""
    env = make_stepping_env(kin)
    env.reset(seed=1)

    assert env.sim_interface.arm_dts[-1] == cfg.RESET_DT
    assert env.sim_interface.gripper_dts[-1] == cfg.RESET_DT
    assert cfg.RESET_DT > env.dt


def test_reset_fails_loudly_when_the_arm_cannot_reach_ready(kin):
    """Regression (12.09.2026): a joint stuck on its limit must fail reset before the
    scene is touched, instead of silently starting a broken episode."""
    q_stuck = cfg.Q_READY.copy()
    q_stuck[1] = -1.836
    env = make_stepping_env(kin, q_arm=q_stuck)

    with pytest.raises(RuntimeError, match="fr3_joint2"):
        env.reset(seed=1)

    assert len(env.sim_interface.reserved) == cfg.N_SETTLE   # arm got the full settle time
    assert env.sim_interface.cube_poses == []                # scene untouched


def test_reset_waits_for_the_state_stream_before_reading_it(kin):
    """The first `reset()` has no `/joint_states` yet; it must block on `wait_for_state`."""
    env = make_env()
    env.kin = kin
    env.sim_interface = ColdStartSim()
    _, env.R_frozen = kin.fk(cfg.Q_READY)

    obs, info = env.reset(seed=1)

    assert obs.shape == (cfg.OBS_DIM,)


def test_step_waits_for_the_state_stream_before_reading_it(kin):
    """`step()` reads state the same way as `reset()`, regardless of call order."""
    env = make_stepping_env(kin, sim_cls=ColdStartSim)

    obs, reward, terminated, truncated, info = env.step(np.zeros(4))

    assert obs.shape == (cfg.OBS_DIM,)


def test_ik_failure_holds_the_arm_in_place(kin):
    """A failed IK is a no-op for the arm; the gripper still acts."""
    env = make_stepping_env(kin)
    q_start = env.sim_interface.state["q_arm"].copy()

    failures = 0
    for _ in range(30):
        _, _, _, _, info = env.step(np.array([1.0, 0.0, 1.0, -1.0]))
        if info["ik_failures"] > failures:
            failures = info["ik_failures"]
            np.testing.assert_allclose(env.sim_interface.arm_commands[-1],
                                       env.sim_interface.state["q_arm"])
            assert env.sim_interface.gripper_commands[-1] == cfg.GRIPPER_RANGE[0]
        env.sim_interface.state["q_arm"] = env.sim_interface.arm_commands[-1]

    assert failures > 0, "the test never reached the singular corner - adjust the action"
    assert not np.allclose(env.sim_interface.state["q_arm"], q_start), "the arm never moved"


def test_close_is_idempotent(kin):
    """`close()` is called from `__del__` and by SB3; calling it twice must be safe."""
    env = make_stepping_env(kin)
    env._owns_rclpy = False          # rclpy context was not created here, so do not shut it down
    sim = env.sim_interface

    env.close()
    env.close()

    assert sim.destroyed == 1
    assert env.sim_interface is None


def test_reset_seed_determines_the_scene(kin):
    """`reset(seed=42)` twice gives the same scene."""
    a, b = make_stepping_env(kin), make_stepping_env(kin)
    a.level = b.level = "L2"

    a.reset(seed=42)
    b.reset(seed=42)

    np.testing.assert_allclose(a.sim_interface.cube_poses[-1], b.sim_interface.cube_poses[-1])
    np.testing.assert_allclose(a.goal_pos, b.goal_pos)


# ---------------------------------------------------------------------------
# 6. Real control period (`sim_dt_*` in `info`)
# ---------------------------------------------------------------------------
#
# `reserve_t(dt)` is only a lower bound, so with `real_time_factor=0` the real period
# grows with compute between steps. `info` must report that, not the nominal `dt`.

def test_control_period_matches_dt_without_overshoot(kin):
    env = make_stepping_env(kin)
    env.reset()
    for _ in range(4):
        env.step(np.zeros(4))

    assert env._get_info()["sim_dt_mean"] == pytest.approx(env.dt)
    assert env._get_info()["sim_dt_max"] == pytest.approx(env.dt)


def test_control_period_reports_overshoot(kin):
    """Sim time burned between steps counts, since JTC holds the last command meanwhile."""
    env = make_stepping_env(kin)
    env.reset()
    env.sim_interface.overshoot = 0.03
    for _ in range(3):
        env.step(np.zeros(4))

    info = env._get_info()

    assert info["sim_dt_mean"] == pytest.approx(env.dt + 0.03)
    assert info["sim_dt_max"] == pytest.approx(env.dt + 0.03)


def test_control_period_max_catches_a_single_stall(kin):
    """One stall barely moves the mean; `sim_dt_max` is there to catch it."""
    env = make_stepping_env(kin)
    env.reset()
    for i in range(4):
        env.sim_interface.overshoot = 0.5 if i == 2 else 0.0
        env.step(np.zeros(4))

    info = env._get_info()

    assert info["sim_dt_max"] == pytest.approx(env.dt + 0.5)
    assert info["sim_dt_mean"] < info["sim_dt_max"]


def test_control_period_resets_between_episodes(kin):
    """Per-episode statistic; leaking across episodes would corrupt monitor.csv."""
    env = make_stepping_env(kin)
    env.reset()
    env.sim_interface.overshoot = 0.5
    env.step(np.zeros(4))
    env.step(np.zeros(4))

    env.sim_interface.overshoot = 0.0
    env.reset()
    env.step(np.zeros(4))
    env.step(np.zeros(4))

    assert env._get_info()["sim_dt_max"] == pytest.approx(env.dt)


def test_control_period_defaults_to_dt_before_the_second_step(kin):
    """The first step after reset has no predecessor; `info` must not divide by zero."""
    env = make_stepping_env(kin)
    env.reset()

    assert env._get_info()["sim_dt_mean"] == pytest.approx(env.dt)

    env.step(np.zeros(4))

    assert env._get_info()["sim_dt_mean"] == pytest.approx(env.dt)


# ---------------------------------------------------------------------------
# 7. Live simulation - `@pytest.mark.sim`, skipped without a running bringup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_env():
    env = FrankaPickPlaceEnv(dt=0.05, max_episode_steps=50, level="L1")
    yield env
    import rclpy
    if rclpy.ok():
        rclpy.shutdown()


@pytest.mark.sim
def test_live_state_is_complete_and_physical(live_env):
    """The bridges must deliver every field `_get_obs` uses."""
    state = live_env.sim_interface.wait_for_state(timeout=10.0)

    assert set(state) == {"gripper_opening", "q_arm", "dq_arm", "cube_pos", "cube_vel"}
    assert state["q_arm"].shape == (7,)
    assert state["cube_pos"][2] > 0.3


@pytest.mark.sim
def test_live_reserve_t_runs_on_sim_time(live_env):
    """`use_sim_time=True` keeps the env step tied to sim time, not wall time."""
    sim = live_env.sim_interface
    sim.wait_for_state(timeout=10.0)
    assert sim.get_parameter("use_sim_time").value is True

    # `wait_for_state` does not cover `_sim_time`; the first `reserve_t` waits for a tick.
    sim.reserve_t(0.05)
    assert sim._sim_time is not None

    before = sim._sim_time
    sim.reserve_t(0.2)

    assert sim._sim_time - before >= 0.2


@pytest.mark.sim
def test_live_reset_actually_moves_the_cube(live_env):
    """`SetEntityPose` reporting success without moving the cube would fake L2 randomisation."""
    live_env.level = "L2"
    cubes = []
    try:
        for seed in (1, 2, 3):
            live_env.reset(seed=seed)
            cubes.append(live_env.sim_interface._cube_pos.copy())
    finally:
        live_env.level = "L1"

    spread = max(np.linalg.norm(a - b) for a in cubes for b in cubes)
    assert spread > 0.01


@pytest.mark.sim
def test_live_env_honours_its_declared_spaces(live_env):
    """What the env declares must match what it returns, or SB3 fails mid-training."""
    obs, info = live_env.reset(seed=0)

    assert live_env.observation_space.contains(obs)
    assert isinstance(info, dict)

    action = live_env.action_space.sample()
    obs, reward, terminated, truncated, info = live_env.step(action)

    assert live_env.observation_space.contains(obs)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


@pytest.mark.sim
@pytest.mark.xfail(
    reason="phase 1 is free-run: `check_env` needs a bit-identical step after "
           "`reset(seed)`, but the sim clock runs independently of the Gym loop. "
           "Passing means phase 2 (`multi_step`) is done.",
    strict=False,
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_live_env_passes_full_gym_checker(live_env):
    from gymnasium.utils.env_checker import check_env

    check_env(live_env, skip_render_check=True)


@pytest.mark.sim
def test_live_random_policy_survives_full_episodes(live_env):
    """Episodes run without exceptions and `ik_failures` stay below 5%."""
    rng = np.random.default_rng(0)
    steps = 0

    for episode in range(2):
        live_env.reset(seed=episode)
        for _ in range(live_env.max_episode_steps):
            _, _, terminated, truncated, info = live_env.step(rng.uniform(-1.0, 1.0, 4))
            steps += 1
            if terminated or truncated:
                break

    assert info["ik_failures"] / steps < 0.05


@pytest.mark.sim
def test_live_arm_follows_the_commanded_target(live_env):
    """Closes the loop command -> JTC -> gz_ros2_control -> `/joint_states`."""
    live_env.reset(seed=0)

    for _ in range(20):
        live_env.step(np.array([0.0, 0.0, -1.0, 1.0]))

    q_cmd = live_env.sim_interface._q_arm
    p_ee, _ = live_env.kin.fk(q_cmd)

    assert p_ee[2] < cfg.WORKSPACE_BOX[1][2]
    assert np.all(np.isfinite(q_cmd))


@pytest.mark.sim
def test_live_reset_is_repeatable(live_env):
    """Free-run gives repeatability only within tolerance; beyond it argues for `multi_step`."""
    obs_a, _ = live_env.reset(seed=42)
    obs_b, _ = live_env.reset(seed=42)

    np.testing.assert_allclose(obs_a, obs_b, atol=2e-2)
