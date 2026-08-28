import numpy as np
import pytest
import franka_rl.config as cfg
from franka_rl.kinematics import FrankaKinematics
from franka_rl.observations import (
    build_obs_dict,
    normalize,
)

def load_kinematics():
    return FrankaKinematics("/tmp/fr3.urdf")

def make_obs_dict(kin, q_arm=cfg.Q_READY, gripper=0.02, cube_pos=cfg.CUBE_POS_DEFAULT,
                  goal_pos=cfg.GOAL_LEFT, is_grasped=False):
    # is_grasped is not produced by build_obs_dict - the env injects it between
    # build_obs_dict() and normalize(), because the grasp predicate needs ee_to_cube.
    obs_dict = build_obs_dict(q_arm, gripper, cube_pos, goal_pos, kin)
    obs_dict["is_grasped"] = is_grasped
    return obs_dict

def test_normalize_shape_dtype_range():
    kin = load_kinematics()
    obs = normalize(make_obs_dict(kin))

    assert obs.shape == (cfg.OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

def test_build_obs_dict_keeps_si_units():
    kin = load_kinematics()
    p_ee, _ = kin.fk(cfg.Q_READY)
    obs_dict = make_obs_dict(kin)

    np.testing.assert_allclose(obs_dict["ee_pos"], p_ee, atol=1e-12)
    np.testing.assert_allclose(obs_dict["ee_to_cube"], cfg.CUBE_POS_DEFAULT - p_ee, atol=1e-12)
    np.testing.assert_allclose(obs_dict["cube_to_goal"], cfg.GOAL_LEFT - cfg.CUBE_POS_DEFAULT, atol=1e-12)
    assert obs_dict["gripper"] == pytest.approx(0.02)

def test_ee_pos_normalization_center_and_corners():
    kin = load_kinematics()
    lower, upper = cfg.WORKSPACE_BOX
    center = 0.5 * (lower + upper)

    obs_dict = make_obs_dict(kin)

    obs_dict["ee_pos"] = center
    np.testing.assert_allclose(normalize(obs_dict)[:3], np.zeros(3), atol=1e-6)

    obs_dict["ee_pos"] = lower
    np.testing.assert_allclose(normalize(obs_dict)[:3], -np.ones(3), atol=1e-6)

    obs_dict["ee_pos"] = upper
    np.testing.assert_allclose(normalize(obs_dict)[:3], np.ones(3), atol=1e-6)

def test_ee_pos_outside_workspace_is_clipped():
    kin = load_kinematics()
    obs_dict = make_obs_dict(kin)
    obs_dict["ee_pos"] = cfg.WORKSPACE_BOX[1] + 10.0

    np.testing.assert_allclose(normalize(obs_dict)[:3], np.ones(3), atol=1e-6)

def test_gripper_normalization_endpoints():
    kin = load_kinematics()
    g_min, g_max = cfg.GRIPPER_RANGE
    obs_dict = make_obs_dict(kin)

    obs_dict["gripper"] = g_min
    assert normalize(obs_dict)[9] == pytest.approx(-1.0, abs=1e-6)

    obs_dict["gripper"] = g_max
    assert normalize(obs_dict)[9] == pytest.approx(1.0, abs=1e-6)

    obs_dict["gripper"] = 0.5 * (g_min + g_max)
    assert normalize(obs_dict)[9] == pytest.approx(0.0, abs=1e-6)

def test_rot6d_is_orthonormal_pair():
    kin = load_kinematics()
    rot6d = make_obs_dict(kin)["ee_rot6d"]

    assert rot6d.shape == (6,)
    col0, col1 = rot6d[:3], rot6d[3:]
    assert np.linalg.norm(col0) == pytest.approx(1.0, abs=1e-9)
    assert np.linalg.norm(col1) == pytest.approx(1.0, abs=1e-9)
    assert col0 @ col1 == pytest.approx(0.0, abs=1e-9)

    # third column recoverable -> the pair really encodes the full rotation
    _, R = kin.fk(cfg.Q_READY)
    np.testing.assert_allclose(np.cross(col0, col1), R[:, 2], atol=1e-9)

def test_rot6d_passes_through_unnormalized():
    kin = load_kinematics()
    obs_dict = make_obs_dict(kin)

    np.testing.assert_allclose(normalize(obs_dict)[3:9], obs_dict["ee_rot6d"], atol=1e-6)

def test_relative_vectors_scaled_and_clipped():
    kin = load_kinematics()
    obs_dict = make_obs_dict(kin)

    # REL_SCALE is per-axis, so the target fractions must be scaled component-wise
    obs_dict["ee_to_cube"] = np.array([0.5, -0.5, 0.0]) * cfg.REL_SCALE
    obs_dict["cube_to_goal"] = np.array([10.0, -10.0, 10.0])

    obs = normalize(obs_dict)
    np.testing.assert_allclose(obs[10:13], [0.5, -0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs[13:16], [1.0, -1.0, 1.0], atol=1e-6)


def test_is_grasped_flag_maps_to_pm_one():
    kin = load_kinematics()

    assert normalize(make_obs_dict(kin, is_grasped=True))[16] == pytest.approx(1.0)
    assert normalize(make_obs_dict(kin, is_grasped=False))[16] == pytest.approx(-1.0)


def test_normalize_requires_is_grasped():
    """Missing flag must fail loudly - a silent default would poison the observation."""
    kin = load_kinematics()
    obs_dict = make_obs_dict(kin)
    del obs_dict["is_grasped"]

    with pytest.raises(KeyError):
        normalize(obs_dict)
