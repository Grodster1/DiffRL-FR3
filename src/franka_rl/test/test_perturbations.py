"""Tests for `perturbations.py` - pure, no rclpy, no Gazebo."""

import numpy as np
import pytest

import franka_rl.config as cfg
from franka_rl.perturbations import sample_displaced_cube

BOX = cfg.CUBE_START_XY_BOX


def sample(rng, cube_pos, ee_pos, jitter=cfg.CUBE_START_XY_JITTER):
    return sample_displaced_cube(rng, np.asarray(cube_pos, dtype=float), np.asarray(ee_pos, dtype=float),
                                 BOX, jitter, cfg.CUBE_START_EE_CLEARANCE, cfg.CUBE_START_MAX_TRIES)


def test_displaced_cube_satisfies_every_constraint():
    """L2 spawn zone x EE anywhere in the trigger band, many seeds."""
    for seed in range(500):
        rng = np.random.default_rng(seed)
        cube = np.array([rng.uniform(0.4, 0.6), rng.uniform(-0.2, 0.2), cfg.CUBE_POS_DEFAULT[2]])
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(*cfg.CUBE_START_JITTER_TRIGGER)
        ee = cube + np.array([r * np.cos(angle), r * np.sin(angle), 0.05])

        new_pos = sample(rng, cube, ee)

        assert new_pos.shape == (3,)
        assert np.all(new_pos[:2] >= BOX[0]) and np.all(new_pos[:2] <= BOX[1])
        d = np.linalg.norm(new_pos[:2] - cube[:2])
        assert cfg.CUBE_START_XY_JITTER[0] <= d <= cfg.CUBE_START_XY_JITTER[1]
        assert np.linalg.norm(new_pos[:2] - ee[:2]) > cfg.CUBE_START_EE_CLEARANCE
        assert new_pos[2] == pytest.approx(cfg.CUBE_POS_DEFAULT[2])


def test_box_lies_on_the_main_table():
    """Main table: 0.6 x 0.8 centred at (0.5, 0); the whole 5 cm cube must stay on it."""
    half_cube = 0.025
    assert BOX[0][0] - half_cube >= 0.2 and BOX[1][0] + half_cube <= 0.8
    assert BOX[0][1] - half_cube >= -0.4 and BOX[1][1] + half_cube <= 0.4


def test_sampler_is_reproducible_for_a_seed():
    cube, ee = cfg.CUBE_POS_DEFAULT, cfg.CUBE_POS_DEFAULT + np.array([0.1, 0.0, 0.05])
    a = sample(np.random.default_rng(5), cube, ee)
    b = sample(np.random.default_rng(5), cube, ee)

    np.testing.assert_allclose(a, b)


def test_impossible_constraints_fail_loudly():
    """No point in the box is 10-20 m away; returning None would start an unperturbed L3 episode."""
    with pytest.raises(RuntimeError):
        sample(np.random.default_rng(0), cfg.CUBE_POS_DEFAULT, cfg.CUBE_POS_DEFAULT, jitter=(10.0, 20.0))
