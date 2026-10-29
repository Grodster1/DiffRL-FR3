"""Tests for `expert.py`- no Gazebo. Observations are built by hand."""

import numpy as np

import franka_task.config as cfg
from franka_task.expert import (HOVER_Z, PHASE_BUDGET, PLACE, REPLAN_XY_TOL, SWITCH_TRANSPORT_BUDGET,
                              TRANSPORT, ScriptedExpert)

CUBE = cfg.CUBE_POS_DEFAULT
HOVER = np.array([CUBE[0], CUBE[1], HOVER_Z])


def obs(ee, cube, goal=cfg.GOAL_LEFT):
    ee, cube = np.asarray(ee, dtype=float), np.asarray(cube, dtype=float)
    return {"ee_pos": ee, "ee_to_cube": cube - ee, "cube_to_goal": np.asarray(goal) - cube}


def descending_expert():
    """Expert that has just reached the hover point above CUBE."""
    expert = ScriptedExpert()
    expert.act(obs(HOVER, CUBE))
    assert expert.phase == "DESCEND"
    return expert


def test_descend_keeps_going_while_the_cube_stays():
    expert = descending_expert()
    ee = HOVER - np.array([0.0, 0.0, 0.05])
    small_drift = CUBE + np.array([0.5 * REPLAN_XY_TOL, 0.0, 0.0])

    expert.act(obs(ee, small_drift))

    assert expert.phase == "DESCEND"
    assert expert.replans == 0


def test_moved_cube_sends_descend_back_to_approach():
    """Regression (26.10.2026): DESCEND chased a displaced cube diagonally and timed
    out its budget in 23/40 L3 demos."""
    expert = descending_expert()
    ee = HOVER - np.array([0.0, 0.0, 0.03])
    moved = CUBE + np.array([0.10, 0.0, 0.0])

    action = expert.act(obs(ee, moved))

    assert expert.phase == "APPROACH"
    assert expert.replans == 1
    assert not expert.failed
    assert action[2] > 0          # climbs back towards the hover height
    assert action[0] > 0          # and heads over the new cube position
    assert action[3] > 0          # gripper stays open


def test_replan_restarts_the_approach_budget():
    expert = descending_expert()
    for _ in range(5):
        expert.act(obs(HOVER - np.array([0.0, 0.0, 0.03]), CUBE))   # spends DESCEND steps

    expert.act(obs(HOVER - np.array([0.0, 0.0, 0.03]), CUBE + np.array([0.10, 0.0, 0.0])))

    assert expert.phase == "APPROACH"
    assert expert._phase_steps == 1


def test_reset_clears_replans():
    expert = descending_expert()
    expert.act(obs(HOVER, CUBE + np.array([0.10, 0.0, 0.0])))

    expert.reset()

    assert expert.replans == 0


CARRIED = np.array([0.30, 0.20, cfg.CUBE_POS_DEFAULT[2] + 0.12])   # held cube on the way to the left goal


def carried_obs(goal, cube=CARRIED):
    return obs(cube, cube, goal)     # held: TCP sits at the cube centre


def expert_in(phase, steps=5, goal=cfg.GOAL_LEFT):
    expert = ScriptedExpert()
    expert._phase = phase
    for _ in range(steps):
        expert.act(carried_obs(goal))
    expert._phase = phase          # stay in the phase even if a step happened to reach it
    return expert


def test_goal_switch_restarts_the_transport_budget():
    expert = expert_in(TRANSPORT)

    action = expert.act(carried_obs(cfg.GOAL_RIGHT))

    assert expert.phase == "TRANSPORT"
    assert expert._phase_steps == 1
    assert expert.replans == 1
    assert action[1] < 0           # heads to the right table
    assert action[3] < 0           # keeps the gripper shut


def test_goal_switch_during_place_returns_to_transport():
    """PLACE lowers the cube onto the goal; after a switch it must be carried at hover height."""
    expert = expert_in(PLACE)

    expert.act(carried_obs(cfg.GOAL_RIGHT))

    assert expert.phase == "TRANSPORT"
    assert expert.replans == 1


def test_a_steady_goal_is_not_a_switch():
    expert = expert_in(TRANSPORT, steps=10)

    assert expert.replans == 0


def test_transport_after_a_switch_gets_the_longer_budget():
    """Regression (26.10.2026): 3/40 goal_switch demos timed out TRANSPORT exactly
    76 steps after the switch."""
    expert = expert_in(TRANSPORT)
    expert.act(carried_obs(cfg.GOAL_RIGHT))
    far = CARRIED + np.array([0.0, 0.3, 0.0])     # never reaches the goal

    for _ in range(PHASE_BUDGET[TRANSPORT] + 5):
        expert.act(carried_obs(cfg.GOAL_RIGHT, cube=far))
    assert not expert.failed

    for _ in range(SWITCH_TRANSPORT_BUDGET):
        expert.act(carried_obs(cfg.GOAL_RIGHT, cube=far))
    assert expert.failed and expert.failure_phase == "TRANSPORT"


def test_regular_transport_keeps_its_budget():
    expert = expert_in(TRANSPORT, steps=0)
    far = CARRIED + np.array([0.0, 0.3, 0.0])

    for _ in range(PHASE_BUDGET[TRANSPORT]):
        expert.act(carried_obs(cfg.GOAL_LEFT, cube=far))

    assert expert.failed


def test_reset_restores_the_transport_budget():
    expert = expert_in(TRANSPORT)
    expert.act(carried_obs(cfg.GOAL_RIGHT))

    expert.reset()

    assert expert._budget() == PHASE_BUDGET[expert._phase]
