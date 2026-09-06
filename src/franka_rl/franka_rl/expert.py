""" Scripted pick & place expert for FrankaPickPlaceEnv.

    Reference implementation with two jobs: prove the task is reachable through `env.step()`, 
    and record demonstrations for Diffusion Policy. It therefore emits actions in the same 
    as SAC and DP - (dx, dy, dz, g) in [-1, 1], routed through the same `step()`.
    Hence every demonstration is something a policy could actually reproduce.
"""

import numpy as np
import franka_rl.config as cfg


HOVER_Z = 0.55
POS_TOL = 0.01
GRIP_STEPS = 8
OPEN, SHUT = 1.0, -1.0

APPROACH, DESCEND, CLOSE, LIFT, TRANSPORT, PLACE, RELEASE, SETTLE = range(8)

PHASE_NAMES = {
    APPROACH: "APPROACH", DESCEND: "DESCEND", CLOSE: "CLOSE", LIFT: "LIFT",
    TRANSPORT: "TRANSPORT", PLACE: "PLACE", RELEASE: "RELEASE", SETTLE: "SETTLE"
}

# Steps per phase, with margin. The action space moves the EE by at most 5 cm per
# step, so the floor is distance / 0.05; the budget is that times a safety factor.
# The margin has to be generous: the control law tapers near the target and JTC
# lags, so the last centimetre costs several steps. Sum must stay under
# max_episode_steps (200), or the late phases never get their turn.

PHASE_BUDGET = {
    APPROACH: 30, DESCEND: 20, CLOSE: GRIP_STEPS + 2, LIFT:20,
    TRANSPORT: 40, PLACE: 20, RELEASE: GRIP_STEPS + 2, SETTLE: 10**9
}

class ScriptedExpert():
    """ Eight-phase state machine: approach above the cube, descend onto it, close,
        lift, transport, place, release, settle."""

    def __init__(self, hover_z = HOVER_Z, pos_tol = POS_TOL, grip_steps = GRIP_STEPS):
        self.hover_z = hover_z
        self.pos_tol = pos_tol
        self.grip_steps = grip_steps
        self.reset()
        
    def reset(self):
        self._phase = APPROACH
        self._phase_steps = 0
        self.failed = False
        self.failure_phase = None
    
    @property
    def phase(self):
        return PHASE_NAMES[self._phase]
    
    @staticmethod
    def _scene(obs_dict):
        ee = np.asarray(obs_dict["ee_pos"], dtype=float)
        cube = ee + np.asarray(obs_dict["ee_to_cube"], dtype=float)
        goal = cube + np.asarray(obs_dict["cube_to_goal"], dtype=float)
        return ee, cube, goal
    
    def _drive(self, error_xyz, g):
        delta = np.clip(np.asarray(error_xyz, dtype=float) / cfg.ACTION_SCALE, -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], g], dtype=np.float32)
    
    def _tick(self, reached):
        self._phase_steps += 1
        if reached and self._phase < SETTLE:
            self._phase += 1
            self._phase_steps = 0
        elif self._phase_steps >= PHASE_BUDGET[self._phase]:
            self.failed = True
            self.failure_phase = PHASE_NAMES[self._phase]
            
    def _phase_target(self, ee, cube, goal):
        if self._phase == APPROACH:
            error = np.array([cube[0] - ee[0],
                              cube[1] - ee[1],
                              self.hover_z - ee[2]])
            reached = np.linalg.norm(error) < self.pos_tol
            return error, OPEN, reached
        
        if self._phase == DESCEND:
            error = np.array([cube[0] - ee[0],
                              cube[1] - ee[1],
                              cube[2] - ee[2]])
            reached = np.linalg.norm(error) < self.pos_tol
            return error, OPEN, reached
        
        if self._phase == CLOSE:
            reached = self._phase_steps + 1 >= self.grip_steps
            return np.zeros(3), SHUT, reached
        
        if self._phase == LIFT:
            error = np.array([0, 0, self.hover_z - ee[2]])
            lifted = cube[2] > cfg.CUBE_POS_DEFAULT[2] + cfg.LIFT_MARGIN_ENTER
            reached = np.linalg.norm(error) < self.pos_tol and lifted
            return error, SHUT, reached
        
        if self._phase == TRANSPORT:
            error = np.array([goal[0] - cube[0],
                              goal[1] - cube[1],
                              self.hover_z - ee[2]])
            reached = np.linalg.norm(error) < self.pos_tol
            return error, SHUT, reached
        
        if self._phase == PLACE:
            error = goal - cube
            reached = np.linalg.norm(error) < self.pos_tol
            return error, SHUT, reached
        
        if self._phase == RELEASE:
            reached = self._phase_steps + 1 >= self.grip_steps
            return np.zeros(3), OPEN, reached
        
        if self._phase == SETTLE:
            return np.zeros(3), OPEN, False
        
        raise RuntimeError(f"Unhandled phase {self._phase}")
            
    def act(self, obs_dict):
        ee, cube, goal = self._scene(obs_dict)
        error, g, reached = self._phase_target(ee, cube, goal)
        action = self._drive(error, g)
        self._tick(reached)
        return action
    
    