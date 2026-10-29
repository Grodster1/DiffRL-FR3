""" Scripted pick & place expert for FrankaPickPlaceEnv."""

import numpy as np
import franka_task.config as cfg


HOVER_Z = 0.55
POS_TOL = 0.01
GOAL_CHANGE_TOL = 0.05  # goal = cube + cube_to_goal is exact; any jump this large is a goal switch
REPLAN_XY_TOL = 0.03   # APPROACH ends within POS_TOL, so a larger XY error in DESCEND means the cube moved
GRIP_STEPS = 8
OPEN, SHUT = 1.0, -1.0

APPROACH, DESCEND, CLOSE, LIFT, TRANSPORT, PLACE, RELEASE, SETTLE = range(8)

PHASE_NAMES = {
    APPROACH: "APPROACH", DESCEND: "DESCEND", CLOSE: "CLOSE", LIFT: "LIFT",
    TRANSPORT: "TRANSPORT", PLACE: "PLACE", RELEASE: "RELEASE", SETTLE: "SETTLE"
}

PHASE_BUDGET = {
    APPROACH: 45, DESCEND: 18, CLOSE: GRIP_STEPS + 2, LIFT:18,
    TRANSPORT: 75, PLACE: 16, RELEASE: GRIP_STEPS + 2, SETTLE: 10**9
}
# The trip back after an L3 goal switch (~0.8 m, past the robot base) outlasts the longest
# regular transport; 75 timed out 3/40 episodes, each exactly 76 steps after the switch
SWITCH_TRANSPORT_BUDGET = 120

class ScriptedExpert():
    """ Eight-phase state machine: approach above the cube, descend onto it, close,
        lift, transport, place, release, settle."""

    def __init__(self, hover_z = HOVER_Z, pos_tol = POS_TOL, grip_steps = GRIP_STEPS,
                 replan_xy_tol = REPLAN_XY_TOL):
        self.hover_z = hover_z
        self.pos_tol = pos_tol
        self.grip_steps = grip_steps
        self.replan_xy_tol = replan_xy_tol
        self.reset()
        
    def reset(self):
        self._phase = APPROACH
        self._phase_steps = 0
        self.failed = False
        self.failure_phase = None
        self.replans = 0
        self._goal = None
        self._transport_budget = PHASE_BUDGET[TRANSPORT]
    
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
    
    def _budget(self):
        return self._transport_budget if self._phase == TRANSPORT else PHASE_BUDGET[self._phase]

    def _tick(self, reached):
        self._phase_steps += 1
        if reached and self._phase < SETTLE:
            self._phase += 1
            self._phase_steps = 0
        elif self._phase_steps >= self._budget():
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
            
    def _replan_if_cube_moved(self, ee, cube):
        """ DESCEND aims straight at the cube. If the cube moved (L3 displace), going
            back to APPROACH re-hovers above it and descends vertically, instead of a
            diagonal descent into the cube that times out the phase budget. """
        if self._phase == DESCEND and np.linalg.norm(cube[:2] - ee[:2]) > self.replan_xy_tol:
            self._phase = APPROACH
            self._phase_steps = 0
            self.replans += 1

    def _replan_if_goal_moved(self, goal):
        """ TRANSPORT follows the goal by itself, but its budget runs from the phase start;
            after a goal switch (L3) the trip back would time it out. PLACE goes back to
            TRANSPORT so the cube is carried at hover height, not lowered towards the old goal. """
        moved = self._goal is not None and np.linalg.norm(goal - self._goal) > GOAL_CHANGE_TOL
        self._goal = goal
        if moved and self._phase in (TRANSPORT, PLACE):
            self._phase = TRANSPORT
            self._phase_steps = 0
            self._transport_budget = SWITCH_TRANSPORT_BUDGET
            self.replans += 1

    def act(self, obs_dict):
        ee, cube, goal = self._scene(obs_dict)
        self._replan_if_cube_moved(ee, cube)
        self._replan_if_goal_moved(goal)
        error, g, reached = self._phase_target(ee, cube, goal)
        action = self._drive(error, g)
        self._tick(reached)
        return action
    
    