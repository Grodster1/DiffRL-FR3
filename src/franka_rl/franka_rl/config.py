import numpy as np

# Pose
Q_READY = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

# Scene
CUBE_POS_DEFAULT = np.array([0.5, 0.0, 0.425])
GOAL_LEFT = np.array([0.0, 0.525, 0.425])
GOAL_RIGHT = np.array([0.0, -0.525, 0.425])
WORKSPACE_BOX = np.array([[-0.15, -0.6, 0.42], [0.70, 0.6, 0.65]])

# Reset
TOL = 0.05
N_SETTLE = 60
N_CUBE_SETTLE = 15

# Action
ACTION_SCALE = 0.05

# Grasp
LIFT_MARGIN_ENTER = 0.02
LIFT_MARGIN_EXIT = 0.01
GRASP_OPENING_MAX = 0.032 
GRASP_OPENING_MIN = 0.018
GRASP_DIST_THRESH = 0.05
N_GRASP_CONFIRM = 2

# Success / termination
CUBE_DROP_Z = 0.35        # [W] ponizej blatu (0.40) -> kostka w szczelinie, epizod bez sensu
SUCCESS_XY_TOL = 0.05     # [W] stol 0.25, kostka 0.05 -> 0.10 to kres "cala na blacie";
SUCCESS_Z_TOL = 0.01      # [W] kostka lezaca ma srodek na 0.425; pasmo [0.415, 0.435].
SUCCESS_VEL_EPS = ...     # [P] szum solvera DART na lezacej kostce - zmierzyc

# Observation
OBS_DIM = 3 + 6 + 1 + 3 + 3 + 1  # EE_pos + ori6D + gripper + ee_to_cube + cube_to_goal + is_grasped
GRIPPER_RANGE = (0.0, 0.04)
REL_SCALE = np.array([0.8, 0.8, 0.25])

# Joints
ARM_JOINTS = [f"fr3_joint{i}" for i in range (1, 8)]
GRIPPER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]

# Reward Weights
# Wagi 1.0 -> czlony geste sa doslownie "minus odleglosc w metrach".
W_REACH = 1.0       # [P] d_reach startowo ~0.20 m (poza ready -> kostka), max ~0.35 m
W_TRANSPORT = 1.0   # [P] d_transport startowo ~0.73 m, max ~0.75 m
W_ENERGY = 0.01     # [P] ||a||^2 <= 4, wiec max 0.04/krok - dwa rzedy mniej niz R_GRASP

# Rewards
R_GRASP = 1.0       # [P] JEDNORAZOWO na zboczu False->True
R_SUCCESS = 20.0    # [P] musi przebic sume kar gestych udanego epizodu (~ -11) z zapasem
