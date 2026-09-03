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
RESET_DT = 1.5            # [W] return to Q_READY is a movement through full space not a step of P
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
CUBE_DROP_Z = 0.35        # [W] it means cube has fallen below table level
SUCCESS_XY_TOL = 0.05     # [W] table at 0.25, cube 0.05 
SUCCESS_Z_TOL = 0.01      # [W] cube laying on the table has center (Z) in 0.425; tolerated [0.415, 0.435].
SUCCESS_VEL_EPS = 0.01    # [Z]/[P] DART posts exactly 0.0 (2500 samples);

# Observation
OBS_DIM = 3 + 6 + 1 + 3 + 3 + 1  # EE_pos + ori6D + gripper + ee_to_cube + cube_to_goal + is_grasped
GRIPPER_RANGE = (0.0, 0.04)
REL_SCALE = np.array([0.8, 0.8, 0.25])

# Joints
ARM_JOINTS = [f"fr3_joint{i}" for i in range (1, 8)]
GRIPPER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]

# Reward Weights
W_REACH = 1.0       # [P] 
W_TRANSPORT = 1.0   # [P] 
W_ENERGY = 0.01     # [P] 

# Rewards
R_GRASP = 1.0       # [P] Once False->True
R_SUCCESS = 20.0    # [P] must exceed the sum of negative rewards (~ -11)
