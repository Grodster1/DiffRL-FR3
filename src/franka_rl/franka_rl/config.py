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

# Observation
OBS_DIM = 3 + 6 + 1 + 3 + 3   # EE_pos + ori6D + gripper + ee_to_cube + cube_to_goal
GRIPPER_RANGE = (0.0, 0.04)
REL_SCALE = np.array([0.8, 0.8, 0.25])

# Joints
ARM_JOINTS = [f"fr3_joint{i}" for i in range (1, 8)]
GRIPPER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]

# Reward Weights
W_REACH = ...
W_TRANSPORT = ...
W_ENERGY = ...

# Rewards
R_GRASP = ...
R_SUCCESS = ...