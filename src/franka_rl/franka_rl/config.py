import numpy as np

# Pose
Q_READY = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

# Scene
CUBE_POS_DEFAULT = np.array([0.5, 0.0, 0.425])
GOAL_LEFT = np.array([0.0, 0.525, 0.425])
GOAL_RIGHT = np.array([0.0, -0.525, 0.425])
WORKSPACE_BOX = np.array([[-0.15, -0.6, 0.42], [0.70, 0.6, 0.65]])
CLIP_MARGIN = 0.05         # [Z] set to 0.05 rad. It prevents joints from reaching its limit. Joint2 after reaching its limit blocks itself and reset() doesn't work.
                           # joint2 driven exactly onto its limit locks in DART and ignores commands; mechanism unexplained

# Reset
TOL = 0.05
RESET_DT = 1.5             # [W] return to Q_READY is a movement through full space not a step of P
N_SETTLE = 60
N_CUBE_SETTLE = 15

# Action
ACTION_SCALE = 0.05
MAX_JOINT_VEL = 1.0        # [Z] rad/s - this value gives overshoot at level of 0.005 rad. It prevents reaching joint's limit

# Grasp
LIFT_MARGIN_ENTER = 0.02
LIFT_MARGIN_EXIT = 0.01
GRASP_OPENING_MAX = 0.064  # [Z] cube side 0.05 +28%; measured grasp gives 0.040+0.010=0.050
GRASP_OPENING_MIN = 0.036  # [Z] excludes both fully open (0.08) and closed on air (0.0)
GRASP_DIST_THRESH = 0.05
N_GRASP_CONFIRM = 2

# Potential stages
R_XY = GRASP_DIST_THRESH
R_Z = 0.05

# Success / termination
CUBE_DROP_Z = 0.35        # [W] it means cube has fallen below table level
SUCCESS_XY_TOL = 0.05     # [W] table at 0.25, cube 0.05 
SUCCESS_Z_TOL = 0.01      # [W] cube laying on the table has center (Z) in 0.425; tolerated [0.415, 0.435].
SUCCESS_VEL_EPS = 0.01    # [Z]/[P] DART posts exactly 0.0 (2500 samples);

# Observation
OBS_DIM = 3 + 6 + 1 + 3 + 3 + 1  # EE_pos + ori6D + gripper + ee_to_cube + cube_to_goal + is_grasped
GRIPPER_RANGE = (0.0, 0.08)      # [W] total aperture: both fingers, each 0.0-0.04
REL_SCALE = np.array([0.8, 0.8, 0.25])

# Joints
ARM_JOINTS = [f"fr3_joint{i}" for i in range (1, 8)]
GRIPPER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]

# Reward Weights
W_REACH = 1.0       # [P] 
W_TRANSPORT = 1.0   # [P] 
W_ENERGY = 0.01     # [P] 
W_G = 0.5

# Rewards
R_GRASP = 1.0       # [P] Once False->True
R_SUCCESS = 20.0    # [P] dominates every other term; under PBRS the shaping sum is bounded by |Phi|
R_DROP = -2.0       # [P] must outweigh the shaping paid out by ending early (|Phi(s0)| /approx 0.93 on L1)

# Training
GAMMA = 0.99
