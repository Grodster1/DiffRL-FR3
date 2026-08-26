import numpy as np
from franka_rl.config import WORKSPACE_BOX, GRIPPER_RANGE, REL_SCALE

def build_obs_dict(q_arm, gripper_opening, cube_pos, goal_pos, kin):
    """ Raw observation in SI units - using to compute reward """
    p_ee, R_ee = kin.fk(np.asarray(q_arm, dtype=float))
    
    return {
            "ee_pos": p_ee,
            "ee_rot6d": R_ee[:, :2].T.reshape(6),   
            "gripper": float(gripper_opening),
            "ee_to_cube": np.asarray(cube_pos, dtype=float) - p_ee,
            "cube_to_goal": np.asarray(goal_pos, dtype=float) - np.asarray(cube_pos, dtype=float),
        }
    
def _to_unit(x, lower, upper):
    """ Maps [lower, upper] -> [-1, 1] """
    return np.clip(2.0 * (x - lower) / (upper - lower) - 1.0, -1.0, 1.0)

def normalize(obs_dict):
    """ obs_dict -> np.float32 in (OBS_DIM) [-1, 1] """
    lower, upper = WORKSPACE_BOX
    gr_min, gr_max = GRIPPER_RANGE
    
    ee_pos = _to_unit(np.asarray(obs_dict["ee_pos"], dtype=float), lower, upper)
    ee_rot6d = np.clip(np.asarray(obs_dict["ee_rot6d"], dtype=float), -1.0, 1.0)
    gripper = _to_unit(np.asarray(obs_dict["gripper"], dtype=float), gr_min, gr_max)
    ee_to_cube = np.clip(np.asarray(obs_dict["ee_to_cube"], dtype=float) / REL_SCALE, -1.0, 1.0)
    cube_to_goal = np.clip(np.asarray(obs_dict["cube_to_goal"], dtype=float) / REL_SCALE, -1.0, 1.0)

    obs = np.concatenate([ee_pos, ee_rot6d, [gripper], ee_to_cube, cube_to_goal])

    return obs.astype(np.float32)