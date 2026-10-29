import numpy as np
import franka_task.config as cfg

def sample_displaced_cube(rng, cube_pos, ee_pos, box, jitter, ee_clearance, max_tries):
    for _ in range(max_tries):
        new_pos_xy = rng.uniform(box[0], box[1])
        d = np.linalg.norm(new_pos_xy - cube_pos[:2])
        d_in_bounds = jitter[0] <= d <= jitter[1]
        if d_in_bounds and np.linalg.norm(new_pos_xy - ee_pos[:2]) > ee_clearance:
            return np.append(new_pos_xy, cfg.CUBE_POS_DEFAULT[2])
    raise RuntimeError(f"Sampling displaced cube failed. New pos: {new_pos_xy}")