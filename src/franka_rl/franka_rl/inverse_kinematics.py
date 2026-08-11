import pinocchio as pin
import numpy as np

def pose_error(p_cur, R_cur, p_des, R_des):
    """ Position error as a 6D vector: [Pose error (3), Orientation error (3)].
        Input for DLS"""
    err_pos = p_des - p_cur
    err_rot = pin.log3(R_des @ R_cur.T)
    
    return np.concatenate([err_pos, err_rot])

def dls_step(J, err6, _lambda = 0.05):
    """ Computes Dumped Least Sqquares: dq = J.T @ (J@J.T + lambda^2 * I)^-1 @ err6 """
    JJt = J @ J.T
    A = JJt + _lambda**2 * np.eye(6)
    x = np.linalg.solve(A, err6)
    dq = J.T @ x
    
    return dq
    
def clip_to_joint_limits(q, model):
    """ Clipping joint position to available model's range """    
    
    return np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)

def clip_to_workspace(p_des, box):
    """ Clipping position to box workspace.
        box: np.array shape (2,3) -> [[x_min, y_min, z_min], [x_max, y_max, z_max]] """
    
    return np.clip(p_des, box[0], box[1])

def solve_ik(kin, q_init, p_des, R_des, _lambda = 0.05, tol = 1e-3, max_iters = 50, dq_max = 1.0):
    """ Iterative IK Solver.
        kin: FrankaKinematics, q_init: 7 primary angles """
    
    q = q_init.copy()
    
    for i in range(max_iters):
        p, R = kin.fk()
        err6 = pose_error(p, R, p_des, R_des)
        
        if np.linalg.norm(err6) < tol:
            return q, {"success": True, "iters": i, "final_error": np.linalg.norm(err6), "reason": "converged"}
        
        J = kin.jacobian(q)
        dq = dls_step(J, err6, _lambda)
        
        if np.linalg.norm(dq) > dq_max:
            return q, {"success": False, "iters": i, "final_error": np.linalg.norm(err6), "reason": "no-op"}
            
        q = clip_to_joint_limits(q + dq, kin.model)
        
    return q, {"success": False, "iters": max_iters, "final_error": np.linalg.norm(err6), "reason": "max_iters"}