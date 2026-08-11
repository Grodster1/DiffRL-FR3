import numpy as np
import pinocchio as pin
from franka_rl.inverse_kinematics import dls_step, solve_ik
from franka_rl.kinematics import FrankaKinematics

Q_READY = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

def load_kinematics():
    urdf = f"/tmp/fr3.urdf"
    return FrankaKinematics(urdf)

def make_singular_jacobian(sigma_min, seed = 0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    V, _ = np.linalg.qr(rng.standard_normal((7, 7)))

    sigmas = np.array([1.0, 1.0, 1.0, 1.0, 1.0, sigma_min])
    Sigma = np.zeros((6, 7))
    Sigma[:6, :6] = np.diag(sigmas)
    return U @ Sigma @ V.T

def test_dls_bounds_dq_near_singularity():
    J = make_singular_jacobian(sigma_min=1e-6)
    err6 = np.ones(6) * 0.1
    lam = 0.05
    bound = np.linalg.norm(err6) / (2 * lam)

    dq_dls = dls_step(J, err6, lam)
    dq_pinv = np.linalg.pinv(J) @ err6    

    # Comparison to pseudoinverion
    assert np.linalg.norm(dq_dls) < np.linalg.norm(dq_pinv)
    # Using DLS
    assert np.linalg.norm(dq_dls) < bound   

def test_convergence():
    kin = load_kinematics()
    p0, R0 = kin.fk(Q_READY)
    p_des = p0 + np.array([0.05, 0.0, 0.0])
    q_sol, info = solve_ik(kin, Q_READY, p_des, R0)
    assert info["success"]
    p_final, _ = kin.fk(q_sol)
    np.testing.assert_allclose(p_final, p_des, atol=1e-3)
    
def test_orientation_frozen():
    kin = load_kinematics()
    p0, R0 = kin.fk(Q_READY)
    p_des = p0 + np.array([0.05, 0.05, 0.0])
    q_sol, info = solve_ik(kin, Q_READY, p_des, R0)
    assert info["success"]
    _, R_final = kin.fk(q_sol)
    ang = np.linalg.norm(pin.log3(R0.T @ R_final))
    assert ang < np.deg2rad(5)
    

    