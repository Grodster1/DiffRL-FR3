import numpy as np
import pinocchio as pin
from franka_rl.kinematics import FrankaKinematics

Q_READY = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

def _numerical_jacobian(kin, q_arm, eps = 1e-6):
    J = np.zeros((6,7))
    p0, R0 = kin.fk(q_arm)
    for i in range(7):
        dq = q_arm.copy()
        dq[i] += eps
        p1, R1 = kin.fk(dq)
        
        #Translation
        J[:3, i] = (p1 - p0)/eps
        
        #Rotation — world-aligned, to match LOCAL_WORLD_ALIGNED in kin.jacobian()
        J[3:, i] = pin.log3(R1 @ R0.T)/eps
        
    return J
    

def load_kinematics():
    urdf = f"/tmp/fr3.urdf"
    return FrankaKinematics(urdf)

def test_fk_ready_position():
    kin = load_kinematics()
    p, R = kin.fk(Q_READY)
    np.testing.assert_allclose(p, [0.307, 0.0, 0.487], atol=2e-3)
    
def test_fk_ready_orientation():
    kin = load_kinematics()
    p, R = kin.fk(Q_READY)
    np.testing.assert_allclose(R[:, 2], [0.0, 0.0, -1.0], atol=1e-2)
    
def test_jacobian_numeric():
    kin = load_kinematics()
    q = Q_READY
    np.testing.assert_allclose(kin.jacobian(q), _numerical_jacobian(kin, q), atol=1e-5)
    