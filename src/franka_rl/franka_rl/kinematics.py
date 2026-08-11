import pinocchio as pin
import numpy as np
from pathlib import Path

class FrankaKinematics:
    def __init__(self, urdf_path: str, ee_frame: str = "fr3_hand_tcp"):
        """ urdf_path — path to a rendered URDF file (e.g. /tmp/fr3.urdf from xacro) """

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        self.frame_id = self.model.getFrameId(ee_frame)
        if self.frame_id == self.model.nframes:
            raise ValueError(f"Frame '{ee_frame}' does not exist in the model")
        
        arm_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
        self.arm_q_indices = []
        for name in arm_joint_names:
            jid = self.model.getJointId(name)
            idx_q = self.model.joints[jid].idx_q
            self.arm_q_indices.append(idx_q)
        
        self.arm_q_indices = np.array(self.arm_q_indices)
        
    def _full_q(self, q_arm: np.ndarray):
        """ 7 arm angels -> full vector q (nq). Fingers remain neutral. """
        q = pin.neutral(self.model)
        q[self.arm_q_indices] = q_arm
        return q
    
    def fk(self, q_arm: np.ndarray):
        """ FK: 7 angles -> (p, R) of effector in base frame
            Returns p (3), R (3, 3) """
        q = self._full_q(q_arm)
        pin.framesForwardKinematics(self.model, self.data, q)
        placement = self.data.oMf[self.frame_id]

        p = placement.translation.copy()
        R = placement.rotation.copy()

        return p, R
    
    def jacobian(self, q_arm: np.ndarray):
        """ Computes jacobian with reduced number of colums (6x9 -> 6x7) """
        q = self._full_q(q_arm)
        J_full = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = J_full[:, self.arm_q_indices]
        
        return J