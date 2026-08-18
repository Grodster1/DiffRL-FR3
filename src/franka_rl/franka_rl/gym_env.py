import gymnasium as gym
import numpy as np
import tempfile
import xacro
import rclpy
import franka_rl.inverse_kinematics
from franka_rl.urdf_utils import strip_finger_mimic
from franka_rl.kinematics import FrankaKinematics
from franka_rl.ros_bridge import SimInterface


OBS_DIM = 3 + 6 + 1 + 3 + 3   # EE_pos + ori6D + gripper + ee_to_cube + cube_to_goal
Q_READY = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

class FrankaPickPlaceEnv(gym.Env):
    
    def __init__(self, dt = 0.05, max_episode_steps = 200, level = 'L1'):
        xacro_file = "/ws/src/franka_sim/urdf/fr3_gazebo.urdf.xacro"
        urdf_str = strip_finger_mimic(xacro.process_file(xacro_file).toxml())
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
            f.write(urdf_str)
            urdf_path = f.name
        
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.level = level
        
        self.kin = FrankaKinematics(urdf_path)
        
        rclpy.init()
        self.sim_interface = SimInterface()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        _, self.R_frozen = self.kin.fk(Q_READY)
