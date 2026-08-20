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
TOL = 0.05
N_settle = 60
N_cube_settle = 15

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
    
    def _sample_cube_pos(self):
        x = self.np_random.uniform(0.4, 0.6)
        y = self.np_random.uniform(-0.2, 0.2)
        z = 0.425
        return np.array([x,y,z])
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        """ Sequence: move away arm -> spawn cube """
        
        self.sim_interface.publish_arm_command(Q_READY, self.dt)
        
        for _ in range(N_settle):
            self.sim_interface.reserve_t(self.dt)
            state = self.sim_interface.get_state()
            if np.linalg.norm(state["q_arm"] - Q_READY) < TOL:
                break
            
        if self.level == "L1":
            cube_pos = np.array([0.5, 0.0, 0.425])
        else:
            cube_pos = self._sample_cube_pos()
        
        self.sim_interface.set_cube_pose(cube_pos)
        
        for _ in range(N_cube_settle):
            self.sim_interface.reserve_t(self.dt)
            
        
        return 
        
        
