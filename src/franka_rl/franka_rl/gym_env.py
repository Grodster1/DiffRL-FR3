import gymnasium as gym
import numpy as np
import tempfile
import xacro
import rclpy
import franka_rl.inverse_kinematics as ik
import franka_rl.observations as obs
import franka_rl.config as cfg
from franka_rl.urdf_utils import strip_finger_mimic
from franka_rl.kinematics import FrankaKinematics
from franka_rl.ros_bridge import SimInterface

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
        self._step_count = 0
        
        self.kin = FrankaKinematics(urdf_path)
        
        rclpy.init()
        self.sim_interface = SimInterface()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (cfg.OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32) # [dx, dy, dz, g]
        _, self.R_frozen = self.kin.fk(cfg.Q_READY)
    
    def _sample_cube_pos(self):
        x = self.np_random.uniform(0.4, 0.6)
        y = self.np_random.uniform(-0.2, 0.2)
        z = 0.425
        return np.array([x,y,z])
    
    def _sample_goal_cube_pos(self):
        """L1: static target (left table). L2/L3: random left or right."""
        if self.level == "L1":
            return cfg.GOAL_LEFT.copy()
        return (cfg.GOAL_LEFT if self.np_random.random() < 0.5 else cfg.GOAL_RIGHT).copy()

    def _get_obs(self):
        state = self.sim_interface.wait_for_state()
        self._last_obs_dict = obs.build_obs_dict(
            state["q_arm"], state["gripper_opening"], state["cube_pos"], self.goal_pos, self.kin 
        )
        return obs.normalize(self._last_obs_dict)
    
    def _get_info(self):
        return self._last_obs_dict["cube_to_goal"]
    
    def compute_reward(self, obs, action, grasped, success):
        if not grasped:
            reward = -obs["ee_to_cube"] * cfg.W_REACH
        else:
            reward = cfg.R_GRASP
            reward -= obs["cube_to_goal"] * cfg.W_TRANSPORT
        
        if success:
            reward += cfg.R_SUCCESS
            
        reward -= cfg.W_ENERGY * np.sum(action**2)
        
        return reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        """ Sequence: move away arm -> spawn cube """
        
        self.sim_interface.publish_arm_command(cfg.Q_READY, self.dt)
        self._step_count = 0
        
        for _ in range(cfg.N_SETTLE):
            self.sim_interface.reserve_t(self.dt)
            state = self.sim_interface.get_state()
            if np.linalg.norm(state["q_arm"] - cfg.Q_READY) < cfg.TOL:
                break
            
        if self.level == "L1":
            cube_pos = cfg.CUBE_POS_DEFAULT.copy()
        else:
            cube_pos = self._sample_cube_pos()
        
        self.sim_interface.set_cube_pose(cube_pos)
        self.goal_pos = self._sample_goal_cube_pos()

        for _ in range(cfg.N_CUBE_SETTLE):
            self.sim_interface.reserve_t(self.dt)
            
        observation = self._get_obs()
        info = self._get_info()
            
        return observation, info
    
    def step(self, action):
        action = np.asarray(action, dtype=float)
        state = self.sim_interface.get_state()
        q_current = state["q_arm"]
        
        p_ee_current, _ = self.kin.fk(q_current) # or self._last_obs_dict["ee_pos"]
        delta = action[:3] * cfg.ACTION_SCALE
        p_des = p_des + delta
        p_des= ik.clip_to_workspace(p_des)
        
        q_target, ik_output = ik.solve_ik(self.kin, q_current, p_des, self.R_frozen)
        opening = cfg.GRIPPER_RANGE[1] if action[3] > 0 else cfg.GRIPPER_RANGE[0] 
        
        self.sim_interface.publish_arm_command(q_target, self.dt)
        self.sim_interface.publish_gripper_command(opening, self.dt)
        self.sim_interface.reserve_t(self.dt)
        
        obs = self._get_obs()
        
        
        
        
        
        
        
        
        
        
