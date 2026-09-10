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
        self._grasped = False
        self._grasped_streak = 0
        self._just_grasped = False
        self._ever_grasped = False
        self._success = False
        self._dropped = False
        self._min_ee_to_cube = np.inf
        self._step_count = 0
        self._ik_failures = 0
        self._reset_control_period_stats()

        self.kin = FrankaKinematics(urdf_path)
        self._owns_rclpy = not rclpy.ok()
        
        if self._owns_rclpy:
            rclpy.init()
        self.sim_interface = SimInterface()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (cfg.OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32) # [dx, dy, dz, g]
        _, self.R_frozen = self.kin.fk(cfg.Q_READY)
    
    @property
    def obs_dict(self):
        return self._last_obs_dict
    
    def _sample_cube_pos(self):
        """ Samples cube's starting position - used for L2. """
        x = self.np_random.uniform(0.4, 0.6)
        y = self.np_random.uniform(-0.2, 0.2)
        z = 0.425
        return np.array([x,y,z])
    
    def _potential(self):
        """Computes Potential for Potential-Based Shaping Reward """
        d_reach = np.linalg.norm(self._last_obs_dict["ee_to_cube"])
        d_transport = np.linalg.norm(self._last_obs_dict["cube_to_goal"])
        
        phi = -cfg.W_TRANSPORT * d_transport
        
        if not self._last_obs_dict["is_grasped"]:
            phi -= cfg.W_REACH * d_reach
        
        return phi
    
    def _sample_goal_cube_pos(self):
        """ L1: static target (left table). L2/L3: random left or right. """
        if self.level == "L1":
            return cfg.GOAL_LEFT.copy()
        return (cfg.GOAL_LEFT if self.np_random.random() < 0.5 else cfg.GOAL_RIGHT).copy()

    def _get_obs(self):
        """ Returns normalized observation dictionary and saves last state. """
        self._last_state = self.sim_interface.wait_for_state()
        state = self._last_state
        self._last_obs_dict = obs.build_obs_dict(
            state["q_arm"], 
            state["gripper_opening"], 
            state["cube_pos"], 
            self.goal_pos, 
            self.kin
        )
        self._last_obs_dict["is_grasped"] = self._update_grasped(
            self._last_obs_dict["ee_to_cube"],
            state["cube_pos"],
            state["gripper_opening"]
        )
        self._min_ee_to_cube = min(
            self._min_ee_to_cube,
            float(np.linalg.norm(self._last_obs_dict["ee_to_cube"]))
        )
        return obs.normalize(self._last_obs_dict)

    def _reset_control_period_stats(self):
        self._t_last_cmd = None
        self._sim_dt_sum = 0.0
        self._sim_dt_max = 0.0
        self._sim_dt_n = 0

    def _record_control_period(self, t_cmd):
        """ Measures the real control period: sim time between two consecutive
            command publications. """

        if self._t_last_cmd is not None:
            sim_dt = t_cmd - self._t_last_cmd
            self._sim_dt_sum += sim_dt
            self._sim_dt_max = max(self._sim_dt_max, sim_dt)
            self._sim_dt_n += 1

        self._t_last_cmd = t_cmd

    @property
    def sim_dt_mean(self):
        """ Mean control period so far this episode; `dt` when nothing overruns. """
        return self._sim_dt_sum / self._sim_dt_n if self._sim_dt_n else self.dt

    def _get_info(self):
        """ Per-step info. The episode-level flags ("is_success", "is_grasped",
            "min_ee_to_cube") summarize the whole episode, so
            the value read at the final step is the episode's verdict. That is
            what Monitor(info_keywords=...) and SB3's rollout/success_rate read. """
        return {
            "cube_to_goal": self._last_obs_dict["cube_to_goal"],
            "ik_failures": self._ik_failures,
            "is_success": self._success,
            "is_grasped": self._ever_grasped,
            "dropped": self._dropped,
            "min_ee_to_cube": self._min_ee_to_cube,
            "sim_dt_mean": self.sim_dt_mean,
            "sim_dt_max": self._sim_dt_max,
        }
    
    def _compute_reward(self, action, phi_prev, phi_next, just_grasped, success):
        """ Computes reward additively.
            Reward is computed based on Potential which uses Potential-Based Shaping Reward 
            R_GRASP is paid once, on the rising edge - paying it per step
            would make hovering over the goal beat releasing the cube. """

        reward = cfg.GAMMA * phi_next - phi_prev

        if just_grasped:
            reward += cfg.R_GRASP

        if success:
            reward += cfg.R_SUCCESS

        reward -= cfg.W_ENERGY * np.sum(action**2)

        return reward
    
    def _update_grasped(self, ee_to_cube, cube_pos, gripper_opening):
        """ Checks whether cube is being held using _grasped_streak tracking
            We need to confirm that cube is really being held - that's why 
            we need streak to be over N_GRASP_CONFIRM."""
        
        self._just_grasped = False

        holding = (cfg.GRASP_OPENING_MIN < gripper_opening < cfg.GRASP_OPENING_MAX and np.linalg.norm(ee_to_cube) < cfg.GRASP_DIST_THRESH)
        if self._grasped:
            if not holding or cube_pos[2] < cfg.CUBE_POS_DEFAULT[2] + cfg.LIFT_MARGIN_EXIT:
                self._grasped = False
                self._grasped_streak = 0
        else:
            if holding and cube_pos[2] > cfg.CUBE_POS_DEFAULT[2] + cfg.LIFT_MARGIN_ENTER:
                self._grasped_streak += 1
            else:
                self._grasped_streak = 0
            if self._grasped_streak >= cfg.N_GRASP_CONFIRM:
                self._grasped = True
                self._just_grasped = True
                self._ever_grasped = True

        return self._grasped
        
    def _is_success(self, cube_pos, cube_vel, grasped):
        """ Checks if cube is final goal position within tolerance """
        
        on_target_xy = np.linalg.norm(cube_pos[:2] - self.goal_pos[:2]) < cfg.SUCCESS_XY_TOL
        on_table = abs(cube_pos[2] - self.goal_pos[2]) < cfg.SUCCESS_Z_TOL
        at_rest = np.linalg.norm(cube_vel) < cfg.SUCCESS_VEL_EPS
        
        return bool(on_target_xy and on_table and at_rest and not grasped)
    
    def reset(self, seed=None, options=None):
        """ Sequence: move away arm -> spawn cube """
        
        super().reset(seed=seed)
        self.sim_interface.publish_gripper_command(cfg.GRIPPER_RANGE[1], cfg.RESET_DT)
        self.sim_interface.publish_arm_command(cfg.Q_READY, cfg.RESET_DT)
        self._step_count = 0
        self._ik_failures = 0
        self._grasped_streak = 0
        self._grasped = False
        self._just_grasped = False
        self._ever_grasped = False
        self._success = False
        self._dropped = False
        self._min_ee_to_cube = np.inf
        self._reset_control_period_stats()

        for _ in range(cfg.N_SETTLE):
            self.sim_interface.reserve_t(self.dt)
            state = self.sim_interface.wait_for_state()
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
        phi_prev = self._potential()
        action = np.asarray(action, dtype=float)
        state = self.sim_interface.wait_for_state()
        q_current = state["q_arm"]
        
        p_ee_current, _ = self.kin.fk(q_current)
        delta = action[:3] * cfg.ACTION_SCALE
        p_des = p_ee_current + delta
        p_des= ik.clip_to_workspace(p_des, cfg.WORKSPACE_BOX)
        
        q_target, ik_output = ik.solve_ik(self.kin, q_current, p_des, self.R_frozen)
        
        if not ik_output["success"]:
            q_target = q_current
        opening = cfg.GRIPPER_RANGE[1] if action[3] > 0 else cfg.GRIPPER_RANGE[0]

        self._record_control_period(self.sim_interface.sim_time())
        self.sim_interface.publish_arm_command(q_target, self.dt)
        self.sim_interface.publish_gripper_command(opening, self.dt)
        self.sim_interface.reserve_t(self.dt)
        
        observation = self._get_obs()
        state = self._last_state

        grasped = self._last_obs_dict["is_grasped"]
        success = self._is_success(state["cube_pos"], state["cube_vel"], grasped)
        dropped = bool(state["cube_pos"][2] < cfg.CUBE_DROP_Z)

        self._success = success
        self._dropped = dropped
        
        self._step_count += 1
        terminated = bool(success or dropped)
        truncated = bool(self._step_count >= self.max_episode_steps)
        
        phi_next = 0.0 if terminated else self._potential()

        reward = self._compute_reward(action, phi_prev, phi_next, self._just_grasped, success)

        if not ik_output["success"]:
            self._ik_failures += 1

        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def close(self):
        """ Cleans up ROS node. Gymnasium calls close() from __del__,
            and SB3 at the end of the training phase. `rclpy.shutdown()` 
            only if we performed init(). """
        
        if getattr(self, "sim_interface", None) is not None:
            self.sim_interface.destroy_node()
            self.sim_interface = None

        if getattr(self, "_owns_rclpy", False) and rclpy.ok():
            rclpy.shutdown()
            self._owns_rclpy = False
        
        
        
        
        
        
        
        
        
        
        
