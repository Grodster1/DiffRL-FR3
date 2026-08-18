import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Duration

ARM_JOINTS = [f"fr3_joint{i}" for i in range (1, 8)]
GRIPPER_JOINTS = ["fr3_finger_joint1", "fr3_finger_joint2"]

class SimInterface(Node):
    def __init__(self):
        super().__init__("sim_interface")
        
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.parameter.Type.BOOL, True)])
        
        self._q_arm = None
        self._dq_arm = None
        self._cube_pos = None
        self._sim_time = None
        
        
        self._joint_states_sub = self.create_subscription(JointState, "/joint_states", self._on_joint_states_callback, 10)
        self._cube_odom_sub = self.create_subscription(Odometry, "/model/cube/odometry", self._on_cube_callback, 10)
        self._clock_sub = self.create_subscription(Clock, "/clock", self._on_clock_callback, 10)
        
        self._arm_pub = self.create_publisher(JointTrajectory, "/fr3_arm_controller/joint_trajectory", 10)
        self._gripper_pub = self.create_publisher(JointTrajectory, "/fr3_gripper_controller/joint_trajectory", 10)
        
    def _on_joint_states_callback(self, msg: JointState):
        name_pos = dict(zip(msg.name, msg.position))
        name_vel = dict(zip(msg.name, msg.velocity))
        
        self._gripper_opening = name_pos["fr3_finger_joint1"]
        
        self._q_arm = np.array([name_pos[j] for j in ARM_JOINTS])
        self._dq_arm = np.array([name_vel[j] for j in ARM_JOINTS])
        
    def _on_cube_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        self._cube_pos = np.array([pos.x, pos.y, pos.z])
        
    def _on_clock_callback(self, msg: Clock):
        self._sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        
    def publish_arm_command(self, q_arm_target, dt):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        traj_point = JointTrajectoryPoint()
        traj_point.positions = list(q_arm_target)
        
        traj_point.time_from_start = Duration(sec=0, nanosec=int(dt*1e9))
        msg.points = [traj_point]
        self._arm_pub.publish(msg)
        
    def publish_gripper_command(self, opening, dt):
        msg = JointTrajectory()
        msg.joint_names = GRIPPER_JOINTS
        traj_point = JointTrajectoryPoint()
        traj_point.positions = [opening, opening]
        
        traj_point.time_from_start = Duration(sec=0, nanosec=int(dt*1e9))
        msg.points = [traj_point]
        self._gripper_pub.publish(msg)
        
    def reserve_t(self, dt):
        start = self._sim_time
        while self._sim_time is None:
            rclpy.spin_once(self, timeout_sec=0.01)
            
        start = self._sim_time
        while (self._sim_time - start) < dt:
            rclpy.spin_once(self, timeout_sec=0.01) 
        
    def get_state(self):
        return {
            "gripper_opening": self._gripper_opening,
            "q_arm": self._q_arm,
            "dq_arm": self._dq_arm,
            "cube_pos": self._cube_pos,
        }