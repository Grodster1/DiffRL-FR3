import rclpy
from franka_rl.ros_bridge import SimInterface

rclpy.init()
node = SimInterface()

for _ in range(100):
    rclpy.spin_once(node, timeout_sec=0.01)
    
state = node.get_state()
print(f"gripper_opening:    {state["gripper_opening"]}")
print(f"q_arm:              {state["q_arm"]}")
print(f"dq_arm:             {state["dq_arm"]}")
print(f"cube_pos:           {state["cube_pos"]}")

node.reserve_t(0.05)
print("advance OK, sim_time:", node._sim_time)
rclpy.shutdown()