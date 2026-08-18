import os
from franka_rl.franka_rl.urdf_utils import strip_finger_mimic 
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, IncludeLaunchDescription
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro

def generate_launch_description():
    pkg = get_package_share_directory('franka_sim')

    #--- URDF ---
    xacro_file = os.path.join(pkg, 'urdf', 'fr3_gazebo.urdf.xacro')
    robot_description = strip_finger_mimic(xacro.process_file(xacro_file).toxml())

    #--- WORLD ---
    world_file = os.path.join(pkg, 'worlds', 'fr3_world.sdf')
    
    #--- Gazebo ---
    gz_sim_pkg = get_package_share_directory('ros_gz_sim')
    gz_launch_file = os.path.join(gz_sim_pkg, 'launch', 'gz_sim.launch.py')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_launch_file),
        launch_arguments={'gz_args':f'-r -s {world_file}'}.items()
    )
    
    #--- Robot State Publisher ---
    rsp = Node(
        package = 'robot_state_publisher',
        executable = 'robot_state_publisher',
        output = 'screen',
        parameters = [{'robot_description': robot_description,
                     'use_sim_time': True}], 
    )
    
    # --- Robot Spawn ---
    robot_spawn = Node(
        package = 'ros_gz_sim',
        executable = 'create',
        arguments = ['-topic', 'robot_description', '-name', 'fr3'],
        output = 'screen'
    )
    
    # --- Controller Spawn ---
    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],   
    )
    arm_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['fr3_arm_controller'],   
    )
    gripper_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['fr3_gripper_controller'],   
    )
    
    # --- Clock Bridge ---
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )
    
    # --- Cube Pose Bridge ---
    # Poza kostki z pluginu PosePublisher wpietego w model `cube` (worlds/fr3_world.sdf).

    cube_pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/model/cube/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'],
        output='screen',
    )
    
    
    #--- Sequence Spawn ---
    delay_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=robot_spawn,
            on_exit=[jsb_spawner]
        )
    )
    
    delay_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jsb_spawner,
            on_exit=[arm_spawner, gripper_spawner]
        )
    )
    
    return LaunchDescription([gazebo,
                              clock_bridge,
                              cube_pose_bridge,
                              rsp,
                              robot_spawn,
                              delay_jsb,
                              delay_controllers])
    
    
    
    
    
    