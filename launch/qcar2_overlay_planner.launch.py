#!/usr/bin/env python3
# =============================================================================
# qcar2_overlay_planner.launch.py
# =============================================================================
# Launches the complete qcar2_planner architecture with Cartographer and Map Overlay:
#   1. fixed_lidar_frame (TF)
#   2. cartographer_node (SLAM)
#   3. cartographer_occupancy_grid_node (/map)
#   4. tf_to_odom_node.py (Odometry)
#   5. map_overlay_node.py (Aligns PGM with Carto)
#   6. directional_planner_server.py (Path planning)
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Package paths ───────────────────────────────────────────────────────
    pkg_dir = get_package_share_directory('qcar2_planner')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')
    carto_config_dir = os.path.join(pkg_dir, 'config')

    # ── Launch arguments ────────────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')
    
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml_path',
        default_value=os.path.join(pkg_dir, 'config', 'map_ros.yaml'),
        description='Path to the PGM map YAML file'
    )
    
    alignment_mode_arg = DeclareLaunchArgument(
        'alignment_mode',
        default_value='manual', #cambiar aqui
        description='Alignment mode: auto (ICP) or manual (/initialpose)'
    )

    goal_topic_arg = DeclareLaunchArgument(
        'goal_input_topic',
        default_value='/bt/goal',
        description='Topic for receiving the main navigation goal'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml_path = LaunchConfiguration('map_yaml_path')
    alignment_mode = LaunchConfiguration('alignment_mode')
    goal_input_topic = LaunchConfiguration('goal_input_topic')

    # ── Nodes ───────────────────────────────────────────────────────────────

    # 1. TF for Lidar
    qcar2_to_lidar_tf_node = Node(
        package='qcar2_nodes',
        executable='fixed_lidar_frame',
        name='fixed_lidar_frame'
    )

    # 2. Cartographer SLAM
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('imu', '/qcar2_imu')],
        arguments=[
            '-configuration_directory', carto_config_dir,
            '-configuration_basename', 'cartographer_2d.lua'
        ]
    )

    # 3. Cartographer Occupancy Grid (/map)
    cartographer_occ_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=['-resolution', '0.05', '-publish_period_sec', '1.0']
    )

    # 4. TF to Odom
    tf_to_odom_node = Node(
        package='qcar2_planner',
        executable='tf_to_odom_node.py',
        name='tf_to_odom',
        output='screen',
        parameters=[{
            'output_topic': '/odom_filtered',
            'use_sim_time': use_sim_time,
        }],
    )

    # 5. Map Overlay Node
    map_overlay_node = Node(
        package='qcar2_planner',
        executable='map_overlay_node.py',
        name='map_overlay_node',
        output='screen',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'map_yaml_path': map_yaml_path,
                'alignment_mode': alignment_mode
            }
        ],
        emulate_tty=True
    )

    # 6. Directional Planner Server
    planner_server = Node(
        package='qcar2_planner',
        executable='directional_planner_server.py',
        name='directional_planner_server',
        output='screen',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'auto_enable': True, # Always enabled in this pipeline
                'goal_input_topic': goal_input_topic,
            }
        ],
        emulate_tty=True
    )

    return LaunchDescription([
        use_sim_time_arg,
        map_yaml_arg,
        alignment_mode_arg,
        goal_topic_arg,
        
        qcar2_to_lidar_tf_node,
        cartographer_node,
        cartographer_occ_grid_node,
        #tf_to_odom_node,
        map_overlay_node,
        planner_server
    ])
