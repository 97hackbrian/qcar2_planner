#!/usr/bin/env python3
# =============================================================================
# qcar2_mapping_planner.launch.py
# =============================================================================
# Launches the complete qcar2_planner architecture:
#   1. map_processor_node      — Perception engine (mesh → GridMap)
#   2. exploration_manager_node — State monitor (MAPPING / READY)
#   3. directional_planner_server — Oriented A* planner service
#
# All nodes load parameters from config/params.yaml.
# Supports use_sim_time argument for simulation environments.
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Package paths ───────────────────────────────────────────────────────
    pkg_dir = get_package_share_directory('qcar2_planner')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    # ── Launch arguments ────────────────────────────────────────────────────
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock (set to true for Gazebo/rosbag)'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')

    # ── Node 1: map_processor_node ──────────────────────────────────────────
    map_processor = Node(
        package='qcar2_planner',
        executable='map_processor_node.py',
        name='map_processor_node',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time}
        ],
        emulate_tty=True
    )

    # ── Node 2: exploration_manager_node ────────────────────────────────────
    exploration_manager = Node(
        package='qcar2_planner',
        executable='exploration_manager_node.py',
        name='exploration_manager_node',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time}
        ],
        emulate_tty=True
    )

    # ── Node 3: directional_planner_server ──────────────────────────────────
    directional_planner = Node(
        package='qcar2_planner',
        executable='directional_planner_server.py',
        name='directional_planner_server',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time}
        ],
        emulate_tty=True
    )

    return LaunchDescription([
        use_sim_time_arg,
        map_processor,
        exploration_manager,
        directional_planner,
    ])
