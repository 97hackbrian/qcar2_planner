#!/usr/bin/env python3
# =============================================================================
# qcar2_planner_server.launch.py
# =============================================================================
# Launch file with two modes via the `auto_enable` argument:
#
#   auto_enable=true  → Run existing mapping pipeline
#                        (map_processor + exploration_manager + planner_server)
#
#   auto_enable=false → Load a saved map and navigate with planner_server
#                        (map_loader + planner_server only)
#
# All topic names are configurable via launch arguments.
# =============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ── Package paths ───────────────────────────────────────────────────────
    pkg_dir = get_package_share_directory('qcar2_planner')
    params_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    # ── Launch arguments ────────────────────────────────────────────────────
    auto_enable_arg = DeclareLaunchArgument(
        'auto_enable',
        default_value='false',
        description='true = run mapping pipeline, false = load saved map'
    )
    map_yaml_arg = DeclareLaunchArgument(
        'map_yaml_path',
        default_value='/workspaces/isaac_ros-dev/ros2/src/qcar2_planner/config/mapV4uncertainty.yaml',
        description='Path to the saved map YAML file (required for auto_enable=false)'
    )
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock'
    )
    goal_topic_arg = DeclareLaunchArgument(
        'goal_input_topic',
        default_value='/bt/goal',
        description='Topic for receiving the main navigation goal'
    )
    mission_topic_arg = DeclareLaunchArgument(
        'mission_goals_topic',
        default_value='/mission_goals',
        description='Topic for publishing sequential semi-goals'
    )

    auto_enable = LaunchConfiguration('auto_enable')
    map_yaml_path = LaunchConfiguration('map_yaml_path')
    use_sim_time = LaunchConfiguration('use_sim_time')
    goal_input_topic = LaunchConfiguration('goal_input_topic')
    mission_goals_topic = LaunchConfiguration('mission_goals_topic')

    # =====================================================================
    # MODE 1: auto_enable=true → Include existing mapping launch
    # =====================================================================
    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_dir, 'launch', 'qcar2_mapping_planner.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
        condition=IfCondition(auto_enable),
    )

    # =====================================================================
    # MODE 2: auto_enable=false → Map loader + Planner server
    # =====================================================================
    map_loader = Node(
        package='qcar2_planner',
        executable='map_loader_node.py',
        name='map_loader_node',
        output='screen',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'map_yaml_path': map_yaml_path,
            }
        ],
        emulate_tty=True,
        condition=UnlessCondition(auto_enable),
    )

    planner_server = Node(
        package='qcar2_planner',
        executable='directional_planner_server.py',
        name='directional_planner_server',
        output='screen',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'auto_enable': True,
                'goal_input_topic': goal_input_topic,
                'mission_goals_topic': mission_goals_topic,
            }
        ],
        emulate_tty=True,
        condition=UnlessCondition(auto_enable),
    )

    return LaunchDescription([
        auto_enable_arg,
        map_yaml_arg,
        use_sim_time_arg,
        goal_topic_arg,
        mission_topic_arg,
        # Mode 1: mapping pipeline
        mapping_launch,
        # Mode 2: saved-map navigation
        map_loader,
        planner_server,
    ])
