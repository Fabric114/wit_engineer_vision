"""ROS 2 显示 launch: 加载 rm26_engineer URDF, 起 robot_state_publisher +
joint_state_publisher_gui + rviz2, 用来在 rviz 里查看/拖动关节。

用法:
    ros2 launch rm26_engineer_description display.launch.py
可选:
    ros2 launch rm26_engineer_description display.launch.py rviz_config:=/path/to.rviz
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("rm26_engineer_description")
    urdf_path = os.path.join(pkg_share, "urdf", "GC.urdf")

    with open(urdf_path, "r", encoding="utf-8") as fh:
        robot_description = fh.read()

    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription([
        DeclareLaunchArgument(
            "rviz_config",
            default_value="",
            description="rviz2 配置文件路径, 留空则以默认视图启动",
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
        ),
        # 有传 rviz_config 就用 -d 加载, 否则空配置启动
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(
                PythonExpression(["'", rviz_config, "' != ''"])
            ),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(
                PythonExpression(["'", rviz_config, "' == ''"])
            ),
        ),
    ])
