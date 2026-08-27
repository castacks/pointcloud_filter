"""
Launch file for outlier filtering on the Super Odometry pointcloud.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("pointcloud_filter")

    robot_name = LaunchConfiguration("robot_name")

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=os.path.join(pkg_share, "config", "outlier_filter.yaml"),
        description="Path to the outlier filter configuration file",
    )
    robot_name_arg = DeclareLaunchArgument(
        "robot_name",
        default_value="spot2",
        description="Robot namespace",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock",
    )

    # Outlier filter node
    outlier_filter_node = Node(
        package="pointcloud_filter",
        executable="outlier_filter_node",
        name="outlier_filter",
        namespace=robot_name,
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    return LaunchDescription(
        [config_file_arg, robot_name_arg, use_sim_time_arg, outlier_filter_node]
    )
