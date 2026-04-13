from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    yaml_dir = get_package_share_directory('limo_control')
    yaml_file = os.path.join(yaml_dir, 'param', 'limo_params.yaml')

    return LaunchDescription([
        Node(
            package='limo_control',
            namespace='limo1',
            executable='EKF_node',
            name='ekf1',
            parameters=[yaml_file]
        ),
        Node(
            package='limo_control',
            namespace='limo2',
            executable='EKF_node',
            name='ekf2',
            parameters=[yaml_file]
        ),
            Node(
            package='limo_control',
            namespace='limo3',
            executable='EKF_node',
            name='ekf3',
            parameters=[yaml_file]
        ),
            Node(
            package='limo_control',
            namespace='limo1',
            executable='MPC_node',
            name='mpc1',
            parameters=[yaml_file]

        ),
            Node(
            package='limo_control',
            namespace='limo2',
            executable='MPC_node',
            name='mpc2',
            parameters=[yaml_file]
        ),
            Node(
            package='limo_control',
            namespace='limo3',
            executable='MPC_node',
            name='mpc3',
            parameters=[yaml_file]
        )
    ])