from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='localization',
            namespace='',
            executable='EKF_node',
            name='ekf_node',
            arguments=['2']
        ),
        Node(
            package='vision',
            namespace='',
            executable='vision_node',
            name='vision_node',
            arguments=['2']
        ),
        Node(
            package='limo_control',
            namespace='',
            executable='MPC_node',
            name='MPC_node',
            arguments=['2']
        ),
        Node(
            package='plots',
            namespace='',
            executable='EKF_plot_node',
            name='ekf_plot_node'
        ),
        Node(
            package='limo_control',
            namespace='',
            executable='measurement_router',
            name='measurement_router'
        )
    ])