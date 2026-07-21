from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='localization',
            namespace='/limo_0',
            executable='EKF_node',
            name='ekf_node',
            arguments=['0']
        ),
        Node(
            package='localization',
            namespace='/limo_1',
            executable='EKF_node',
            name='ekf_node',
            arguments=['1']
        ),
        Node(
            package='localization',
            namespace='/limo_2',
            executable='EKF_node',
            name='ekf_node',
            arguments=['2']
        ),
        Node(
            package='limo_control',
            namespace='/limo_0',
            executable='MPC_node',
            name='MPC_node',
            arguments=['0']
        ),
        Node(
            package='limo_control',
            namespace='/limo_1',
            executable='MPC_node',
            name='MPC_node',
            arguments=['1']
        ),
        Node(
            package='limo_control',
            namespace='/limo_2',
            executable='MPC_node',
            name='MPC_node',
            arguments=['2']
        ),
        Node(
            package='plots',
            namespace='',
            executable='EKF_plot_node',
            name='ekf_plot_node'
        )
    ])