import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np

class PersonSimNode(Node):
    def __init__(self, traj_type = 'sin', period = 1.0):
        super().__init__('person_sim_node')
        self.publisher_ = self.create_publisher(String, 'person_pos', 10)
        time_period = period
        self.timer = self.create_timer(time_period, self.timer_callback)
        self.i = 0
        self.traj_type = traj_type

    def timer_callback(self):
        msg = String()
        pos = self.trajectory()
        msg.data = [self.i, pos]
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

    def trajectory(self):
        if self.traj_type == 'sin':
            T = 20.0
            A = 10.0
            offset = 2.0
            phi = 0.0
            return A * np.sin(2.0 * np.pi * self.i / T + phi) + offset
        if self.traj_type == 'exp':
            tau = 10.0
            A = 2.0
            return A * np.exp(-self.i / tau)

def main(args=None):
    rclpy.init(args=args)
    person_sim_node = PersonSimNode(period = 0.5)
    rclpy.spin(person_sim_node)
    person_sim_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()