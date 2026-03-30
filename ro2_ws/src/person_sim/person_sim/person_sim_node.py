import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
import numpy as np

class PersonSimNode(Node):
    def __init__(self, traj_type = 'sin', period = 1.0):
        super().__init__('person_sim_node')
        self.publisher_pp = self.create_publisher(Float64MultiArray, 'person_pos', 10)
        self.publisher_e = self.create_publisher(Float64MultiArray, 'enc', 10)
        self.publisher_i = self.create_publisher(Float64MultiArray, 'imu', 10)
        self.publisher_l = self.create_publisher(Float64MultiArray, 'lidar', 10)
        self.publisher_a = self.create_publisher(String, 'admin', 10)
        time_period = period
        self.timer = self.create_timer(time_period, self.timer_callback)
        self.i = 0
        self.traj_type = traj_type

    def timer_callback(self):
        msg = Float64MultiArray()
        pos = self.trajectory()
        msg.data = [float(self.i), pos]
        self.publisher_pp.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
        enc_msg = Float64MultiArray()
        enc_msg.data = [0.5, float(self.i) * 0.1]
        self.publisher_e.publish(enc_msg)
        self.get_logger().info('Publishing: "%s"' % enc_msg.data)
        imu_msg = Float64MultiArray()
        imu_msg.data = [float(self.i) * 0.1]
        self.publisher_i.publish(imu_msg)
        self.get_logger().info('Publishing: "%s"' % imu_msg.data)
        lidar_msg = Float64MultiArray()
        lidar_msg.data = [0.0, 0.0, 0.0]
        self.publisher_l.publish(lidar_msg)
        self.get_logger().info('Publishing: "%s"' % lidar_msg.data)

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
    msg = String()
    msg.data = "start"
    msg_end = String()
    msg_end.data = "stop"
    person_sim_node.publisher_a.publish(msg)
    rclpy.spin(person_sim_node)
    person_sim_node.publisher_a.publish(msg_end)
    person_sim_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()