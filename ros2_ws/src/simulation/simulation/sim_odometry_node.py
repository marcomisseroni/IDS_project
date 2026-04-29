import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import random

class SimOdometry(Node):

    def __init__(self):
        super().__init__('sim_odometry')
        self.publisher_odom = self.create_publisher(Float64MultiArray, 'odom', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Float64MultiArray()
        msg.data = self.sim_data()
        self.publisher_odom.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

    def sim_data(self):
        odom_msg = Float64MultiArray()
        odom_msg = [random.random(), random.random()]
        return odom_msg
    
def main(args=None):

    rclpy.init(args=args)
    sim_odometry = SimOdometry()

    rclpy.spin(sim_odometry)
    sim_odometry.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()