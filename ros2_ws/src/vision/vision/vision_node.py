import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy.linalg import logm, expm
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import conf_limo

class Vision(Node):

    def __init__(self):
        super().__init__('vision')
        self.publisher_vision = self.create_publisher(Float64MultiArray, '/vision', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = Float64MultiArray()
        odom_msg = Float64MultiArray()
        odom_msg = [random.random(), random.random()]
        msg.data = odom_msg
        self.publisher_vision.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
    
def main(args=None):

    rclpy.init(args=args)
    vision_test = Vision()

    rclpy.spin(vision)
    vision.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()