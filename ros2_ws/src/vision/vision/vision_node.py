import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
import numpy as np
from sensor_msgs.msg import Image
from vision.Vision_class import Vision
from cv_bridge import CvBridge
import cv2

class Vision_node(Node):

    def __init__(self):
        super().__init__('vision_node')
        self.sub_rgb = self.create_subscription(
            Image,
            '/vision',
            self.frame_callback,
            10)
        self.bridge = CvBridge()
        self.vision_obj = Vision()

    def frame_callback(self, msg):
        self.get_logger().info('Received image')
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        target, limo0, limo1, limo2 = self.vision_obj.vision_main(frame, None, visualize=False)
        print(limo0)
    
def main(args=None):
    rclpy.init()
    node = Vision_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()