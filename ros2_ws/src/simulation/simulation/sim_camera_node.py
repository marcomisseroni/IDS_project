import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'vision', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.publish_image)

        self.cap = cv2.VideoCapture("src/simulation/RGB3.mp4")

    def publish_image(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing image')
        else:
            self.get_logger().info('Publishing image error')

def main():
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()