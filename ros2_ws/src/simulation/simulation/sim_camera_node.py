import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
#from cv_bridge import CvBridge
import cv2

def numpy_to_image_msg(frame, encoding='bgr8'):
    msg = Image()
    msg.height, msg.width = frame.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    channels = 1 if len(frame.shape) == 2 else frame.shape[2]
    msg.step = msg.width * channels 
    msg.data = frame.tobytes()
    return msg

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.pub_rgb = self.create_publisher(Image, '/camera', 10)
        self.pub_depth = self.create_publisher(Image, '/depth', 10)
        #self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.publish_image)

        self.cap = cv2.VideoCapture("src/simulation/RGB3.mp4")
        self.cap_d = cv2.VideoCapture("src/simulation/depth3.mp4")

    def publish_image(self):
        ret, frame = self.cap.read()
        ret_d, frame_d = self.cap_d.read()
        if ret and ret_d:
            # publishing rgb frame
            #rgb = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            rgb = numpy_to_image_msg(frame, encoding='bgr8')
            self.pub_rgb.publish(rgb)
            # publishing depth frame
            #depth = self.bridge.cv2_to_imgmsg(frame_d, encoding='bgr8')
            depth = numpy_to_image_msg(frame_d, encoding='bgr8')
            self.pub_depth.publish(depth)

            self.get_logger().info('Publishing image')
        else:
            self.get_logger().info('Publishing image error')

def main():
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()