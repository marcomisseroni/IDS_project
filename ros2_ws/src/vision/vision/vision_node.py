import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from project_interfaces.msg import Measurement
from vision.Vision_class import Vision
import cv2

'''
Node that reads from /camera topic the current frame, elaborates the image with the vision class
and sends the message in the /measurement topic
'''

def image_msg_to_numpy(msg):
    # converting the image in the correct format for opencv
    if msg.encoding not in ['bgr8', 'rgb8']:
        raise ValueError(f"Encoding not supported: {msg.encoding}")
    frame = np.frombuffer(msg.data, dtype=np.uint8)
    frame = frame.reshape((msg.height, msg.width, 3))
    if msg.encoding == 'rgb8':
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

def depth_msg_to_numpy(msg):
    frame = np.frombuffer(msg.data, dtype=np.uint16)
    frame = frame.reshape((msg.height, msg.width))
    depth = frame.astype('float32')
    return depth

def compressed_image_msg_to_numpy(msg):
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode compressed image")
    return frame

def compressed_depth_to_numpy(msg):
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]
    np_arr = np.frombuffer(raw_data, np.uint8)
    depth_img = cv2.imdecode(
        np_arr,
        cv2.IMREAD_UNCHANGED
    )
    return depth_img

class Vision_node(Node):

    def __init__(self,id=0):
        super().__init__('vision_node')
        self.id = id
        # subscibed topics
        self.rgb_sub = self.create_subscription(CompressedImage, '/out/compressed', self.rgb_callback, 1)
        # publishing topic
        self.pub_measurement = self.create_publisher(Measurement, '/measurement', 10)
        self.vision_obj = Vision(self.get_logger())



    def rgb_callback(self, rgb_msg):
        # converting the images in the correct format
        frame = compressed_image_msg_to_numpy(rgb_msg)

        # elaborating the data
        target, limo0, limo1, limo2 = self.vision_obj.vision_main(frame, visualize=True)

        # publishing te data in 4 different messages (if we have a measure)
        # limo0 measured
        if limo0[0] != 0:
            msg = Measurement()
            msg.id_a = self.id
            msg.id_b = 0
            msg.x = limo0[0]
            msg.y = limo0[1]
            msg.dtheta = limo0[2]
            self.pub_measurement.publish(msg)
        # limo1
        if limo1[0] != 0:
            msg = Measurement()
            msg.id_a = self.id
            msg.id_b = 1
            msg.x = limo1[0]
            msg.y = limo1[1]
            msg.dtheta = limo1[2]
            self.pub_measurement.publish(msg)
        # limo2
        if limo2[0] != 0:
            msg = Measurement()
            msg.id_a = self.id
            msg.id_b = 2
            msg.x = limo2[0]
            msg.y = limo2[1]
            msg.dtheta = limo2[2]
            self.pub_measurement.publish(msg)
        # target
        if target[0] != 0:
            msg = Measurement()
            msg.id_a = self.id
            msg.id_b = 3
            msg.x = target[0]
            msg.y = target[1]
            msg.dtheta = 0.0
            self.pub_measurement.publish(msg)
    
def main(args=None):
    rclpy.init()
    node = Vision_node(id=2)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()