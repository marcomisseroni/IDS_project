import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import rclpy
from rclpy.node import Node
import numpy as np
from project_interfaces.msg import Measurement


class MeasPublisher(Node):
    def __init__(self):
        super().__init__('meas_publisher')
        self.id = id
        # publishing topic
        self.pub_measurement = self.create_publisher(Measurement, '/measurement', 10)
        self.timer = self.create_timer(0.033, self.publish_meas)
        self.idx = 0

    def publish_meas(self):
        #print("meas")
        msg = Measurement()
        msg.id_a = 2
        msg.id_b = 3
        msg.x = 4.0
        msg.y = 0.5 + 0.8 * np.sin(2 * np.pi * 0.1 * self.idx)
        msg.dtheta = 0.0
        self.pub_measurement.publish(msg)
        self.idx = self.idx+1
        

def main():
    rclpy.init()
    node = MeasPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()