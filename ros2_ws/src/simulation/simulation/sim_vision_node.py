import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import rclpy
from rclpy.node import Node
import numpy as np
from project_interfaces.msg import Measurement
from limo_description import conf_limo


class MeasPublisher(Node):
    def __init__(self):
        super().__init__('meas_publisher')
        self.id = id
        # publishing topic
        self.pub_measurement = self.create_publisher(Measurement, '/measurement_raw', 10)
        self.timer = self.create_timer(0.02, self.publish_meas)
        self.idx = 0
        self.counter = 0

    def publish_meas(self):
        rc = conf_limo.r_circle + 0.2
        tx = conf_limo.target_init[0] + 0.2 + self.counter*0.001
        self.counter = self.counter + 1

        msg = Measurement()
        msg.id_b = 3
        msg.dtheta = 0.0
        if self.idx == 0:
            msg.id_a = 0
            msg.x = tx + rc*np.cos(60*np.pi/180)
            msg.y = rc*np.sin(60*np.pi/180)
            self.idx = self.idx+1
        elif self.idx == 1:
            msg.id_a = 1
            msg.x = tx - rc
            msg.y = 0.0
            self.idx = self.idx+1
        elif self.idx == 2:
            msg.id_a = 2
            msg.x = tx + rc*np.cos(60*np.pi/180)
            msg.y = -rc*np.sin(60*np.pi/180)
            self.idx = 0
        self.pub_measurement.publish(msg)
        

def main():
    rclpy.init()
    node = MeasPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()