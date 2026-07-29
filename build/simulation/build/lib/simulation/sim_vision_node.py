import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import rclpy
from rclpy.node import Node
import numpy as np
from project_interfaces.msg import Measurement
from limo_description import conf_limo
import random as rm


class MeasPublisher(Node):
    def __init__(self):
        super().__init__('meas_publisher')
        self.id = id
        # publishing topic
        self.pub_measurement = self.create_publisher(Measurement, '/measurement_raw', 10)
        self.timer = self.create_timer(0.05, self.publish_meas)
        self.idx = 0
        self.counter = 0

    def publish_meas(self):
        rc = conf_limo.r_circle + 0.2
        tx = conf_limo.target_init[0] + 0.2 + self.counter*0.01
        ty = self.counter*0.005
        self.counter = self.counter + 1
        noise = 1

        msg = Measurement()
        msg.dtheta = 0.0
        # target measures
        if self.idx == 0:
            msg.id_a = 0
            msg.id_b = 3
            msg.x = tx + rc*np.cos(60*np.pi/180) + rm.uniform(-noise, noise)
            msg.y = rc*np.sin(60*np.pi/180) + rm.uniform(-noise, noise) + ty
            self.idx = self.idx+1
        elif self.idx == 1:
            msg.id_a = 1
            msg.id_b = 3
            msg.x = tx - rc + rm.uniform(-noise, noise)
            msg.y = 0.0 + 0.15 + rm.uniform(-noise, noise) + ty
            self.idx = self.idx+1
        elif self.idx == 2:
            msg.id_a = 2
            msg.id_b = 3
            msg.x = tx + rc*np.cos(60*np.pi/180) + rm.uniform(-noise, noise)
            msg.y = -rc*np.sin(60*np.pi/180) + rm.uniform(-noise, noise) + ty
            self.idx = self.idx+1

        # each other measures
        elif self.idx == 3:
            msg.id_a = 0
            msg.id_b = 1
            msg.x = rc + rc*np.cos(60*np.pi/180) + rm.uniform(-noise, noise)
            msg.y = rc*np.sin(60*np.pi/180) + rm.uniform(-noise, noise)
            self.idx = self.idx+1
        elif self.idx == 4:
            msg.id_a = 2
            msg.id_b = 1
            msg.x = rc + rc*np.cos(60*np.pi/180) + rm.uniform(-noise, noise)
            msg.y = -rc*np.sin(60*np.pi/180) + rm.uniform(-noise, noise)
            self.idx = 0
        self.pub_measurement.publish(msg)
        

def main():
    rclpy.init()
    node = MeasPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()