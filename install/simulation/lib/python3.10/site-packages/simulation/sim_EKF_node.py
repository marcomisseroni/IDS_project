import rclpy
from rclpy.node import Node
from project_interfaces.msg import State
import numpy as np
import random

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('ekf_simulator')
        self.pub_person_state = self.create_publisher(State, '/person_state', 10)
        self.pub_limo_state = self.create_publisher(State, '/limo_state', 10)
        self.timer = self.create_timer(0.1, self.publish_state)
        self.idx = 0

    def publish_state(self):
        # person
        msg_person = State()
        msg_person.id = 3
        msg_person.x = 1 + 0.01*self.idx
        msg_person.y = 0.8 * np.sin(2 * np.pi * 0.2 * self.idx * 0.1)
        msg_person.theta = 0.0
        self.pub_person_state.publish(msg_person)
        # limo
        msg_limo = State()
        msg_limo.id = 2
        msg_limo.x = 0.008*self.idx
        msg_limo.y = 0.001*self.idx + random.uniform(-0.005, 0.005)
        msg_limo.theta = 0.001*self.idx
        self.pub_limo_state.publish(msg_limo)

        self.idx = self.idx + 1

def main():
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()