import rclpy
from rclpy.node import Node
from project_interfaces.msg import State
import numpy as np

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('ekf_simulator')
        self.pub_person_state = self.create_publisher(State, '/person_state', 10)
        self.timer = self.create_timer(0.1, self.publish_state)
        self.idx = 0

    def publish_state(self):
        msg_person = State()
        msg_person.id = 3
        msg_person.x = 1 + 0.01*self.idx
        msg_person.y = 0.8 * np.sin(2 * np.pi * 0.2 * self.idx * 0.1)
        msg_person.theta = 0.0
        self.pub_person_state.publish(msg_person)
        self.idx = self.idx + 1

def main():
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()