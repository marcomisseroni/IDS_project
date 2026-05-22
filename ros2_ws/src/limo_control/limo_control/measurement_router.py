import numpy as np
# ros2 stuff
import rclpy
from rclpy.node import Node
# message types used
from project_interfaces.msg import Measurement
from limo_description import conf_limo as conf_limo

class MeasurementRouter(Node):

    def __init__(self):
        super().__init__('measurement_router')

        # topics to subscribe
        self.sub_meas1 = self.create_subscription(Measurement, '/measurement_raw', self.meas_callback, 10)

        # topics to publish
        self.meas_timer = self.create_timer(conf_limo.Tm, self.meas_pub_callback)
        self.pub_meas = self.create_publisher(Measurement, '/measurement_routed', 10)

        # last measures - 9 possible measurements: 0-1, 0-2, 0-3 | 1-0, 1-2, 1-3 | 2-0, 2-1, 2-3 (takes meas - gets measured) 
        self.meas = [None] * 9
        self.idx = 0
        self.map_idx = {
            (0,1): 0,
            (0,2): 1,
            (0,3): 2,
            (1,0): 3,
            (1,2): 4,
            (1,3): 5,
            (2,0): 6,
            (2,1): 7,
            (2,3): 8,
        }

    def meas_callback(self, msg):

        key = (msg.id_a, msg.id_b)
        idx = self.map_idx.get(key)

        if idx is not None:
            self.meas[idx] = msg


    def meas_pub_callback(self):

        n = len(self.meas)

        if all(m is None for m in self.meas):
            return
        
        for i in range(n):
            idx = (self.idx + i) % n

            if self.meas[idx] is not None:
                self.pub_meas.publish(self.meas[idx])
                self.meas = [None] * n
                self.idx = (idx + 1) % n
                return


def main(args=None):
    rclpy.init(args=args)
    node = MeasurementRouter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
