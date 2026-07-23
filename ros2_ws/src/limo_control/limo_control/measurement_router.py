import numpy as np
# ros2 stuff
import rclpy
from rclpy.node import Node
# message types used
from project_interfaces.msg import Measurement
from limo_description import conf_limo as conf_limo
from pathlib import Path

ROOT_ROS_WS = Path(__file__).parent.parent.parent.parent.parent.parent.parent
CSV_PATH = ROOT_ROS_WS / "csv_data"

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
        # csvs
        self.csv_file = open(CSV_PATH / "data.csv", "w")
        self.csv_file.write(f"id_a,id_b,x,y,dtheta\n")

        self.csv_file_routed = open(CSV_PATH / "data_routed.csv", "w")
        self.csv_file_routed.write(f"id_a,id_b,x,y,dtheta,msg_id\n")

        # msg id to reconstruct the order
        self.msg_id = 0

        # threshold
        self.th = 1.0

    def check_th(self, idx, msg):
        if self.meas[idx] is None:
            return True
        dtheta_diff = self.meas[idx].dtheta - msg.dtheta
        dtheta_diff = abs(np.arctan2(np.sin(dtheta_diff), np.cos(dtheta_diff)))
        if (abs(self.meas[idx].x - msg.x) > self.th or
           abs(self.meas[idx].y - msg.y) > self.th or
           dtheta_diff > self.th):
            return False
        return True

    def meas_callback(self, msg):

        key = (msg.id_a, msg.id_b)
        idx = self.map_idx.get(key)

        if idx is not None and self.check_th(idx, msg):
            self.meas[idx] = msg

        self.csv_file.write(f"{msg.id_a},{msg.id_b},{msg.x},{msg.y},{msg.dtheta}\n")


    def meas_pub_callback(self):
        n = len(self.meas)
        if all(m is None for m in self.meas):
            return
        for i in range(n):
            idx = (self.idx + i) % n
            if self.meas[idx] is not None:
                self.csv_file_routed.write(f"{self.meas[idx].id_a},{self.meas[idx].id_b},{self.meas[idx].x},{self.meas[idx].y},{self.meas[idx].dtheta},{self.msg_id}\n")
                self.msg_id += 1
                self.pub_meas.publish(self.meas[idx])
                #self.meas = [None] * n
                self.meas[idx] = None
                self.idx = (idx + 1) % n
                return

    def destroy_node(self):
        self.csv_file.close()
        self.csv_file_routed.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MeasurementRouter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
