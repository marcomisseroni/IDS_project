import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from project_interfaces.msg import State
import matplotlib.pyplot as plt


class EKFPlot(Node):
    
    def __init__(self):

        super().__init__('ekf_plot')
        self.limo_states = []
        self.person_states = []
        self.t = []

        self.start_time = None

        # subs
        self.sub_admin = self.create_subscription(String, '/admin', self.admin_callback, 10)
        self.sub_ekf_limo = self.create_subscription(State, '/limo_state', self.limo_state_callback, 10)
        self.sub_ekf_person = self.create_subscription(State, '/person_state', self.person_state_callback, 10)

    def admin_callback(self, msg):
        if(msg.data == 'stop'):
            self.limo_states = np.array(self.limo_states)
            self.person_states = np.array(self.person_states)
            self.plot()

    def plot(self):
        plt.figure()

        plt.plot(self.t, self.limo_states[:, 0], label="x")
        plt.plot(self.t, self.limo_states[:, 1], label="y")
        plt.plot(self.t, self.limo_states[:, 2], label="theta")

        plt.xlabel("Time")
        plt.ylabel("State value")
        plt.legend()
        plt.grid()
        plt.show()

    def limo_state_callback(self, msg):
        x = msg.x
        y = msg.y
        theta = msg.theta
        self.limo_states.append((x, y, theta))
        if(self.start_time == None):
            self.start_time = self.get_clock().now()
            self.t.append(0)
        else:
            self.t.append((self.get_clock().now() - self.start_time).nanoseconds * 1e-9)

    def person_state_callback(self, msg):
        x = msg.x
        y = msg.y
        self.person_states.append((x, y))

def main():

    rclpy.init()

    ekf_plot = EKFPlot()
    rclpy.spin(ekf_plot)
    ekf_plot.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()