import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from project_interfaces.msg import State
from project_interfaces.msg import MPCprediction
from project_interfaces.msg import Measurement
from project_interfaces.msg import Desired
import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class EKFPlot(Node):
    
    def __init__(self):

        super().__init__('ekf_plot')
        self.limo_states = []
        self.person_states = []
        self.meas = []
        self.t = []
        self.x_pred = []
        self.y_pred = []
        self.des0 = []
        self.des1 = []
        self.des2 = []

        self.start_time = None
        self.is_running = True

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)
        self._make_window_non_intrusive()
        plt.show(block=False)

        # subs
        self.sub_admin = self.create_subscription(String, '/admin', self.admin_callback, 10)
        self.sub_ekf_limo = self.create_subscription(State, '/limo_state', self.limo_state_callback, 10)
        self.sub_ekf_person = self.create_subscription(State, '/person_state', self.person_state_callback, 10)
        self.sub_mpc_pred = self.create_subscription(MPCprediction, '/mpc_prediction', self.mpc_pred_callback, 10)
        self.sub_meas = self.create_subscription(Measurement, '/measurement', self.meas_callback, 10)
        self.sub_des = self.create_subscription(Desired, '/desired', self.des_callback, 10)

    def admin_callback(self, msg):
        if(msg.data == 'stop_ekf'):
            #self.get_logger().info('Called admin_callback: stop_ekf')
            self.is_running = False
            self.plot_limo_states()

    def _make_window_non_intrusive(self):
        try:
            manager = plt.get_current_fig_manager()
            window = getattr(manager, 'window', None)
            if window is None:
                return

            if hasattr(window, 'wm_attributes'):
                window.wm_attributes('-topmost', 0)
            if hasattr(window, 'attributes'):
                try:
                    window.attributes('-topmost', False)
                except Exception:
                    pass
            if hasattr(window, 'setWindowFlag'):
                try:
                    from matplotlib.backends.qt_compat import QtCore
                    window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, False)
                    window.setWindowFlag(QtCore.Qt.WindowDoesNotAcceptFocus, True)
                except Exception:
                    pass
        except Exception:
            pass

    def update_plot(self, _frame):
        self.ax.cla()

        if len(self.limo_states) == 0:
            self.ax.set_title('Waiting for limo_state and person_state...')
            self.ax.set_xlabel('x [m]')
            self.ax.set_ylabel('y [m]')
            self.ax.grid(True)
            return

        states = np.asarray(self.limo_states, dtype=float)
        person_states = np.asarray(self.person_states, dtype=float) if len(self.person_states) > 0 else None
        measurement = np.asarray(self.meas, dtype=float) if len(self.meas) > 0 else None
        des = np.asarray(self.des0, dtype=float) if len(self.des0) > 0 else None
        x_values = states[:, 0]
        y_values = states[:, 1]

        self.ax.plot(x_values, y_values, '-', color='tab:blue', label='trajectory')
        self.ax.plot(x_values[-1], y_values[-1], 'o', color='tab:red', label='robot')

        if person_states is not None:
            person_x_values = person_states[:, 0]
            person_y_values = person_states[:, 1]
            self.ax.plot(person_x_values, person_y_values, '-', color='tab:green', label='person trajectory')
            self.ax.plot(person_x_values[-1], person_y_values[-1], 'o', color='tab:green', label='person')

        if self.x_pred is not None:
            self.ax.plot(self.x_pred, self.y_pred, '--', color='orange')
        
        if des is not None:
            self.ax.plot(self.des0[0], self.des0[1], 'o')
            self.ax.plot(self.des1[0], self.des1[1], 'o')
            self.ax.plot(self.des2[0], self.des2[1], 'o')


        theta = states[-1, 2]
        arrow_length = 0.4
        self.ax.arrow(
            x_values[-1],
            y_values[-1],
            arrow_length * np.cos(theta),
            arrow_length * np.sin(theta),
            head_width=0.12,
            head_length=0.16,
            fc='tab:red',
            ec='tab:red',
            length_includes_head=True,
        )

        all_x_values = x_values
        all_y_values = y_values
        if person_states is not None:
            all_x_values = np.concatenate([all_x_values, person_x_values])
            all_y_values = np.concatenate([all_y_values, person_y_values])

        x_margin = max(0.5, 0.15 * max(float(np.ptp(all_x_values)), 1e-6))
        y_margin = max(0.5, 0.15 * max(float(np.ptp(all_y_values)), 1e-6))
        self.ax.set_xlim(float(np.min(all_x_values)) - x_margin, float(np.max(all_x_values)) + x_margin)
        self.ax.set_ylim(float(np.min(all_y_values)) - y_margin, float(np.max(all_y_values)) + y_margin)

        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title('EKF robot animation')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.legend(loc='upper right')
        self.ax.grid(True)

    def plot_limo_states(self):
        if len(self.limo_states) == 0:
            return

        states = np.asarray(self.limo_states, dtype=float)

        plt.figure()

        plt.plot(self.t, states[:, 0], label="x")
        plt.plot(self.t, states[:, 1], label="y")
        plt.plot(self.t, states[:, 2], label="theta")

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
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.t.append(0)
        else:
            self.t.append((self.get_clock().now() - self.start_time).nanoseconds * 1e-9)

        #if self.is_running:
            #self.get_logger().info('CALLED: limo_state_callback')

    def person_state_callback(self, msg):
        x = msg.x
        y = msg.y
        self.person_states.append((x, y))

        #if self.is_running:
            #self.get_logger().info('CALLED: person_state_callback')

    def meas_callback(self, msg):
        x = msg.x
        y = msg.y
        self.meas.append((x, y))

    def des_callback(self, msg):
        self.des0 = np.array([msg.x0, msg.y0])
        self.des1 = np.array([msg.x1, msg.y1])
        self.des2 = np.array([msg.x2, msg.y2])

    def mpc_pred_callback(self, msg):
        self.x_pred = msg.x
        self.y_pred = msg.y

def main():

    rclpy.init()

    ekf_plot = EKFPlot()
    try:
        while rclpy.ok() and plt.fignum_exists(ekf_plot.fig.number):
            rclpy.spin_once(ekf_plot, timeout_sec=0.1)
            plt.pause(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        ekf_plot.destroy_node()
        rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()