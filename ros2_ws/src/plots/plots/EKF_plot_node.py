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
        self.limo0_states = []
        self.limo1_states = []
        self.limo2_states = []

        self.person_states = []

        self.x_pred0 = []
        self.y_pred0 = []
        self.x_pred1 = []
        self.y_pred1 = []
        self.x_pred2 = []
        self.y_pred2 = []

        self.des0 = []
        self.des1 = []
        self.des2 = []

        self.is_running = True

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.animation = FuncAnimation(self.fig, self.update_plot, interval=100, cache_frame_data=False)
        self._make_window_non_intrusive()
        plt.show(block=False)

        # subs
        self.sub_admin = self.create_subscription(String, '/admin', self.admin_callback, 1)
        # ekf subscription for the three robots
        self.sub_ekf_limo = self.create_subscription(State, '/limo_state', self.limo_state_callback, 10)
        self.sub_ekf_person = self.create_subscription(State, '/person_state', self.person_state_callback, 10)
        # mpc subscription for the three limos
        self.sub_mpc_pred0 = self.create_subscription(MPCprediction, '/limo_0/mpc_prediction', lambda msg: self.mpc_pred_callback(msg, "limo0"), 10)
        self.sub_mpc_pred1 = self.create_subscription(MPCprediction, '/limo_1/mpc_prediction', lambda msg: self.mpc_pred_callback(msg, "limo1"), 10)
        self.sub_mpc_pred2 = self.create_subscription(MPCprediction, '/limo_2/mpc_prediction', lambda msg: self.mpc_pred_callback(msg, "limo2"), 10)
        # desired state calculated by limo0
        self.sub_des = self.create_subscription(Desired, '/limo_0/desired', self.des_callback, 10)

    def admin_callback(self, msg):
        if(msg.data == 'start_ekf'):
            self.is_running = True
        elif(msg.data == 'stop_ekf'):
            self.is_running = False

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
        if self.is_running == False:
            return

        self.ax.cla()

        states0 = np.asarray(self.limo0_states, dtype=float) if len(self.limo0_states) > 0 else None
        states1 = np.asarray(self.limo1_states, dtype=float) if len(self.limo1_states) > 0 else None
        states2 = np.asarray(self.limo2_states, dtype=float) if len(self.limo2_states) > 0 else None
        person_states = np.asarray(self.person_states, dtype=float) if len(self.person_states) > 0 else None
        des = np.asarray(self.des0, dtype=float) if len(self.des0) > 0 else None

        # limo0 plot
        arrow = 0.2
        if states0 is not None:
            self.ax.plot(states0[:, 0], states0[:, 1], '-', color='tab:blue', alpha=0.2)
            self.ax.plot(states0[-1, 0],states0[-1, 1], 'o', color='tab:blue', label='limo0')
            self.ax.quiver(states0[-1, 0], states0[-1, 1], arrow*np.cos(states0[-1, 2]), arrow*np.sin(states0[-1, 2]), angles='xy', scale_units='xy', scale=0.5, color='tab:blue')
            #self.get_logger().info(f'limo0 {states0[-1,0]}')
        # limo1 plot
        if states1 is not None:
            self.ax.plot(states1[:, 0], states1[:, 1], '-', color='tab:orange', alpha=0.2)
            self.ax.plot(states1[-1, 0],states1[-1, 1], 'o', color='tab:orange', label='limo1')
            self.ax.quiver(states1[-1, 0], states1[-1, 1], arrow*np.cos(states1[-1, 2]), arrow*np.sin(states1[-1, 2]), angles='xy', scale_units='xy', scale=0.5, color='tab:orange')
            #self.get_logger().info(f'limo1 {states1[-1,0]}')
        # limo2 plot
        if states2 is not None: 
            self.ax.plot(states2[:, 0], states2[:, 1], '-', color='tab:green', alpha=0.2)
            self.ax.plot(states2[-1, 0],states2[-1, 1], 'o', color='tab:green', label='limo2')
            self.ax.quiver(states2[-1, 0], states2[-1, 1], arrow*np.cos(states2[-1, 2]), arrow*np.sin(states2[-1, 2]), angles='xy', scale_units='xy', scale=0.5, color='tab:green')
            #self.get_logger().info(f'limo2 {states2[-1,0]}')

        # target plot
        if person_states is not None:
            person_x_values = person_states[:, 0]
            person_y_values = person_states[:, 1]
            self.ax.plot(person_x_values, person_y_values, '-', color='tab:red', alpha=0.2)
            self.ax.plot(person_x_values[-1], person_y_values[-1], 'o', color='tab:red', label='person')

        # mpc predictions
        if self.x_pred0 is not None:
            self.ax.plot(self.x_pred0, self.y_pred0, '--', color='cyan')
        if self.x_pred1 is not None:
            self.ax.plot(self.x_pred1, self.y_pred1, '--', color='pink')
        if self.x_pred2 is not None:
            self.ax.plot(self.x_pred2, self.y_pred2, '--', color='olive')
        
        if des is not None:
            self.ax.plot(self.des0[0], self.des0[1], 'o', color='gray')
            self.ax.plot(self.des1[0], self.des1[1], 'o', color='gray')
            self.ax.plot(self.des2[0], self.des2[1], 'o', color='gray')

        '''
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
        '''

        self.ax.set_xlim(-2, 5)
        self.ax.set_ylim(-2, 2)

        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_title('EKF robot animation')
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('y [m]')
        self.ax.grid(True)

    def limo_state_callback(self, msg):
        x = msg.x
        y = msg.y
        theta = msg.theta
        if msg.id == 0:
            self.limo0_states.append((x, y, theta))
        elif msg.id == 1:
            self.limo1_states.append((x, y, theta))
        elif msg.id == 2:
            self.limo2_states.append((x, y, theta))

    def person_state_callback(self, msg):
        x = msg.x
        y = msg.y
        self.person_states.append((x, y))

    def des_callback(self, msg):
        self.des0 = np.array([msg.x0, msg.y0])
        self.des1 = np.array([msg.x1, msg.y1])
        self.des2 = np.array([msg.x2, msg.y2])

    def mpc_pred_callback(self, msg, limo):
        if limo == "limo0":
            self.x_pred0 = msg.x
            self.y_pred0 = msg.y
        elif limo == "limo1":
            self.x_pred1 = msg.x
            self.y_pred1 = msg.y
        else:
            self.x_pred2 = msg.x
            self.y_pred2 = msg.y

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