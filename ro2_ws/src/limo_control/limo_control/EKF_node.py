import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
import numpy as np                                            

#  ______ _  ________                   _      
# |  ____| |/ /  ____|                 | |     
# | |__  | ' /| |__     _ __   ___   __| | ___ 
# |  __| |  < |  __|   | '_ \ / _ \ / _` |/ _ \
# | |____| . \| |      | | | | (_) | (_| |  __/
# |______|_|\_\_|      |_| |_|\___/ \__,_|\___|

# This node implements an extended kalman filter to estimate the state of the limo
# Pub topuc: ekf_state (String) -> [x, y, theta]
# Sub topic: enc (String) -> [v_enc, w_enc]
#            imu (String) -> w_imu
#            lidar (String) -> [x_lidar, y_lidar, theta_lidar]
#            admin (String) -> "start" or "stop" to start or stop the estimation                                             
                                              

class EKF(Node):

    def __init__(self, enc_weight, imu_weight, initial_state, R, Q, dt = 0.3, dx = 0, dy = 0, dtheta = 0):

        super().__init__('ekf')
        if enc_weight + imu_weight != 1 or enc_weight < 0 or imu_weight < 0:
            print("Error: weights used are not valid, they must be positive and sum up to 1")
            self.weight_enc = 0.5
            self.weight_imu = 0.5
        else:
            self.weight_enc = enc_weight
            self.weight_imu = imu_weight
        # Define parameters
        self.v = 0
        self.yaw_rate = 0
        self.state = initial_state.copy()
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dtheta = dtheta
        self.A = self._A()
        self.G = self._G()
        self.H = self._H()
        self.R = R
        self.Q = Q
        self.P = np.linalg.inv(self.H.T @ np.linalg.inv(self.R) @ self.H)
        # Communication stuff
        self.publisher = self.create_publisher(Float64MultiArray, 'ekf_state', 10)
        self.subscription_e = self.create_subscription(Float64MultiArray, 'enc', self.enc_listener_callback, 10)
        self.subscription_i = self.create_subscription(Float64MultiArray, 'imu', self.imu_listener_callback, 10)
        self.subscription_l = self.create_subscription(Float64MultiArray, 'lidar', self.lidar_listener_callback, 10)
        self.subscription_a = self.create_subscription(String, 'admin', self.admin_listener_callback, 10)
        self.subscription_e # prevent unused variable warning
        self.subscription_e 
        self.subscription_i 
        self.subscription_l 
        self.subscription_a 
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.start = False
        self.new_enc_data = False
        self.new_imu_data = False
        self.new_lidar_data = False
        self.enc_v = 0
        self.enc_w = 0
        self.imu_w = 0
        self.lidar_meas = np.zeros(3)

    def timer_callback(self):
        if not self.start:
            return
        msg = Float64MultiArray()
        if self.new_enc_data and self.new_imu_data:
            self.prediction_step(self.enc_w, self.enc_v, self.imu_w)
            self.new_enc_data = False
            self.new_imu_data = False
        if self.new_lidar_data:
            self.update_step(self.lidar_meas)
            self.new_lidar_data = False
        msg.data = [self.state[0], self.state[1], self.state[2]]
        self.publisher.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

    def enc_listener_callback(self, msg):
        self.enc_v = msg.data[0]
        self.enc_w = msg.data[1]
        self.new_enc_data = True

    def imu_listener_callback(self, msg):
        self.imu_w = msg.data[0]
        self.new_imu_data = True

    def lidar_listener_callback(self, msg):
        self.lidar_meas[0] = msg.data[0]
        self.lidar_meas[1] = msg.data[1]
        self.lidar_meas[2] = msg.data[2]
        self.new_lidar_data = True

    def admin_listener_callback(self, msg):
        if msg.data == "start":
            self.start = True
        elif msg.data == "stop":
            self.start = False

    def _kinematic_model(self, w_enc, v_enc, w_imu):
        self.v = v_enc
        self.yaw_rate = self.weight_enc * w_enc + self.weight_imu * w_imu
        x = self.state[0] + self.dt * self.v * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        y = self.state[1] + self.dt * self.v * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        theta = self.state[2] + self.dt * self.yaw_rate
        return np.array([x, y, theta])
    
    def _A(self):
        A = np.identity(3)
        A[0, 2] = - self.dt * self.v * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        A[1, 2] = self.dt * self.v * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        return A
    
    def _G(self):
        G = np.zeros((3, 3))
        G[0, 0] = self.dt * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        G[0, 1] = - self.dt ** 2 * self.v / 2 * self.weight_enc * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        G[0, 2] = - self.dt ** 2 * self.v / 2 * self.weight_imu * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        G[1, 0] = self.dt * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        G[1, 1] = self.dt ** 2 * self.v / 2 * self.weight_enc * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        G[1, 2] = self.dt ** 2 * self.v / 2 * self.weight_imu * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        G[2, 0] = 0
        G[2, 2] = self.dt * self.weight_imu
        G[2, 1] = self.dt * self.weight_enc
        return G
    
    def _H(self):
        return np.identity(3)
    
    def prediction_step(self, w_enc, v_enc, w_imu):
        self.state = self._kinematic_model(w_enc=w_enc, v_enc=v_enc, w_imu=w_imu)
        self.A = self._A()
        self.G = self._G()
        self.H = self._H()
        self.P = self.A @ self.P @ self.A.T + self.G @ self.Q @ self.G.T

    def update_step(self, lidar_meas):
        lidar_meas[0] += self.dx
        lidar_meas[1] += self.dy
        lidar_meas[2] += self.dtheta
        S = self.H @ self.P @ self.H.T + self.R
        W = self.P @ self.H.T @ np.linalg.inv(S) 
        self.state += W @ (lidar_meas - self.state)
        self.P = (np.identity(3) - W @ self.H) @ self.P 

def main(args=None):
    rclpy.init(args=args)
    R = np.diag([1, 1, 1])
    Q = np.diag([0.1, 0.1, 0.05])
    ekf = EKF(enc_weight=0.5, imu_weight=0.5, initial_state=np.zeros(3), R=R, Q=Q, dt=0.3)
    rclpy.spin(ekf)
    ekf.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()