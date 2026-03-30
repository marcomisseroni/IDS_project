import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np                                            

class EKF(Node):

    def __init__(self, enc_weight, imu_weight, initial_state, R, Q, dt = 0.1, dx = 0, dy = 0, dtheta = 0):

        super().__init__('ekf')
        self.publisher = self.create_publisher(String, 'ekf_state', 10)
        if enc_weight + imu_weight != 1 or enc_weight < 0 or imu_weight < 0:
            print("Error: weights used are not valid, they must be positive and sum up to 1")
            self.weight_enc = 0.5
            self.weight_imu = 0.5
        else:
            self.weight_enc = enc_weight
            self.weight_imu = imu_weight
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
        self.P = (np.identity(3) - W) @ self.H @ self.P 


if __name__ == "__main__":
