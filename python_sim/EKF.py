import numpy as np

class EKF:

    def __init__(self, enc_weight, imu_weight, r, b, initial_state, dt = 0.1):

        if enc_weight + imu_weight != 1 or enc_weight < 0 or imu_weight < 0:
            print("Error: weights used are not valid, they must be positive and sum up to 1")
            self.w_enc = 0.5
            self.w_imu = 0.5
        else:
            self.w_enc = enc_weight
            self.w_imu = imu_weight
        self.r = r
        self.b = b
        self.v = 0
        self.yaw_rate = 0
        self.state = initial_state.copy()
        self.dt = dt

    def _kinematic_model(self, w_enc_r, w_enc_l, w_imu):
        self.v = self.r * (w_enc_l + w_enc_r) / 2
        w_enc = self.r * (w_enc_r - w_enc_l) / self.b
        self.yaw_rate = self.w_enc * w_enc + self.w_imu * w_imu
        x = self.state[0] + self.dt * self.v * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        y = self.state[1] + self.dt * self.v * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        theta = self.state[2] + self.dt * self.yaw_rate
        return [x, y, theta]