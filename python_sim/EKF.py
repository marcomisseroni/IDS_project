import numpy as np

#  ______ _  ________        _               
# |  ____| |/ /  ____|      | |              
# | |__  | ' /| |__      ___| | __ _ ___ ___ 
# |  __| |  < |  __|    / __| |/ _` / __/ __|
# | |____| . \| |      | (__| | (_| \__ \__ \
# |______|_|\_\_|       \___|_|\__,_|___/___/
                                            
                                            

class EKF:

    def __init__(self, enc_weight, imu_weight, r, b, initial_state, R, Q, dt = 0.1, dx = 0, dy = 0, dtheta = 0):

        if enc_weight + imu_weight != 1 or enc_weight < 0 or imu_weight < 0:
            print("Error: weights used are not valid, they must be positive and sum up to 1")
            self.weight_enc = 0.5
            self.weight_imu = 0.5
        else:
            self.weight_enc = enc_weight
            self.weight_imu = imu_weight
        self.r = r
        self.b = b
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


    def _kinematic_model(self, w_enc_r, w_enc_l, w_imu):
        self.v = self.r * (w_enc_l + w_enc_r) / 2
        w_enc = self.r * (w_enc_r - w_enc_l) / self.b
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
    
    def prediction_step(self, w_enc_r, w_enc_l, w_imu):
        self.state = self._kinematic_model(w_enc_r=w_enc_r, w_enc_l=w_enc_l, w_imu=w_imu)
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

    




#  _______        _                     _       
# |__   __|      | |                   (_)      
#    | | ___  ___| |_   _ __ ___   __ _ _ _ __  
#    | |/ _ \/ __| __| | '_ ` _ \ / _` | | '_ \ 
#    | |  __/\__ \ |_  | | | | | | (_| | | | | |
#    |_|\___||___/\__| |_| |_| |_|\__,_|_|_| |_|
                                               
                                               

import conf_limo

if __name__ == "__main__":

    # initial state
    initial_state = np.array([0.0, 0.0, 0.0])

    # covariance matricies
    R = conf_limo.R
    Q = conf_limo.Q

    # parametri robot
    r = conf_limo.r
    b = conf_limo.b

    # crea filtro
    ekf = EKF(
        enc_weight=0.5,
        imu_weight=0.5,
        r=r,
        b=b,
        initial_state=initial_state,
        R=R,
        Q=Q,
        dt=0.1
    )

    # simple sim
    for i in range(10):

        w_enc_r = 1.0
        w_enc_l = 1.0
        w_imu = 0.1

        ekf.prediction_step(w_enc_r, w_enc_l, w_imu)

        # fake meas
        lidar_meas = ekf.state + np.random.normal(0, 0.01, 3)

        ekf.update_step(lidar_meas)

        print("state:", ekf.state)
        print("P matrix: \n", ekf.P)