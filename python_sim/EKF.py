import numpy as np

#  ______ _  ________        _               
# |  ____| |/ /  ____|      | |              
# | |__  | ' /| |__      ___| | __ _ ___ ___ 
# |  __| |  < |  __|    / __| |/ _` / __/ __|
# | |____| . \| |      | (__| | (_| \__ \__ \
# |______|_|\_\_|       \___|_|\__,_|___/___/
                                            
"""
Implementation of the Interim Master Decentralized Cooperative Localization (IMDCL) algorithm.

Based on:
S. S. Kia, S. Rounds, and S. Martínez,
"Cooperative Localization for Mobile Agents",
IEEE Control Systems Magazine.
"""             

class EKF:

    agent_id = 1

    def __init__(
            self, 
            initial_state: np.ndarray, 
            R: np.ndarray, 
            Q: np.ndarray, 
            dt: float):
        
        self.v = 0.0
        self.yaw_rate = 0.0
        self.state = initial_state.copy()
        self.R = R
        self.Q = Q
        self.dt = dt
        self.A = self._A()
        self.G = self._G()
        self.H = self._H()
        self.P = np.linalg.inv(self.H.T @ np.linalg.inv(self.R) @ self.H)
        self.phi = np.eye(3, 3)
        self.pi12 = np.zeros((3, 3))
        self.pi13 = np.zeros((3, 3))
        self.pi14 = np.zeros((3, 3))
        self.pi23 = np.zeros((3, 3))
        self.pi24 = np.zeros((3, 3))
        self.pi34 = np.zeros((3, 3))
        self.gamma = np.zeros((3,2))
        self.agent_id = type(self).agent_id
        type(self).agent_id += 1


    def _kinematic_model(
            self, v: float, 
            yaw_rate: float
            ) -> np.ndarray:
        
        self.v = v
        self.yaw_rate = yaw_rate
        x = self.state[0] + self.dt * self.v * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        y = self.state[1] + self.dt * self.v * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        theta = self.state[2] + self.dt * self.yaw_rate
        return np.array([x, y, theta])
    
    def _A(
            self
            ) -> np.ndarray:
        
        A = np.identity(3)
        A[0, 2] = - self.dt * self.v * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        A[1, 2] = self.dt * self.v * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        return A
    
    def _G(
            self
            ) -> np.ndarray:
        
        G = np.zeros((3, 3))
        G[0, 0] = self.dt * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        G[0, 1] = - self.dt ** 2 * self.v / 2 * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        G[1, 0] = self.dt * np.sin(self.state[2] + self.dt * self.yaw_rate / 2)
        G[1, 1] = self.dt ** 2 * self.v / 2 * np.cos(self.state[2] + self.dt * self.yaw_rate / 2)
        G[2, 1] = self.dt
        return G
    
    def _H(
            self
            ) -> np.ndarray:
        
        return np.identity(3)
    
    def prediction_step(
            self, 
            v: float, 
            yaw_rate: float):
        
        self.state = self._kinematic_model(v, yaw_rate)
        self.A = self._A()
        self.G = self._G()
        self.H = self._H()
        self.P = self.A @ self.P @ self.A.T + self.G @ self.Q @ self.G.T
        self.phi = self.A @ self.phi

    def measurement():
        return

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