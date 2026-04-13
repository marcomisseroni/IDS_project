import numpy as np
from scipy.linalg import sqrtm

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

    agent_id = 0

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
        self.Ha = np.eye(3, 3)
        self.Hb = np.eye(3, 3)
        self.P = np.eye(3, 3)
        self.phi = np.eye(3, 3)
        self.cross_cov = {}
        self.gamma = np.zeros((3,3))
        type(self).agent_id += 1
        self.agent_id = type(self).agent_id
        self.is_initialized = False


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
    
    def _Ha(
            self,
            b_state: np.ndarray):
        
        Ha = np.zeros((3, 3))
        Ha[0, 0] = np.cos(self.state[2])
        Ha[0, 1] = np.sin(self.state[2])
        Ha[0, 2] = (self.state[1] - b_state[1]) * np.cos(self.state[2]) + (b_state[0] - self.state[0]) * np.sin(self.state[2])
        Ha[1, 0] = - np.sin(self.state[2])
        Ha[1, 1] = np.cos(self.state[2])
        Ha[1, 2] = (b_state[0] - self.state[0]) * np.cos(self.state[2]) + (b_state[1] - self.state[1]) * np.sin(self.state[2])
        Ha[2, 2] = 1
        self.Ha = Ha
    
    def _Hb(
            self):
    
        Hb = np.zeros((3, 3))
        Hb[0, 0] = np.cos(self.state[2])
        Hb[0, 1] = np.sin(self.state[2])
        Hb[1, 0] = - np.sin(self.state[2])
        Hb[1, 1] = np.cos(self.state[2])
        Hb[2, 2] = 1
        self.Hb = Hb

    def h(
            self,
            b_state: np.ndarray
            ) -> np.ndarray:
        delta_x = (b_state[0] - self.state[0]) * np.cos(self.state[2]) + (b_state[1] - self.state[1]) * np.sin(self.state[2])
        delta_y = (b_state[1] - self.state[1]) + np.cos(self.state[2]) + (self.state[0] - b_state[0]) * np.sin(self.state[2])
        delta_theta = b_state[2] - self.state[2]

        return np.array([delta_x, delta_y, delta_theta])
    
    def prediction_step(
            self, 
            v: float, 
            yaw_rate: float):
        
        if(not self.is_initialized):
            self.is_initialized = True
            N = type(self).agent_id
            dim = (N ** 2 - N) // 2
            payload = [np.zeros((3, 3)) for _ in range(dim)]
            self.cross_cov_set(payload) 

        self.state = self._kinematic_model(v, yaw_rate)
        self.A = self._A()
        self.G = self._G()
        self.H = self._H()
        self.P = self.A @ self.P @ self.A.T + self.G @ self.Q @ self.G.T
        self.phi = self.A @ self.phi

    def measurement(
            self,
            b_state: np.ndarray,
            phi_b: np.ndarray,
            Pb: np.ndarray,
            z_ab: np.ndarray,
            id_b: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        self._Ha(b_state)
        self._Hb()
        piab = self.cross_cov[(min(self.agent_id, id_b), max(self.agent_id, id_b))]

        if self.agent_id > id_b: 
            piab = piab.T
        
        Pab = self.phi @ piab @ phi_b.T
        Pba = phi_b @ piab.T @ self.phi.T
        r_a = z_ab - self.h(b_state)
        S_ab = self.R + self.Ha @ self.P @ self.Ha.T + self.Hb @ Pb @ self.Hb.T - self.Ha @ Pab @ self.Hb.T - self.Hb @ Pba @ self.Ha.T
        inv_sqrt_S = sqrtm(np.linalg.inv(S_ab))
        gamma_a = (piab @ phi_b.T @ self.Hb.T - np.linalg.inv(self.phi) @ self.P @ self.Ha.T) @ inv_sqrt_S
        gamma_b = (np.linalg.inv(phi_b) @ Pb @ self.Hb.T - piab.T @ self.phi.T @ self.Ha.T) @ inv_sqrt_S
        W1 = phi_b.T @ self.Hb.T @ inv_sqrt_S
        W2 = self.phi.T @ self.Ha.T @ inv_sqrt_S

        return inv_sqrt_S @ r_a, gamma_a, gamma_b, W1, W2

    def update_step(
            self,
            r_a: np.ndarray,
            gamma_a: np.ndarray,
            gamma_b: np.ndarray,
            W1: np.ndarray,
            W2: np.ndarray,
            id_a: int,
            id_b: int):
        N = type(self).agent_id
        gamma = [None] * N

        for j in range(1, N + 1):
            if(j == id_a):
                gamma[j - 1] = gamma_a
            
            elif(j == id_b):
                gamma[j - 1] = gamma_b

            else:
                pija = self.cross_cov[(min(j, id_a), max(j, id_a))]
                pijb = self.cross_cov[(min(j, id_b), max(j, id_b))]
                if j > id_a:
                    pija = pija.T

                if j > id_b:
                    pijb = pijb.T

                gamma[j - 1] = pijb @ W1 - pija @ W2

        self.gamma = gamma[self.agent_id - 1]
        self.state = self.state + self.phi @ self.gamma @ r_a
        self.P = self.P - self.phi @ self.gamma @ self.gamma.T @ self.phi.T
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):

                self.cross_cov[(i, j)] -= gamma[i - 1] @ gamma[j - 1].T

    
    def cross_cov_set(
            self,
            payload: list[np.ndarray]):
        
        N = type(self).agent_id
        exp_len = (N ** 2 - N) // 2
        if(len(payload) != exp_len):
            print("Error: payload length is different from: ", exp_len)
            return
        
        k = 0
        for i in range(1, N + 1):
            for j in range(i + 1, N + 1):

                self.cross_cov[(i, j)] = payload[k]
                k += 1
