import numpy as np
from scipy.linalg import sqrtm
from localization.meas_model import MeasurementModel
from localization.agent_type import AgentType
from localization.agent_type import MobileRobotModel
from localization.agent_type import PersonModel

#  ______ _  ________ 
# |  ____| |/ /  ____|
# | |__  | ' /| |__   
# |  __| |  < |  __|  
# | |____| . \| |     
# |______|_|\_\_|     
                     
                     

"""
Implementation of the Interim Master Decentralized Cooperative Localization (IMDCL) algorithm.

Based on:
S. S. Kia, S. Rounds, and S. Martínez,
"Cooperative Localization for Mobile Agents",
IEEE Control Systems Magazine.
"""       

class EKF:

    agent_id = 0
    agent_dims = {}

    def __init__(
            self, 
            initial_state: np.ndarray, 
            R_rr: np.ndarray,
            R_rp: np.ndarray, 
            Q: np.ndarray, 
            dt: float,
            agent_type: AgentType):
        
        self.agent_type = agent_type
        if self.agent_type == AgentType.ROBOT:
            self.model = MobileRobotModel()
        elif self.agent_type == AgentType.PERSON:
            self.model = PersonModel()
        else:
            raise ValueError(f"Unsupported agent_type: {self.agent_type}")
        
        self.meas_model = MeasurementModel(R_rr, R_rp)
        self.u = np.array([0, 0])
        self.state = initial_state.copy()
        self.Q = Q
        self.dt = dt
        self.n = len(self.state)
        self.A = self.model.A(self.state, self.u, self.dt)
        self.G = self.model.G(self.state, self.u, self.dt)
        self.Ha = None
        self.Hb = None
        self.P = np.identity(self.n)
        self.phi = np.identity(self.n)
        self.cross_cov = {}
        self.gamma = None
        self.agent_id = type(self).agent_id
        type(self).agent_dims[self.agent_id] = self.n 
        type(self).agent_id += 1
        self.is_initialized = False

    def _cross_cov_set(self):

            N = type(self).agent_id
            dims = type(self).agent_dims
            
            k = 0
            for i in range(N):
                for j in range(i + 1, N):
                    shape = (dims[i], dims[j])
                    self.cross_cov[(i, j)] = np.zeros(shape)
                    k += 1

    def prediction_step(
            self, 
            u: np.ndarray):
        
        self.u = u
        if(not self.is_initialized):
            self.is_initialized = True
            self._cross_cov_set() 

        self.A = self.model.A(self.state, self.u, self.dt)
        self.G = self.model.G(self.state, self.u, self.dt)
        self.state = self.model.f(self.state, self.u, self.dt)
        self.P = self.A @ self.P @ self.A.T + self.G @ self.Q @ self.G.T
        self.phi = self.A @ self.phi

    def measurement(
            self,
            b_state: np.ndarray,
            phi_b: np.ndarray,
            Pb: np.ndarray,
            z_ab: np.ndarray,
            id_b: int,
            b_agent_type: AgentType
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if self.agent_type != AgentType.ROBOT:
            raise RuntimeError("measurement() is supported only for robot observers")

        if not self.cross_cov:
            self._cross_cov_set()

        self.Ha = self.meas_model.Ha(b_agent_type, b_state, self.state)
        self.Hb = self.meas_model.Hb(b_agent_type, b_state, self.state)
        piab = self.cross_cov[(min(self.agent_id, id_b), max(self.agent_id, id_b))]

        if self.agent_id > id_b: 
            piab = piab.T
        
        R = self.meas_model.R(b_agent_type)
        Pab = self.phi @ piab @ phi_b.T
        Pba = phi_b @ piab.T @ self.phi.T
        r_a = z_ab - self.meas_model.h(b_agent_type, b_state, self.state)
        S_ab = R + self.Ha @ self.P @ self.Ha.T + self.Hb @ Pb @ self.Hb.T - self.Ha @ Pab @ self.Hb.T - self.Hb @ Pba @ self.Ha.T
        inv_sqrt_S = np.real(sqrtm(np.linalg.inv(S_ab)))
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

        for j in range(N):
            if(j == id_a):
                gamma[j] = gamma_a
            
            elif(j == id_b):
                gamma[j] = gamma_b

            else:
                pija = self.cross_cov[(min(j, id_a), max(j, id_a))]
                pijb = self.cross_cov[(min(j, id_b), max(j, id_b))]
                if j > id_a:
                    pija = pija.T

                if j > id_b:
                    pijb = pijb.T

                gamma[j] = pijb @ W1 - pija @ W2

        self.gamma = gamma[self.agent_id]
        self.state = self.state + self.phi @ self.gamma @ r_a
        self.P = self.P - self.phi @ self.gamma @ self.gamma.T @ self.phi.T
        for i in range(N):
            for j in range(i + 1, N):

                self.cross_cov[(i, j)] -= gamma[i] @ gamma[j].T