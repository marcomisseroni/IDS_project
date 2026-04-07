import numpy as np
from scipy.linalg import sqrtm
import random

class IMDCL:

    agents_number = 0

    def __init__(self, s0, R, Q, dt, mu, sigma):
        self.state = s0
        self.R = R
        self.Q = Q
        self.dt = dt
        self.F = np.eye(2, 2)
        self.F[0, 1] = self.dt
        self.P = np.eye(2, 2) * 10**-3
        self.phi = np.eye(2, 2)
        self.pi12 = np.zeros((2, 2))
        self.pi13 = np.zeros((2, 2))
        self.pi23 = np.zeros((2, 2))
        self.mu = mu
        self.sigma = sigma
        self.H = np.eye(2, 2)  # to check
        IMDCL.agents_number += 1
        self.id = IMDCL.agents_number

    def prediction(self):
        epsilon = random.gauss(self.mu, self.sigma)
        self.state = self.F @ self.state + np.array([0, epsilon]) 
        self.P = self.F @ self.P @ self.F.transpose()
        self.phi = self.F @ self.phi

    def rel_meas(self, state_b, phi_b, P_b, z_ab, H_b, id_b):
        id_a = self.id
        pi_ab = None
        if((id_a == 1 and id_b == 2) or (id_a == 2 and id_b == 1)):
            pi_ab = self.pi12
        elif((id_a == 2 and id_b == 3) or (id_a == 3 and id_b == 2)):
            pi_ab = self.pi23
        elif((id_a == 1 and id_b == 3) or (id_a == 3 and id_b == 1)):
            pi_ab = self.pi13
        if(pi_ab is None):
            print("pi_ab not assigned")
            return
        Pab = self.phi @ pi_ab @ phi_b.transpose()
        Pba = phi_b @ pi_ab @ self.phi.transpose()
        r_a = z_ab - (state_b - self.state)
        S_ab = self.R + self.H @ self.P @ self.H.transpose() + H_b @ P_b @ H_b.transpose() - self.H @ Pab @ H_b.transpose() - H_b @ Pba @ self.H.transpose()
        gamma_a = (pi_ab @ phi_b.transpose() @ H_b.transpose() - np.linalg.inv(self.phi) @ self.P @ self.H.transpose()) @ np.linalg.inv(sqrtm(S_ab))
        gamma_b = (np.linalg.inv(phi_b) @ P_b @ H_b.transpose() - pi_ab @ self.phi.transpose() @ self.H.transpose()) @ np.linalg.inv(sqrtm(S_ab))
        return r_a, gamma_a, gamma_b, phi_b.transpose() @ H_b.transpose() @ np.linalg.inv(sqrtm(S_ab)), self.phi.transpose() @ self.H.transpose() @ np.linalg.inv(sqrtm(S_ab))

    def update(self, r_a, gamma_a, gamma_b, W1, W2, id_a, id_b):
        pi_a = None
        pi_b = None
        if(id_a != 1 and id_b != 1):
            if(id_b == 2):
                pi_b = self.pi12
                pi_a = self.pi13
            if(id_b == 3):
                pi_b = self.pi13
                pi_a = self.pi12

        if(id_a != 2 and id_b != 2):
            if(id_b == 1):
                pi_b = self.pi12
                pi_a = self.pi23
            if(id_b == 3):
                pi_b = self.pi23
                pi_a = self.pi12

        if(id_a != 3 and id_b != 3):
            if(id_b == 1):
                pi_b = self.pi13
                pi_a = self.pi23
            if(id_b == 2):
                pi_b = self.pi23
                pi_a = self.pi13
        if(pi_a is None or pi_b is None):
            print("pi_a or pi_b not assigned")
            return
        gamma = pi_b @ W1 - pi_a @ W2
        self.state = self.state + self.phi @ gamma @ r_a
        self.P = self.P - self.phi @ gamma @ gamma.transpose() @ self.phi.transpose()
        # update pi jl


if __name__ == "__main__":
    
    s0 = np.zeros(2)
    R = 1
    Q = 1
    dt = 0.1
    imdcl = IMDCL(s0, R, Q, dt)
    print(imdcl.F)
        