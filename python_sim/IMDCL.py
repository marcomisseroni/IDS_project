import numpy as np
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

    def prediction(self):
        epsilon = random.gauss(self.mu, self.sigma)
        self.state = self.F @ self.state + np.array([0, epsilon]) 
        self.P = self.F @ self.P @ self.F.transpose()
        self.phi = self.F @ self.phi

    def rel_meas(self, state_b, phi_b, P_b, z_ab, H_b):
        pi_ab = self.pi12
        Pab = self.phi @ pi_ab @ phi_b.transpose()
        Pba = phi_b @ pi_ab @ self.phi.transpose()
        r_a = z_ab - (state_b - self.state)
        S_ab = self.R + self.H @ self.P @ self.H.transpose() + H_b @ P_b @ H_b.transpose() - self.H @ Pab @ H_b.transpose() - H_b @ Pba @ self.H.transpose()
        gamma_a = (pi_ab @ phi_b.transpose() @ H_b.transpose() - self.phi.inverse() @ self.P @ self.H.transpose()) @ S_ab **-0.5
        gamma_b = (phi_b.inverse() @ P_b @ H_b.transpose() - pi_ab @ self.phi.transpose() @ self.H.transpose()) @ S_ab **-0.5
        return r_a, gamma_a, gamma_b, phi_b.transpose() @ H_b.transpose() @ S_ab**-0-5, self.phi.transpose() @ self.H.transpose() @ S_ab **-0.5

    #def update(self, r_a, gamma_a, gamma_b, W1, W2, a_id, b_id):


if __name__ == "__main__":
    
    s0 = np.zeros(2)
    R = 1
    Q = 1
    dt = 0.1
    imdcl = IMDCL(s0, R, Q, dt)
    print(imdcl.F)
        