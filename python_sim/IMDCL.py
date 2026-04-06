import numpy as np
import random

class IMDCL:

    agents_number = 0

    def __init__(self, s0, R, Q, dt, mu, sigma, id):
        self.state = s0
        self.R = R
        self.Q = Q
        self.dt = dt
        self.F = np.eye(4, 4)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        self.P = np.eye(4, 4) * 10**-3
        self.phi = np.eye(4, 4)
        self.pi = np.zeros((4, 4))
        self.mu = mu
        self.sigma = sigma
        self.G = np.zeros((4, 2))
        self.G[2, 0] = 1
        self.G[3, 1] = 1
        self.id = id    # To identify each agent
        IMDCL.agents_number += 1

    def prediction(self):
        epx = random.gauss(self.mu, self.sigma)
        epy = random.gauss(self.mu, self.sigma)
        self.state = self.F @ self.state + np.array[0, 0, epx, epy]
        self.P = self.F @ self.P @ self.F.transpose() + self.G @ self.Q @ self.G.transpose()
        self.phi = self.F @ self.phi


if __name__ == "__main__":
    
    s0 = np.zeros(4)
    R = 1
    Q = 1
    dt = 0.1
    imdcl = IMDCL(s0, R, Q, dt)
    print(imdcl.F)
        