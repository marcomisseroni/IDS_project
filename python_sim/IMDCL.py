import numpy as np
import random

class IMDCL:

    agents_number = 0

    def __init__(self, s0, R, Q, dt, mu, sigma):
        self.state = s0
        self.R = R  #scalar
        self.Q = Q  #2x2
        self.dt = dt
        self.F = np.eye(2, 2)
        self.F[0, 1] = self.dt
        self.G = np.eye(2, 2)
        self.G[0, 0] = 0
        self.P = np.eye(2, 2) * 10**-3
        self.phi = np.eye(2, 2)
        self.pi12 = np.zeros((2, 2))
        self.pi23 = np.zeros((2, 2))
        self.pi31 = np.zeros((2, 2))
        self.gamma = np.zeros((2,1))
        self.mu = mu
        self.sigma = sigma
        IMDCL.agents_number += 1
        self.id = IMDCL.agents_number

    def prediction(self):
        epsilon = random.gauss(self.mu, self.sigma)
        self.state = self.F @ self.state + self.G @ np.ones((2,1)) * epsilon
        self.P = self.F @ self.P @ self.F.transpose() + self.G @ self.Q @ self.G.transpose()
        self.phi = self.F @ self.phi

    def rel_meas(self, state_b, phi_b, P_b, z_ab, id_b):
        H_a1 = -(self.state[0] - state_b[0]) / np.sqrt((self.state[0] - state_b[0]) ** 2)
        H_b1 = -(self.state[0] - state_b[0]) / np.sqrt((self.state[0] - state_b[0]) ** 2)
        H_a = np.array([[H_a1, 0]])  
        H_b = np.array([[H_b1, 0]])  
        id_a = self.id
        pi_ab = None
        if((id_a == 1 and id_b == 2) or (id_a == 2 and id_b == 1)):
            pi_ab = self.pi12
        elif((id_a == 2 and id_b == 3) or (id_a == 3 and id_b == 2)):
            pi_ab = self.pi23
        elif((id_a == 1 and id_b == 3) or (id_a == 3 and id_b == 1)):
            pi_ab = self.pi31
        if(pi_ab is None):
            print("pi_ab not assigned")
            return
        Pab = self.phi @ pi_ab @ phi_b.transpose()
        Pba = phi_b @ pi_ab.transpose() @ self.phi.transpose()
        r_a = z_ab - abs(state_b[0, 0] - self.state[0, 0])
        S_ab = self.R + H_a @ self.P @ H_a.transpose() + H_b @ P_b @ H_b.transpose() - H_a @ Pab @ H_b.transpose() - H_b @ Pba @ H_a.transpose()
        gamma_a = (pi_ab @ phi_b.transpose() @ H_b.transpose() - np.linalg.inv(self.phi) @ self.P @ H_a.transpose()) * S_ab ** -0.5
        gamma_b = (np.linalg.inv(phi_b) @ P_b @ H_b.transpose() - pi_ab.transpose() @ self.phi.transpose() @ H_a.transpose()) * S_ab ** -0.5
        W1 = phi_b.transpose() @ H_b.transpose() * S_ab ** -0.5
        W2 = self.phi.transpose() @ H_a.transpose() * S_ab ** -0.5
        return r_a * S_ab ** -0.5, gamma_a, gamma_b, W1, W2

    def update(self, r_a, gamma_a, gamma_b, W1, W2, id_a, id_b):
        pi_a = None
        pi_b = None
        gamma1 = None
        gamma2 = None
        gamma3 = None
        if(id_a != 1 and id_b != 1):
            if(id_b == 2):
                pi_b = self.pi12
                pi_a = self.pi31
                gamma1 = pi_b @ W1 - pi_a @ W2
                gamma2 = gamma_b
                gamma3 = gamma_a
            if(id_b == 3):
                pi_b = self.pi31
                pi_a = self.pi12
                gamma1 = pi_b @ W1 - pi_a @ W2
                gamma2 = gamma_a
                gamma3 = gamma_b

        if(id_a != 2 and id_b != 2):
            if(id_b == 1):
                pi_b = self.pi12
                pi_a = self.pi23
                gamma2 = pi_b @ W1 - pi_a @ W2
                gamma1 = gamma_b
                gamma3 = gamma_a
            if(id_b == 3):
                pi_b = self.pi23
                pi_a = self.pi12
                gamma2 = pi_b @ W1 - pi_a @ W2
                gamma1 = gamma_a
                gamma3 = gamma_b

        if(id_a != 3 and id_b != 3):
            if(id_b == 1):
                pi_b = self.pi31
                pi_a = self.pi23
                gamma3 = pi_b @ W1 - pi_a @ W2
                gamma1 = gamma_b
                gamma2 = gamma_a
            if(id_b == 2):
                pi_b = self.pi23
                pi_a = self.pi31
                gamma3 = pi_b @ W1 - pi_a @ W2
                gamma2 = gamma_b
                gamma1 = gamma_a
        if(pi_a is None or pi_b is None):
            print("pi_a or pi_b not assigned")
            return
        if(gamma1 is None or gamma2 is None or gamma3 is None):
            print("Gamma not assiciated")
            return
        if(self.id == 1):
            self.gamma = gamma1
        elif(self.id == 2):
            self.gamma = gamma2
        elif(self.id == 3):
            self.gamma = gamma3
        self.state = self.state + self.phi @ (self.gamma * r_a)
        self.P = self.P - self.phi @ self.gamma @ self.gamma.transpose() @ self.phi.transpose()   #to check
        self.pi12 = self.pi12 - gamma1 @ gamma2.transpose()
        self.pi23 = self.pi23 - gamma2 @ gamma3.transpose()
        self.pi31 = self.pi31 - gamma1 @ gamma3.transpose()


if __name__ == "__main__":
    
    s1 = np.array([0, 1]).reshape(-1, 1)
    s2 = np.array([1, 1]).reshape(-1, 1)
    s3 = np.array([2, 1]).reshape(-1, 1)
    R = 1
    Q = np.eye(2, 2)
    dt = 0.1
    mu = 0
    sigma = 0.1
    N_sim = 50
    agent1 = IMDCL(s1, R, Q, dt, mu, sigma)
    agent2 = IMDCL(s2, R, Q, dt, mu, sigma)
    agent3 = IMDCL(s3, R, Q, dt, mu, sigma)
    relative_meas_flag = 1
    for i in range(N_sim):
        agent1.prediction()
        agent2.prediction()
        agent3.prediction()
        if(relative_meas_flag == 1):
            relative_meas_flag += 1
            epsilon = random.gauss(mu, sigma)
            r_a, gamma_a, gamma_b, W1, W2 = agent1.rel_meas(agent2.state, agent2.phi, agent2.P, agent2.state[0] - agent1.state[0] + epsilon, 2)
            agent1.update(r_a, gamma_a, gamma_b, W1, W2, 1, 2)
            agent2.update(r_a, gamma_a, gamma_b, W1, W2, 1, 2)
            agent3.update(r_a, gamma_a, gamma_b, W1, W2, 1, 2)

        if(relative_meas_flag == 2):
            relative_meas_flag += 1
            epsilon = random.gauss(mu, sigma)
            r_a, gamma_a, gamma_b, W1, W2 = agent2.rel_meas(agent3.state, agent3.phi, agent3.P, agent3.state[0] - agent2.state[0] + epsilon, 3)
            agent1.update(r_a, gamma_a, gamma_b, W1, W2, 2, 3)
            agent2.update(r_a, gamma_a, gamma_b, W1, W2, 2, 3)
            agent3.update(r_a, gamma_a, gamma_b, W1, W2, 2, 3)

        if(relative_meas_flag == 3):
            relative_meas_flag = 1
            epsilon = random.gauss(mu, sigma)
            r_a, gamma_a, gamma_b, W1, W2 = agent2.rel_meas(agent1.state, agent1.phi, agent1.P, agent1.state[0] - agent2.state[0] + epsilon, 1)
            agent1.update(r_a, gamma_a, gamma_b, W1, W2, 3, 1)
            agent2.update(r_a, gamma_a, gamma_b, W1, W2, 2, 1)
            agent3.update(r_a, gamma_a, gamma_b, W1, W2, 3, 1)

        print("Agent1: ", agent1.state.flatten())
        print("Agent2: ", agent2.state.flatten())
        print("Agent3: ", agent3.state.flatten())
        
        