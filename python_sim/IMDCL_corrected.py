import numpy as np
import random
import matplotlib.pyplot as plt

class IMDCL:

    agents_number = 0

    def __init__(self, s0, R, Q, P, dt, mu, sigma, damping=0.95):
        self.state = s0
        self.R = R  #scalar
        self.Q = Q  #2x2
        self.dt = dt
        self.damping = damping  # velocity damping factor (0 < damping < 1)
        self.F = np.eye(2, 2)
        self.F[0, 1] = self.dt
        self.F[1, 1] = damping  # velocity decays with factor 'damping'
        self.G = np.eye(2, 2)
        self.G[0, 0] = 0
        self.P = P
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
        diff = float(self.state[0, 0] - state_b[0, 0])
        abs_diff = abs(diff)
        if abs_diff < 1e-9:
            return None
        H_a1 = diff / abs_diff
        H_b1 = -H_a1
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
        S_ab = S_ab.item()
        if not np.isfinite(S_ab) or S_ab <= 0.0:
            return None
        inv_sqrt_S = S_ab ** -0.5
        gamma_a = (pi_ab @ phi_b.transpose() @ H_b.transpose() - np.linalg.inv(self.phi) @ self.P @ H_a.transpose()) * inv_sqrt_S
        gamma_b = (np.linalg.inv(phi_b) @ P_b @ H_b.transpose() - pi_ab.transpose() @ self.phi.transpose() @ H_a.transpose()) * inv_sqrt_S
        W1 = phi_b.transpose() @ H_b.transpose() * inv_sqrt_S
        W2 = self.phi.transpose() @ H_a.transpose() * inv_sqrt_S
        return r_a * inv_sqrt_S, gamma_a, gamma_b, W1, W2

    def update(self, r_a, gamma_a, gamma_b, W1, W2, id_a, id_b):
        pi_ja = None
        pi_jb = None
        gamma1 = None
        gamma2 = None
        gamma3 = None
        if(id_a != 1 and id_b != 1):
            if(id_b == 2):
                pi_jb = self.pi12
                pi_ja = self.pi31.transpose()
                gamma2 = gamma_b
                gamma3 = gamma_a
            if(id_b == 3):
                pi_jb = self.pi31.transpose()
                pi_ja = self.pi12
                gamma2 = gamma_a
                gamma3 = gamma_b
            gamma1 = pi_jb @ W1 - pi_ja @ W2

        if(id_a != 2 and id_b != 2):
            if(id_b == 1):
                pi_jb = self.pi12.transpose()
                pi_ja = self.pi23
                gamma1 = gamma_b
                gamma3 = gamma_a
            if(id_b == 3):
                pi_jb = self.pi23
                pi_ja = self.pi12.transpose()
                gamma1 = gamma_a
                gamma3 = gamma_b
            gamma2 = pi_jb @ W1 - pi_ja @ W2

        if(id_a != 3 and id_b != 3):
            if(id_b == 1):
                pi_jb = self.pi31
                pi_ja = self.pi23.transpose()
                gamma1 = gamma_b
                gamma2 = gamma_a
            if(id_b == 2):
                pi_jb = self.pi23.transpose()
                pi_ja = self.pi31
                gamma2 = gamma_b
                gamma1 = gamma_a
            gamma3 = pi_jb @ W1 - pi_ja @ W2
        if(pi_ja is None or pi_jb is None):
            print("pi_ja or pi_jb not assigned")
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
        self.pi31 = self.pi31 - gamma3 @ gamma1.transpose()


if __name__ == "__main__":
    
    s1 = np.array([0, 1]).reshape(-1, 1)
    s2 = np.array([1, 1]).reshape(-1, 1)
    s3 = np.array([2, 1]).reshape(-1, 1)
    R = 1
    Q = np.eye(2, 2)
    P1 = np.eye(2, 2) * 10**-3
    P2 = np.eye(2, 2) * 10**-2
    P3 = np.eye(2, 2) * 10**-1
    dt = 0.1
    mu = 0
    sigma1 = 0.001
    sigma2 = 0.01
    sigma3 = 0.1
    N_sim = 500
    agent1 = IMDCL(s1, R, Q, P1, dt, mu, sigma1, damping=0.98)
    agent2 = IMDCL(s2, R, Q, P2, dt, mu, sigma2, damping=0.98)
    agent3 = IMDCL(s3, R, Q, P3, dt, mu, sigma3, damping=0.98)
    relative_meas_flag = 1
    state_hist = []
    P_trace = []
    for i in range(N_sim):
        agent1.prediction()
        agent2.prediction()
        agent3.prediction()
        if(relative_meas_flag == 1):
            relative_meas_flag += 1
            epsilon = random.gauss(mu, sigma1)
            meas_out = agent1.rel_meas(agent2.state, agent2.phi, agent2.P, agent2.state[0] - agent1.state[0] + epsilon, 2)
            if meas_out is not None:
                r_a, gamma_a, gamma_b, W1, W2 = meas_out
                agent1.update(r_a, gamma_a, gamma_b, W1, W2, 1, 2)
                agent2.update(r_a, gamma_a, gamma_b, W1, W2, 1, 2)
                agent3.update(r_a, gamma_a, gamma_b, W1, W2, 1, 2)

        if(relative_meas_flag == 2):
            relative_meas_flag += 1
            epsilon = random.gauss(mu, sigma2)
            meas_out = agent2.rel_meas(agent3.state, agent3.phi, agent3.P, agent3.state[0] - agent2.state[0] + epsilon, 3)
            if meas_out is not None:
                r_a, gamma_a, gamma_b, W1, W2 = meas_out
                agent1.update(r_a, gamma_a, gamma_b, W1, W2, 2, 3)
                agent2.update(r_a, gamma_a, gamma_b, W1, W2, 2, 3)
                agent3.update(r_a, gamma_a, gamma_b, W1, W2, 2, 3)

        if(relative_meas_flag == 3):
            relative_meas_flag = 1
            epsilon = random.gauss(mu, sigma3)
            meas_out = agent3.rel_meas(agent1.state, agent1.phi, agent1.P, agent1.state[0] - agent3.state[0] + epsilon, 1)
            if meas_out is not None:
                r_a, gamma_a, gamma_b, W1, W2 = meas_out
                agent1.update(r_a, gamma_a, gamma_b, W1, W2, 3, 1)
                agent2.update(r_a, gamma_a, gamma_b, W1, W2, 3, 1)
                agent3.update(r_a, gamma_a, gamma_b, W1, W2, 3, 1)

        state_hist.append([agent1.state[0, 0], agent2.state[0, 0], agent3.state[0, 0]])
        P_trace.append([np.trace(agent1.P), np.trace(agent2.P), np.trace(agent3.P)])

    state_hist = np.array(state_hist)
    P_trace = np.array(P_trace)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(state_hist[:, 0], label="Agent 1", linewidth=2)
    ax1.plot(state_hist[:, 1], label="Agent 2", linewidth=2)
    ax1.plot(state_hist[:, 2], label="Agent 3", linewidth=2)
    ax1.set_title("Position Evolution (damping={})".format(agent1.damping))
    ax1.set_xlabel("time step")
    ax1.set_ylabel("position (m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogy(P_trace[:, 0], label="Agent 1", linewidth=2)
    ax2.semilogy(P_trace[:, 1], label="Agent 2", linewidth=2)
    ax2.semilogy(P_trace[:, 2], label="Agent 3", linewidth=2)
    ax2.set_title("Covariance Trace Evolution (log scale)")
    ax2.set_xlabel("time step")
    ax2.set_ylabel("trace(P)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('imdcl_evolution.png', dpi=150)
    print("Plot saved to imdcl_evolution.png")
    plt.show()
