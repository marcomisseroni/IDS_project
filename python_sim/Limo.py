import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import time
import conf_limo
import EKF
import MPC

class Limo:
    def __init__(self, state_init, target_init):
        # weight for the kalman filter
        enc_weigth = 0.5
        imu_weigth = 0.5
        # MPC and kalman timestep
        self.dt = 0.1
        # target position and previous center position
        self.target = target_init
        self.center = np.array([0,0])
        # create the EKF object
        self.ekf = EKF.EKF(enc_weigth, imu_weigth, conf_limo.r, conf_limo.b, state_init, conf_limo.R, conf_limo.Q, self.dt)
        # create the MPC object
        self.mpc = MPC.MPC(state_init,self.dt)
        self.mpc.create_OCP_problem()
        self.sol = np.zeros(conf_limo.N_sim)
        self.r = conf_limo.r_collision
        self.frame_movement = np.array([0,0])

    def desired_pos(self, target_meas, target_1, target_2, state_1, state_2):
        # best target estimation --> to improve with the variances
        self.target = (target_meas + target_1 + target_2) / 3

        # ------------- new center position --------------------
        # angle between previous center and new target
        alpha = np.arctan2(self.target[1]-self.center[1], self.target[0]-self.center[0])
        # distance to move the center
        d = np.sqrt( (self.target[1]-self.center[1])**2 + (self.target[0]-self.center[0])**2 ) - conf_limo.dist
        # new center
        old_center = self.center
        self.center = np.array([self.center[0]+d*np.cos(alpha), self.center[1]+d*np.sin(alpha)])

        self.frame_movement = self.center[:2] - old_center[:2]

        # -------------- position of each limo -----------------
        # three possible positions
        # - p0: along the target direction
        alpha0 = np.arctan2(self.target[1]-self.center[1], self.target[0]-self.center[0])
        p0 = np.array([self.center[0]+conf_limo.r_circle*np.cos(alpha0), self.center[1]+conf_limo.r_circle*np.sin(alpha0)])
        # - p1: rotated by 120° clockwise
        alpha1 = alpha0 + np.pi*2/3
        p1 = np.array([self.center[0]+conf_limo.r_circle*np.cos(alpha1), self.center[1]+conf_limo.r_circle*np.sin(alpha1)])
        # - p2: rotated by 120° counterclockwise
        alpha2 = alpha0 - np.pi*2/3
        p2 = np.array([self.center[0]+conf_limo.r_circle*np.cos(alpha2), self.center[1]+conf_limo.r_circle*np.sin(alpha2)])

        # -------------- choice of the position of each limo -----------------
        positions = np.array([p0, p1, p2])
        limo_positions = np.array([self.ekf.state[:2], state_1[:2], state_2[:2]])
        cost_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                cost_matrix[i, j] = np.linalg.norm(limo_positions[i] - positions[j])

        # solve the assignment problem: rows vector contains the limo indices, cols vector contains the position indices
        # example: rows = [0, 1, 2], cols = [2, 0, 1] so limo0 --> p2, limo1 --> p0, limo2 --> p1
        rows, cols = linear_sum_assignment(cost_matrix)
        
        return p0, p1, p2, self.center, np.array([positions[cols[0]][0], positions[cols[0]][1], 0])


    def mpc_sim(self, state_1, state_2, desired_state):
        self.sol, state = self.mpc.MPC_step(self.sol, self.ekf.state, desired_state, state_1, state_2, self.r, self.target)
        inputs = self.sol.value(self.mpc.U[0])
        return inputs
    
    def frame_update(self):
        self.ekf.state[:2] -= self.frame_movement
        self.center -= self.frame_movement

        
