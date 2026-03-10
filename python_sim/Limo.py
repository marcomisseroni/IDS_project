import numpy as np
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

    def desired_pos(self, target_meas, target_1, target_2, state_1, state_2):
        # best target estimation --> to improve with the variances
        self.target = (target_meas + target_1 + target_2) / 3

        # ------------- new center position --------------------
        # angle between previous center and new target
        alpha = np.arctan2(self.target[1]-self.center[1], self.target[0]-self.center[0])
        # distance to move the center
        d = np.sqrt( (self.target[1]-self.center[1])**2 + (self.target[0]-self.center[0])**2 ) - conf_limo.dist
        # new center
        self.center = np.array([self.center[0]+d*np.cos(alpha), self.center[1]+d*np.sin(alpha)])

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
        return p0, p1, p2, self.center


    #def update(self, state_1, state_2):

        

 


