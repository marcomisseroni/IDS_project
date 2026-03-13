import numpy as np
import conf_limo

#     _____        _              _                  _               
#    |  __ \      | |            (_)                | |              
#    | |  | | __ _| |_ __ _   ___ _ _ __ ___     ___| | __ _ ___ ___ 
#    | |  | |/ _` | __/ _` | / __| | '_ ` _ \   / __| |/ _` / __/ __|
#    | |__| | (_| | || (_| | \__ \ | | | | | | | (__| | (_| \__ \__ \
#    |_____/ \__,_|\__\__,_| |___/_|_| |_| |_|  \___|_|\__,_|___/___/
#                         ______                                      
#                        |______|                                             

class data_sim:

    def __init__(self, traj_type, N, dt, std):

        self.x_pos = np.zeros(N)
        self.y_pos = np.zeros(N)
        self.N = N
        self.dt = dt
        self.traj_type = traj_type
        self.r = conf_limo.r
        self.b = conf_limo.b
        self.std = std

        if self.traj_type == "line":
            for i in range(0, self.N):
                self.x_pos[i] = 0.3 * i * self.dt
                self.y_pos[i] = 0.1 * i * self.dt

        if traj_type == "sin":
            for i in range(1, N):
                self.y_pos[i] = 0 + 0.8 * np.sin(1.5 * i * self.dt)
                self.x_pos[i] = 2 + 0.8 * i * self.dt

    def relative_target_pos(self, state, i, flag):
        target_pos = np.zeros(2)
        noise = np.random.normal(0, self.std, 2)
        if flag == "xy":
            target_pos[0] = self.x_pos[i] - state[0]
            target_pos[1] = self.y_pos[i] - state[0]

        if flag == "dth":
            target_pos[0] = np.sqrt((self.x_pos[i] - state[0]) ** 2 + (self.y_pos[i] - state[1]) ** 2)
            target_pos[1] = np.arctan2(self.y_pos[i] - state[1], self.x_pos[i] - state[0])

        return target_pos + noise

    def absolute_target_pos(self, i):
        target_pos = np.array([self.x_pos[i], self.y_pos[i]])
        noise = np.random.normal(0, self.std, 2)
        return target_pos + noise

    def prop_sensors(self, state, des_pos):
        noise = np.random.normal(0, self.std)
        dtheta = np.arctan2(des_pos[1] - state[1], des_pos[0] - state[0]) - state[2]
        delta = np.sqrt((des_pos[0] - state[0]) ** 2 + (des_pos[1] - state[1]) ** 2)
        v = delta / self.dt
        w = dtheta / self.dt
        w_l = np.round((v - w * self.b / 2) / self.r)
        w_r = np.round((v + w * self.b / 2) / self.r)
        return np.array([w_l, w_r, w]) + np.array([0, 0, noise])
    
    def input(self, state, des_pos):
        noise = np.random.normal(0, self.std, 2)
        dtheta = np.arctan2(des_pos[1] - state[1], des_pos[0] - state[0]) - state[2]
        delta = np.sqrt((des_pos[0] - state[0]) ** 2 + (des_pos[1] - state[1]) ** 2)
        v = delta / self.dt
        w = dtheta / self.dt
        return np.array([v, w]) + noise
    
    def ext_sensors(self, state):
        noise = np.random.normal(0, self.std, 3)
        return state + noise
    
    def sensors_from_input(self, inputs):
        v = inputs[0]
        w = inputs[1]
        noise = np.random.normal(0, self.std)
        w_l = np.round((v - w * self.b / 2) / self.r)
        w_r = np.round((v + w * self.b / 2) / self.r)
        return np.array([w_l, w_r])
    



#  _______        _                     _       
# |__   __|      | |                   (_)      
#    | | ___  ___| |_   _ __ ___   __ _ _ _ __  
#    | |/ _ \/ __| __| | '_ ` _ \ / _` | | '_ \ 
#    | |  __/\__ \ |_  | | | | | | (_| | | | | |
#    |_|\___||___/\__| |_| |_| |_|\__,_|_|_| |_|
                                               
                                               

import conf_limo

if __name__ == "__main__":

    # Define initial and final position
    initial_state = np.array([0, 0, 0])
    des_pos = np.array([0.1, 0.3, 0])

    # Create an instance of the class
    N = 1000
    traj_type = "line"
    dt = 0.1
    std = 0.1
    ds = data_sim(traj_type, N, dt, std)

    # Print for test
    print("Relative target position: ", ds.relative_target_pos(initial_state, 999, "xy"))
    print("Absolute target position: ", ds.absolute_target_pos(999))
    print("Proprieceptive sensors: ", ds.prop_sensors(initial_state, des_pos))
    print("Exteroceptive sensors: ", ds.ext_sensors(des_pos))
    print("Input: ", ds.input(initial_state, des_pos))