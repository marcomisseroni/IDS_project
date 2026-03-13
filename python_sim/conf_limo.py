#  _      _____ __  __  ____       _____       _______       
# | |    |_   _|  \/  |/ __ \     |  __ \   /\|__   __|/\    
# | |      | | | \  / | |  | |    | |  | | /  \  | |  /  \   
# | |      | | | |\/| | |  | |    | |  | |/ /\ \ | | / /\ \  
# | |____ _| |_| |  | | |__| |    | |__| / ____ \| |/ ____ \ 
# |______|_____|_|  |_|\____/     |_____/_/    \_\_/_/    \_\

import numpy as np

# limo wheel radius
r = 0.033
# limo baseline (175mm?)
b = 0.162
# limo wheelbase (length)
w = 0.2
# limo collision radius
r_collision = 0.2
# limo max speed (in datasheet 1m/s)
v_max = 1
v_min = -0.1
# limo max yaw rate 2*v_max/(b/2)
w_max = 0.1*4*v_max/b
w_min = -0.1*4*v_max/b
# lidar covariance
R = np.identity(3) * 0.01
# model covariance
Q = np.identity(3) * 0.001
# distance between target and center
dist = 1.5
# raidus of the circle around the central position
r_circle = 0.5
# simulation steps for MPC
N_sim = 100
