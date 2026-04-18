#  _      _____ __  __  ____       _____       _______       
# | |    |_   _|  \/  |/ __ \     |  __ \   /\|__   __|/\    
# | |      | | | \  / | |  | |    | |  | | /  \  | |  /  \   
# | |      | | | |\/| | |  | |    | |  | |/ /\ \ | | / /\ \  
# | |____ _| |_| |  | | |__| |    | |__| / ____ \| |/ ____ \ 
# |______|_____|_|  |_|\____/     |_____/_/    \_\_/_/    \_\

import numpy as np

# ----------------- LIMO --------------------
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
v_min = -1
# limo max yaw rate 2*v_max/(b/2)
w_max = 0.1*4*v_max/b
w_min = -0.1*4*v_max/b

# ----------------- SENSORS -------------------
# lidar covariance
R_rr = np.identity(3) * 0.01
R_rp = np.identity(2) * 0.01
# model covariance
Q = np.identity(3) * 0.001

# ---------------- CONTROLS -------------------
# distance between target and center
dist = 1.5
# raidus of the circle around the central position
r_circle = 0.5
# simulation steps for MPC
N_sim = 100

# ------------- CAMERA ------------------------
rx = 1920 #resolution of the camera along x
d = 10/rx # physical size of a pixel (mm)
f = 6.86 # focal length (mm)
fpx = 1.30092897e+03 # focal length (px)
rx_depth = 1920 # resolution of the depth camera along x
n_d = 179 # number of depth channels
intrinsic_camera = np.array(((1.30092897e+03, 0.0, 9.73008095e+02),(0.0, 1.30280592e+03, 7.22305051e+02),(0,0,1)))
distortion = np.array((0.15521988, -0.57432332, -0.00566708, -0.00113189,  0.68733757))
# relative position of the camera in respect to the limo RF
x_camera = 0.03 #(m)
y_camera = 0    #(m)
z_camera = 0.1  #(m)
fps = 30

# ------------ ARUCO -------------------------
L = 0.10 # half of limo width (along y)
H = 0.13 # distance from back aruco to center of the limo (along x)
h = 0.045 # distance from side arucos to center of the limo (along x)
aruco_size = 0.08 # (m)