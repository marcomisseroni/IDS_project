import numpy as np
#  _      _____ __  __  ____       _____       _______       
# | |    |_   _|  \/  |/ __ \     |  __ \   /\|__   __|/\    
# | |      | | | \  / | |  | |    | |  | | /  \  | |  /  \   
# | |      | | | |\/| | |  | |    | |  | |/ /\ \ | | / /\ \  
# | |____ _| |_| |  | | |__| |    | |__| / ____ \| |/ ____ \ 
# |______|_____|_|  |_|\____/     |_____/_/    \_\_/_/    \_\

# initial positions
r_circle = 0.5 # raidus of the circle around the central position
limo_init = np.array([
    [-r_circle*np.cos(60*np.pi/180),-r_circle*np.sin(60*np.pi/180),0.0], #limo2
    [-r_circle*np.cos(60*np.pi/180),r_circle*np.sin(60*np.pi/180),0.0], #limo1
    [r_circle,0.0,0.0] # limo0
])
target_init = np.array([2.0,0.0])

x_center_init = limo_init[:, 0].mean()
y_center_init = limo_init[:, 1].mean()
initial_center = np.array([x_center_init, y_center_init])

# ----------------- LIMO --------------------
# limo wheel radius
r = 0.096288/2
# limo baseline (width)
b = 0.175
# limo wheelbase (length)
w = 0.2
# limo collision radius
r_collision = 0.02
# limo max speed (in datasheet 1m/s)
v_max = 0.14772322344521518
v_min = -v_max
# limo max yaw rate 2*v_max/(b/2)
w_max = 0.29544644689043037
w_min = -w_max

# -------------- KALMAN ----------------------
# measurement covariance
R_rr = np.identity(3) * 0.01
R_rp = np.identity(2) * 0.01
# model covariance
Q = np.identity(3) * 0.001
# time step
dt = 0.1
# person model covariance
Q_p = np.zeros((4, 4))
Q_p[0, 0] = dt**4 / 4
Q_p[0, 2] = dt**3 / 2
Q_p[1, 1] = dt**4 / 4
Q_p[1, 3] = dt**3 / 2
Q_p[2, 0] = dt**3 / 2
Q_p[2, 2] = dt**2
Q_p[3, 1] = dt**3 / 2
Q_p[3, 3] = dt**2
# period to publish the estimated state
Tp = 0.1

# ---------------- CONTROLS -------------------
# distance between target and center
dist = 1.0
# time horizon MPC (number of steps)
N = 20
# MPC timestep
dt_MPC = 0.1

# ------------- CAMERA ------------------------
rx = 640 #resolution of the camera along x
rx_depth = 640 # resolution of the depth camera along x
ry_depth = 480
# from camera calibration
fpx = 489.49 # focal length (px)
intrinsic_camera = np.array(((489.488, 0, 314.035),(0, 489.488, 219.010),(0,0,1)))
distortion = np.array((0.0719, -0.0947, -9.58e-06, 0.00142, 0.0))

#fpx = 1.30092897e+03 # focal length (px)
#intrinsic_camera = np.array(((1.30092897e+03, 0.0, 9.73008095e+02),(0.0, 1.30280592e+03, 7.22305051e+02),(0,0,1)))
#distortion = np.array((0.15521988, -0.57432332, -0.00566708, -0.00113189,  0.68733757))

# relative position of the camera in respect to the limo RF
x_camera = 0.086 #(m)
y_camera = 0    #(m)
z_camera = 0.18  #(m)
fps = 30

# ------------ ARUCO -------------------------
L = 0.1367/2 # half of limo width (along y)
H = 0.170 # distance from back aruco to center of the limo (along x)
h = 0.0188 # distance from side arucos to center of the limo (along x)
aruco_size = 0.080 # (m)
aruco_size_target = 0.144 # (m)