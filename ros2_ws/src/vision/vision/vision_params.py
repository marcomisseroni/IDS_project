# __      ___     _                                                    _                
# \ \    / (_)   (_)                                                  | |               
#  \ \  / / _ ___ _  ___  _ __    _ __   __ _ _ __ __ _ _ __ ___   ___| |_ ___ _ __ ___ 
#   \ \/ / | / __| |/ _ \| '_ \  | '_ \ / _` | '__/ _` | '_ ` _ \ / _ \ __/ _ \ '__/ __|
#    \  /  | \__ \ | (_) | | | | | |_) | (_| | | | (_| | | | | | |  __/ ||  __/ |  \__ \
#     \/   |_|___/_|\___/|_| |_| | .__/ \__,_|_|  \__,_|_| |_| |_|\___|\__\___|_|  |___/
#                                | |                                                    
#                                |_|                                                    

import numpy as np
# ------------- CAMERA ------------------------
rx = 1920 #resolution of the camera along x
rx_depth = 640 # resolution of the depth camera along x
f = 6.86 # focal length (mm)
n_d = 179 # number of depth channels
# from camera calibration
fpx = 1.30092897e+03 # focal length (px)
intrinsic_camera = np.array(((1.30092897e+03, 0.0, 9.73008095e+02),(0.0, 1.30280592e+03, 7.22305051e+02),(0,0,1)))
distortion = np.array((0.15521988, -0.57432332, -0.00566708, -0.00113189,  0.68733757))
# relative position of the camera in respect to the limo RF
x_camera = 0.03 #(m)
y_camera = 0    #(m)
z_camera = 0.1  #(m)
fps = 30

# ------------ ARUCO -------------------------
L = 0.1367/2 # half of limo width (along y)
H = 0.170 # distance from back aruco to center of the limo (along x)
h = 0.0188 # distance from side arucos to center of the limo (along x)
aruco_size = 0.080 # (m)