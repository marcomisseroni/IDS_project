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
rx = 640 #resolution of the camera along x
rx_depth = 640 # resolution of the depth camera along x
ry_depth = 400
# from camera calibration
fpx = 489.49 # focal length (px)
intrinsic_camera = np.array(((489.488, 0, 314.035),(0, 489.488, 219.010),(0,0,1)))
distortion = np.array((0.0719, -0.0947, -9.58e-06, 0.00142, 0.0))
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