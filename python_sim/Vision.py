import numpy as np
from scipy.linalg import logm, expm
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import conf_limo

class Vision:
#  _____       _ _   _       _ _          _   _             
# |_   _|     (_) | (_)     | (_)        | | (_)            
#   | |  _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __  
#   | | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
#  _| |_| | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
# |_____|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_| 

    def __init__(self, cap, cap_d):
        # type of aruco library used
        self.aruco_type = "DICT_6X6_50"
        # dictionary with different aruco sets
        self.ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }
        self.model = YOLO("yolov8n.pt")
        self.f = conf_limo.f # focal length (mm)
        self.rx = conf_limo.rx #resolution of the camera along x
        self.d = conf_limo.d # physical size of a pixel (mm)
        self.n_d = conf_limo.n_d # number of depth channels


#                                  _____       _            _   _             
#     /\                          |  __ \     | |          | | (_)            
#    /  \   _ __ _   _  ___ ___   | |  | | ___| |_ ___  ___| |_ _  ___  _ __  
#   / /\ \ | '__| | | |/ __/ _ \  | |  | |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
#  / ____ \| |  | |_| | (_| (_) | | |__| |  __/ ||  __/ (__| |_| | (_) | | | |
# /_/    \_\_|   \__,_|\___\___/  |_____/ \___|\__\___|\___|\__|_|\___/|_| |_|
                                                                                                                                                      
                                                                                                                                                    
    # from rotation matrix to transformation matrix
    def rotate(self, rot):
        R = np.zeros((4,4))
        R[:3,:3] = rot
        R[3,3] = 1
        return R

    # given the rotation axis ("X","Y","Z") it returns the rotation matrix
    def rotate_angle(self, axis, theta):
        axis = axis.upper()
        R = np.eye(4)
        if(axis == "X"):
            R[1,1] = np.cos(theta)
            R[1,2] = -np.sin(theta)
            R[2,1] = np.sin(theta)
            R[2,2] = np.cos(theta)
        elif(axis == "Y"):
            R[0,0] = np.cos(theta)
            R[0,2] = np.sin(theta)
            R[2,0] = -np.sin(theta)
            R[2,2] = np.cos(theta)
        elif(axis == "Z"):
            R[0,0] = np.cos(theta)
            R[0,1] = -np.sin(theta)
            R[1,0] = np.sin(theta)
            R[1,1] = np.cos(theta)
        else:
            print("Wrong rotation axis")
        return R

    # from translation vector to translation matrix
    def translate(self, t):
        T = np.eye(4)
        T[:3,3] = t
        return T

    # get coordinates of the point in the center of the reference frame
    def get_point(self, RF):
        return RF[:3,3]

    # get the angle of the reference frame along z
    def get_z_angle(self, RF):
        # assuming only rotation along z
        theta = np.atan2(RF[1,0], RF[0,0])
        return theta

    # print the aruco number on the image and the bounding box
    def aruco_display(self, corners, ids, image):
        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))

                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
                
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                
                cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        return image

    # estimate the aruco pose
    def aruco_pose_estimation(self, frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, aruco_size):
        # converting the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # loading the correct aruco dictionary
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        # using opencv to detect the aruco corners and index
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

        # vector containing all the aruco positions and rotation in this frame
        aruco_pos = np.zeros((9, 3)) # 9 possible aruco ids and 3 elements for each one (x,y,z)
        aruco_rot = np.zeros((9, 3, 3)) # 9 possible aruco and 3x3 rotation matrix

        # if we have detected any aruco in the image
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # tvec is the translation vector [x=right, y=bottom, z=forward]
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], aruco_size, matrix_coefficients, distortion_coefficients)
                cv2.aruco.drawDetectedMarkers(frame, corners)
                # plotting the RF
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
                self.aruco_display(corners, ids, frame)
                # from rodrigues rotation to rotation matrix
                rot, jac = cv2.Rodrigues(rvec)
                # creating the transformation matrix
                T = self.translate(tvec) @ self.rotate(rot)
                # transforming in the camera correct RF (x=forward, y=left, z=top)
                T_aruco_camera = self.rotate_angle("Z", -np.pi/2) @ self.rotate_angle("X", -np.pi/2)
                T_camera = T_aruco_camera @ T
                # saving the values
                aruco_pos[ids[i][0],:] = self.get_point(T_camera)
                aruco_rot[ids[i][0],:,:] = T_camera[:3,:3]
        return frame, aruco_pos, aruco_rot

    # estimate the limo positions
    def limo_estimation(self, aruco_pos, aruco_rot, T_limo_camera):
        # limo aruco positions
        L = conf_limo.L # half of limo width (along y)
        H = conf_limo.H # distance from back aruco to center of the limo (along x)
        h = conf_limo.h # distance from side arucos to center of the limo (along x)

        # vector to store the three limo reference frames
        limo_RF = np.zeros((3,4,4))

        # ---------------------- LIMO 0 ----------------------------
        flag0 = 0;	flag1 = 0;	flag2 = 0
        RF_limo0 = np.eye(4); RF_limo1 = np.eye(4); RF_limo2 = np.eye(4)
        # left side arucos
        if aruco_pos[0][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[0]) @ self.rotate(aruco_rot[0])
            # limo RF
            RF_limo0 = RF_aruco @ self.translate([-h, 0, -L])  @ self.rotate_angle("Y", np.pi) @ self.rotate_angle("X", -np.pi/2)
            flag0 = 1
        # center arucos
        if aruco_pos[1][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[1]) @ self.rotate(aruco_rot[1])
            # limo RF
            RF_limo1 = RF_aruco @ self.translate([0, 0, -H])  @ self.rotate_angle("Y", np.pi/2) @ self.rotate_angle("X", -np.pi/2)
            flag1 = 1
        # right side arucos
        if aruco_pos[2][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[2]) @ self.rotate(aruco_rot[2])
            # limo RF
            RF_limo2 = RF_aruco @ self.translate([h, 0, -L]) @ self.rotate_angle("X", -np.pi/2)
            flag2 = 1
        if flag0 or flag1 or flag2:
            # mean of the three possible reference frames
            limo_RF0 = expm((logm(RF_limo0)*flag0 + logm(RF_limo1)*flag1 + logm(RF_limo2)*flag2) / (flag0 + flag1 + flag2))
            # transforming the reading in the limo RF
            limo_RF[0] = T_limo_camera @ limo_RF0

        
        
        # ---------------------- LIMO 1 ----------------------------
        flag3 = 0;	flag4 = 0;	flag5 = 0
        RF_limo3 = np.eye(4); RF_limo4 = np.eye(4); RF_limo5 = np.eye(4)
        # left side arucos
        if aruco_pos[3][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[3]) @ self.rotate(aruco_rot[3])
            # limo RF
            RF_limo3 = RF_aruco @ self.translate([-h, 0, -L])  @ self.rotate_angle("Y", np.pi) @ self.rotate_angle("X", -np.pi/2)
            flag3 = 1
        # center arucos
        if aruco_pos[4][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[4]) @ self.rotate(aruco_rot[4])
            # limo RF
            RF_limo4 = RF_aruco @ self.translate([0, 0, -H])  @ self.rotate_angle("Y", np.pi/2) @ self.rotate_angle("X", -np.pi/2)
            flag4 = 1
        # right side arucos
        if aruco_pos[5][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[5]) @ self.rotate(aruco_rot[5])
            # limo RF
            RF_limo5 = RF_aruco @ self.translate([h, 0, -L]) @ self.rotate_angle("X", -np.pi/2)
            flag5 = 1
        if flag3 or flag4 or flag5:
            # mean of the three possible reference frames
            limo_RF1 = expm((logm(RF_limo3)*flag3 + logm(RF_limo4)*flag4 + logm(RF_limo5)*flag5) / (flag3 + flag4 + flag5))
            # transforming the reading in the limo RF
            limo_RF[1] = T_limo_camera @ limo_RF1

        # ---------------------- LIMO 2 ----------------------------
        flag6 = 0;	flag7 = 0;	flag8 = 0
        RF_limo6 = np.eye(4); RF_limo7 = np.eye(4); RF_limo8 = np.eye(4)
        # left side arucos
        if aruco_pos[6][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[6]) @ self.rotate(aruco_rot[6])
            # limo RF
            RF_limo6 = RF_aruco @ self.translate([-h, 0, -L])  @ self.rotate_angle("Y", np.pi) @ self.rotate_angle("X", -np.pi/2)
            flag6 = 1
        # center arucos
        if aruco_pos[7][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[7]) @ self.rotate(aruco_rot[7])
            # limo RF
            RF_limo7 = RF_aruco @ self.translate([0, 0, -H])  @ self.rotate_angle("Y", np.pi/2) @ self.rotate_angle("X", -np.pi/2)
            flag7 = 1
        # right side arucos
        if aruco_pos[8][0] != 0:
            # aruco RF
            RF_aruco = self.translate(aruco_pos[8]) @ self.rotate(aruco_rot[8])
            # limo RF
            RF_limo8 = RF_aruco @ self.translate([h, 0, -L]) @ self.rotate_angle("X", -np.pi/2)
            flag8 = 1
        if flag6 or flag7 or flag8:
            # mean of the three possible reference frames
            limo_RF2 = expm((logm(RF_limo6)*flag6 + logm(RF_limo7)*flag7 + logm(RF_limo8)*flag8) / (flag6 + flag7 + flag8))
            # transforming the reading in the limo RF
            limo_RF[2] = T_limo_camera @ limo_RF2

        return limo_RF

#  _______                   _     _____       _            _   _             
# |__   __|                 | |   |  __ \     | |          | | (_)            
#    | | __ _ _ __ __ _  ___| |_  | |  | | ___| |_ ___  ___| |_ _  ___  _ __  
#    | |/ _` | '__/ _` |/ _ \ __| | |  | |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
#    | | (_| | | | (_| |  __/ |_  | |__| |  __/ ||  __/ (__| |_| | (_) | | | |
#    |_|\__,_|_|  \__, |\___|\__| |_____/ \___|\__\___|\___|\__|_|\___/|_| |_|
#                  __/ |                                                      
#                 |___/                                                       

    def detect_target(self, frame):
        results = self.model(frame, verbose=False)
        detected_objects = []

        # we want the closest (biggest) person position
        x_target = None
        y_target = None
        x_1 = None
        x_2 = None
        y_1 = None
        y_2 = None
        biggest_area = 0
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])  # Get class ID
                confidence = box.conf[0].item()  # Confidence score
                label = self.model.names[class_id]

                # drawing the box only around a person
                if confidence > 0.5 and label=="person":
                    detected_objects.append(label)
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    # drawing a circle in the box centroid
                    x_c = int((x1+x2)/2)
                    y_c = int((y1+y2)/2)
                    cv2.circle(frame, (x_c, y_c), 3, (255,0,0), 4)
                    # finding the biggest box area
                    area = (x2-x1)*(y2-y1)
                    if area > biggest_area:
                        biggest_area = area
                        x_target = x_c
                        y_target = y_c
                        x_1 = x1
                        x_2 = x2
                        y_1 = y1
                        y_2 = y2
        if x_target != None:
            cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (0, 0, 255), 2)
            cv2.circle(frame, (x_target, y_target), 3, (0,0,255), 3)
        return frame, x_1, x_2, y_1, y_2, x_target

    def target_estimation_RGBD(self, x1, x2, y1, y2, xc, depth):
        # angle in respect to the camera
        tan_theta = 1/self.f*(self.rx/2-xc)
        # distance from the depth image
        # using max value of the region
        region = depth[y1:y2, x1:x2]
        depth_val = np.max(region)
        D = -2970/self.n_d*depth_val + 3000

        # distances
        x_relative = D
        y_relative = D*tan_theta
        return x_relative, y_relative


#  _______        _   
# |__   __|      | |  
#    | | ___  ___| |_ 
#    | |/ _ \/ __| __|
#    | |  __/\__ \ |_ 
#    |_|\___||___/\__|

if __name__ == "__main__":
    print("prova")
    vision = Vision()