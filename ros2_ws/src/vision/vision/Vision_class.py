import numpy as np
from scipy.linalg import logm, expm
import cv2
import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
#from ultralytics import YOLO
from limo_description import conf_limo
#cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

class Vision:
#  _____       _ _   _       _ _          _   _             
# |_   _|     (_) | (_)     | (_)        | | (_)            
#   | |  _ __  _| |_ _  __ _| |_ ______ _| |_ _  ___  _ __  
#   | | | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \ 
#  _| |_| | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
# |_____|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_| 

    def __init__(self, logger):
        self.logger = logger
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
        #self.model = YOLO("yolov8n-seg.pt")
        self.fpx = conf_limo.fpx # focal length (px)
        self.rx = conf_limo.rx #resolution of the camera along x
        self.T_limo_camera = self.translate([conf_limo.x_camera, conf_limo.y_camera, conf_limo.z_camera])
        # type of aruco library used
        self.aruco_type = "DICT_6X6_50"
        arucoDict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT[self.aruco_type])
        #arucoParams = cv2.aruco.DetectorParameters()
        arucoParams = cv2.aruco.DetectorParameters_create()

#                                  _____       _            _   _             
#     /\                          |  __ \     | |          | | (_)            
#    /  \   _ __ _   _  ___ ___   | |  | | ___| |_ ___  ___| |_ _  ___  _ __  
#   / /\ \ | '__| | | |/ __/ _ \  | |  | |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
#  / ____ \| |  | |_| | (_| (_) | | |__| |  __/ ||  __/ (__| |_| | (_) | | | |
# /_/    \_\_|   \__,_|\___\___/  |_____/ \___|\__\___|\___|\__|_|\___/|_| |_|
                                                                                                                                                      
                                                                                                                                                    
    # from rotation matrix to transformation matrix
    def rotate(
            self, 
            rot: np.ndarray
            ) -> np.ndarray:
        R = np.zeros((4,4))
        R[:3,:3] = rot
        R[3,3] = 1
        return R

    # given the rotation axis ("X","Y","Z") it returns the rotation matrix
    def rotate_angle(
            self, 
            axis: str, 
            theta: float
            ) -> np.ndarray:
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
    def translate(
            self, 
            t: np.ndarray
            ) -> np.ndarray:
        T = np.eye(4)
        T[:3,3] = t
        return T

    # get coordinates of the point in the center of the reference frame
    def get_point(
            self, 
            RF: np.ndarray
            ) -> np.ndarray:
        return RF[:3,3]

    # get the angle of the reference frame along z
    def get_z_angle(
            self, 
            RF: np.ndarray
            ) -> float:
        # assuming only rotation along z
        theta = np.arctan2(RF[1,0], RF[0,0])
        return theta

    # print the aruco number on the image and the bounding box
    def aruco_display(
            self, 
            corners: list[np.ndarray], 
            ids: np.ndarray, 
            image: np.ndarray
            ) -> np.ndarray:
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
    def aruco_pose_estimation(
            self, 
            frame: np.ndarray, 
            aruco_dict_type: int, 
            matrix_coefficients: np.ndarray, 
            distortion_coefficients: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # converting the image to grayscale and improving the values for the aruco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        # loading the correct aruco dictionary
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        #parameters = cv2.aruco.DetectorParameters()
        parameters = cv2.aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parameters.cornerRefinementWinSize = 5
        parameters.cornerRefinementMaxIterations = 50
        parameters.cornerRefinementMinAccuracy = 0.01
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.minMarkerPerimeterRate = 0.03
        parameters.maxMarkerPerimeterRate = 4.0

        # using opencv to detect the aruco corners and index
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

        # vector containing all the aruco positions and rotation in this frame
        aruco_pos = np.zeros((11, 3)) # 11 possible aruco ids and 3 elements for each one (x,y,z)
        aruco_rot = np.zeros((11, 3, 3)) # 11 possible aruco and 3x3 rotation matrix

        # if we have detected any aruco in the image
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # aruco on a limo
                #self.logger.info(f"id {ids[i][0]}")
                if(ids[i][0] < 9):
                    # tvec is the translation vector [x=right, y=bottom, z=forward]
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], conf_limo.aruco_size, matrix_coefficients, distortion_coefficients)
                else:
                    rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                        corners[i], conf_limo.aruco_size_target, matrix_coefficients, distortion_coefficients)
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
                if(ids[i][0]<10):
                    # saving the values
                    aruco_pos[ids[i][0],:] = self.get_point(T_camera)
                    aruco_rot[ids[i][0],:,:] = T_camera[:3,:3]
                else:
                    print("Detected a wrong aruco")
        return frame, aruco_pos, aruco_rot

    # estimate the limo positions
    def limo_estimation(
            self, 
            aruco_pos: np.ndarray, 
            aruco_rot: np.ndarray
            ) -> np.ndarray:
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
            limo_RF[0] = self.T_limo_camera @ limo_RF0

        
        
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
            limo_RF[1] = self.T_limo_camera @ limo_RF1

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
            limo_RF[2] = self.T_limo_camera @ limo_RF2

        return limo_RF

#  _______                   _     _____       _            _   _             
# |__   __|                 | |   |  __ \     | |          | | (_)            
#    | | __ _ _ __ __ _  ___| |_  | |  | | ___| |_ ___  ___| |_ _  ___  _ __  
#    | |/ _` | '__/ _` |/ _ \ __| | |  | |/ _ \ __/ _ \/ __| __| |/ _ \| '_ \ 
#    | | (_| | | | (_| |  __/ |_  | |__| |  __/ ||  __/ (__| |_| | (_) | | | |
#    |_|\__,_|_|  \__, |\___|\__| |_____/ \___|\__\___|\___|\__|_|\___/|_| |_|
#                  __/ |                                                      
#                 |___/                                                       
    '''
    def detect_target(
            self, 
            frame: np.ndarray
            ) -> tuple[int, int, int, int, int, np.ndarray | None]:
        results = self.model(frame, verbose=False)
        detected_objects = []
        # we want the closest (biggest) person position
        x_target = 0; y_target = 0
        x_1 = 0; x_2 = 0; y_1 = 0; y_2 = 0
        biggest_area = 0
        mask = None
        for r in results:
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])  # Get class ID
                confidence = box.conf[0].item()  # Confidence score
                label = self.model.names[class_id]

                # drawing the box only around a person
                if confidence > 0.5 and label=="person":
                    detected_objects.append(label)
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # box centroid
                    x_target = int((x1+x2)/2)
                    y_target = int((y1+y2)/2)
                    # finding the biggest box area
                    area = (x2-x1)*(y2-y1)
                    if area > biggest_area:
                        biggest_area = area
                        x_1 = x1
                        x_2 = x2
                        y_1 = y1
                        y_2 = y2
                        # mask of the human
                        mask = r.masks.data[i].cpu().numpy()
                        if mask is not None:
                            mask = np.squeeze(mask)
                            mask = (mask * 255).astype(np.uint8)
        return x_1, x_2, y_1, y_2, x_target, mask
    '''
    def target_estimation_RGBD(
            self, 
            xc: int, 
            depth: np.ndarray, 
            mask: np.ndarray
            ) -> tuple[float, float, np.ndarray]:
        # angle in respect to the camera
        tan_theta = 1/self.fpx*(self.rx/2-xc)
        mask = cv2.resize(mask, (conf_limo.rx_depth, conf_limo.ry_depth))
        values = depth[(mask > 0) & (depth > 0)]
        # median of the depth values inside the mask
        D = np.median(values)

        # distances
        x_relative = D
        y_relative = D*tan_theta
        # transforming the reading in the limo RF
        target = self.translate([x_relative, y_relative, 0])
        #target_limoRF = self.T_limo_camera @ target
        # depth values should be in the limo rf
        target_limoRF = target
        # result in meters
        x = self.get_point(target_limoRF)[0]/1000.0
        y = self.get_point(target_limoRF)[1]/1000.0
        # showing the contour of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth, contours, -1, (0, 0, 0), 2)
        return x, y, depth

#  __  __       _          __                  _   _             
# |  \/  |     (_)        / _|                | | (_)            
# | \  / | __ _ _ _ __   | |_ _   _ _ __   ___| |_ _  ___  _ __  
# | |\/| |/ _` | | '_ \  |  _| | | | '_ \ / __| __| |/ _ \| '_ \ 
# | |  | | (_| | | | | | | | | |_| | | | | (__| |_| | (_) | | | |
# |_|  |_|\__,_|_|_| |_| |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|
                                                                                                                                
    def vision_main(
            self, 
            frame: np.ndarray,
            id: int,
            visualize: bool = False
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # initializing the values
        target=np.zeros(3); limo0=np.zeros(3); limo1=np.zeros(3); limo2=np.zeros(3)

        # --------------- LIMOS ----------------------
        # measuring the aruco positions
        output, aruco_pos, aruco_rot = self.aruco_pose_estimation(frame, self.ARUCO_DICT[self.aruco_type], conf_limo.intrinsic_camera,
                                                                  conf_limo.distortion)
        # estimating the limo positions
        limo_RF = self.limo_estimation(aruco_pos, aruco_rot)
        # saving the values in the form [x,y,th] for each limo
        limo0 = np.array([self.get_point(limo_RF[0])[0], self.get_point(limo_RF[0])[1], self.get_z_angle(limo_RF[0])])
        limo1 = np.array([self.get_point(limo_RF[1])[0], self.get_point(limo_RF[1])[1], self.get_z_angle(limo_RF[1])])
        limo2 = np.array([self.get_point(limo_RF[2])[0], self.get_point(limo_RF[2])[1], self.get_z_angle(limo_RF[2])])     

        # --------------- TARGET ----------------------
        # reading the aruco values on the target
        if(aruco_pos[9][0] != 0):
            RF_target = self.T_limo_camera @ self.translate(aruco_pos[9]) @ self.rotate(aruco_rot[9])
        else:
            RF_target = self.T_limo_camera @ self.translate(aruco_pos[10]) @ self.rotate(aruco_rot[10])
        target = np.array([self.get_point(RF_target)[0], self.get_point(RF_target)[1], 0])

        # -------------- Visualization --------------------
        if(visualize):
            output = frame
            # to plot the RF i need to return in the aruco RF
            T_aruco_camera = self.rotate_angle("Z", -np.pi/2) @ self.rotate_angle("X", -np.pi/2)
            RF_plot0 = np.linalg.inv(T_aruco_camera) @ np.linalg.inv(self.T_limo_camera) @ limo_RF[0]
            RF_plot1 = np.linalg.inv(T_aruco_camera) @ np.linalg.inv(self.T_limo_camera) @ limo_RF[1]
            RF_plot2 = np.linalg.inv(T_aruco_camera) @ np.linalg.inv(self.T_limo_camera) @ limo_RF[2]
            RF_plot3 = np.linalg.inv(T_aruco_camera) @ np.linalg.inv(self.T_limo_camera) @ RF_target
            cv2.drawFrameAxes(output, conf_limo.intrinsic_camera, conf_limo.distortion, RF_plot0[:3,:3], self.get_point(RF_plot0), 0.1)
            cv2.drawFrameAxes(output, conf_limo.intrinsic_camera, conf_limo.distortion, RF_plot1[:3,:3], self.get_point(RF_plot1), 0.1)
            cv2.drawFrameAxes(output, conf_limo.intrinsic_camera, conf_limo.distortion, RF_plot2[:3,:3], self.get_point(RF_plot2), 0.1)
            cv2.drawFrameAxes(output, conf_limo.intrinsic_camera, conf_limo.distortion, RF_plot3[:3,:3], self.get_point(RF_plot3), 0.1)
            # showing the video
            resized = cv2.resize(output, (conf_limo.rx_depth, conf_limo.ry_depth))
            cv2.imshow(f"Limo {id}", resized)
            cv2.waitKey(1)
        return target, limo0, limo1, limo2 # outputs in the form [x,y,th]