import numpy as np
from scipy.linalg import logm, expm
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import time

plot = True

# dictionary with different aruco sets
ARUCO_DICT = {
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

# from rotation matrix to transformation matrix
def rotate(rot):
	R = np.zeros((4,4))
	R[:3,:3] = rot
	R[3,3] = 1
	return R

# given the rotation axis ("X","Y","Z") it returns the rotation matrix
def rotate_angle(axis, theta):
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
def translate(t):
	T = np.eye(4)
	T[:3,3] = t
	return T

# get coordinates of the point in the center of the reference frame
def get_point(RF):
	return RF[:3,3]

# get the angle of the reference frame along z
def get_z_angle(RF):
	# assuming only rotation along z
	theta = np.atan2(RF[1,0], RF[0,0])
	return theta

# print the aruco number on the image and the bounding box
def aruco_display(corners, ids, image):
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
def aruco_pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, aruco_size):
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
			aruco_display(corners, ids, frame)
			# from rodrigues rotation to rotation matrix
			rot, jac = cv2.Rodrigues(rvec)
			# creating the transformation matrix
			T = translate(tvec) @ rotate(rot)
			# transforming in the camera correct RF (x=forward, y=left, z=top)
			T_aruco_camera = rotate_angle("Z", -np.pi/2) @ rotate_angle("X", -np.pi/2)
			T_camera = T_aruco_camera @ T
			# saving the values
			aruco_pos[ids,:] = get_point(T_camera)
			aruco_rot[ids,:,:] = T_camera[:3,:3]
	return frame, aruco_pos, aruco_rot

# estimate the limo positions
def limo_estimation(aruco_pos, aruco_rot):
	# limo aruco positions
	L = 0.06 # half of limo width (along y)
	H = 0.12 # distance from back aruco to center of the limo (along x)
	h = 0.02 # distance from side arucos to center of the limo (along x)

	# vector to store the three limo reference frames
	limo_RF = np.zeros((3,4,4))

	# ---------------------- LIMO 0 ----------------------------
	flag0 = 0;	flag1 = 0;	flag2 = 0
	rz0 = 0; rz1 = 0; rz2 = 0
	t0 = 0; t1 = 0; t2 = 0
	# left side arucos
	if aruco_pos[0][0] != 0:
		# aruco RF
		RF_aruco = translate(aruco_pos[0]) @ rotate(aruco_rot[0])
		# limo RF
		RF_limo = RF_aruco @ translate([-h, 0, -L])  @ rotate_angle("Y", np.pi) @ rotate_angle("X", -np.pi/2)
		t0 = get_point(RF_limo)
		rz0 = get_z_angle(RF_limo)
		flag0 = 1
	# center arucos
	if aruco_pos[1][0] != 0:
		# aruco RF
		RF_aruco = translate(aruco_pos[1]) @ rotate(aruco_rot[1])
		# limo RF
		RF_limo = RF_aruco @ translate([0, 0, -H])  @ rotate_angle("Y", np.pi/2) @ rotate_angle("X", -np.pi/2)
		t1 = get_point(RF_limo)
		rz1 = get_z_angle(RF_limo)
		flag1 = 1
	# right side arucos
	if aruco_pos[2][0] != 0:
		# aruco RF
		RF_aruco = translate(aruco_pos[2]) @ rotate(aruco_rot[2])
		# limo RF
		RF_limo = RF_aruco @ translate([h, 0, -L]) @ rotate_angle("X", -np.pi/2)
		t2 = get_point(RF_limo)
		rz2 = get_z_angle(RF_limo)
		flag2 = 1
	if flag0 or flag1 or flag2:
		t = (t0*flag0 + t1*flag1 + t2*flag2) / (flag0+flag1+flag2)
		rz = (rz0*flag0 + rz1*flag1 + rz2*flag2) / (flag0+flag1+flag2)
		limo_RF[0] = translate(t) @ rotate_angle("Z",rz)

	
	
	# ---------------------- LIMO 1 ----------------------------
	# left side arucos
	if aruco_pos[3][0] != 0:
		print("left aruco data")
	# center arucos
	if aruco_pos[4][0] != 0:
		print("center aruco data")
	# right side arucos
	if aruco_pos[5][0] != 0:
		print("right aruco data")

	# ---------------------- LIMO 2 ----------------------------
	# left side arucos
	if aruco_pos[6][0] != 0:
		print("left aruco data")
	# center arucos
	if aruco_pos[7][0] != 0:
		print("center aruco data")
	# right side arucos
	if aruco_pos[8][0] != 0:
		print("right aruco data")

	# average position based on multiple limos
	return limo_RF

def main():
	# type of aruco library used
	aruco_type = "DICT_6X6_50"
	arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
	arucoParams = cv2.aruco.DetectorParameters()
	# aruco marker size
	aruco_size = 0.08 # (m)

	# camera matrix and distortion vector obtained after calibration
	intrinsic_camera = np.array(((1.626e+03, 0, 9.351e+02),(0,1.612e+03, 5.145e+02),(0,0,1)))
	distortion = np.array((0.14321164, -0.37941193, -0.00400418, -0.00202883, -0.25072842))

	cap = cv2.VideoCapture("test_videos/Limo_aruco.mp4")
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter('output.mp4', cv2.CAP_FFMPEG, fourcc, 20.0, (width,  height))

	# vector containing the data
	x_pos = []
	y_pos = []
	th = []

	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for i in range(length):
		ret, img = cap.read()

		# measuring the aruco positions
		output, aruco_pos, aruco_rot = aruco_pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, aruco_size)
		# estimating the limo positions
		limo_RF = limo_estimation(aruco_pos, aruco_rot)

		# to plot the RF i need to return in the aruco RF
		T_aruco_camera = rotate_angle("Z", -np.pi/2) @ rotate_angle("X", -np.pi/2)
		RF_plot = np.linalg.inv(T_aruco_camera) @ limo_RF[0]
		cv2.drawFrameAxes(output, intrinsic_camera, distortion, RF_plot[:3,:3], get_point(RF_plot), 0.1)

		# if we want the limo position and orientation along z
		limo_0_states = [get_point(limo_RF[0])[0], get_point(limo_RF[0])[1], get_z_angle(limo_RF[0])]
		x_pos.append(get_point(limo_RF[0])[0])
		y_pos.append(get_point(limo_RF[0])[1])
		th.append(get_z_angle(limo_RF[0]))

		#output_resized = cv2.resize(output, (1280, 720))
		cv2.namedWindow('Estimated Pose', cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty('Estimated Pose', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow('Estimated Pose', output)
		out.write(output)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	cap.release()
	out.release()
	cv2.destroyAllWindows()

	# plots
	if(plot):
		plt.scatter(x_pos, y_pos)
		plt.xlabel("X")
		plt.ylabel("Y")
		plt.title("Limo trajectory")
		plt.show()

		plt.plot(th, marker='o')
		plt.title("Limo angle")
		plt.show()


if __name__ == "__main__":
    main()