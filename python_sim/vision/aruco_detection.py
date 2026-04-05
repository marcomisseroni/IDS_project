import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import time

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
def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, aruco_size):
	# converting the image to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# loading the correct aruco dictionary
	cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
	parameters = cv2.aruco.DetectorParameters()

	# using opencv to detect the aruco corners and index
	corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)

	# vector containing all the aruco positions in this frame
	aruco_pos = []
	# if we have detected any aruco in the image
	if len(corners) > 0:
		for i in range(0, len(ids)):
			# tvec is the translation vector [x=right, y=bottom, z=forward]
			rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
				corners[i], aruco_size, matrix_coefficients, distortion_coefficients)
			cv2.aruco.drawDetectedMarkers(frame, corners)
			cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
			aruco_display(corners, ids, frame)
			# from rodrigues rotation to rotation matrix
			rot, jac = cv2.Rodrigues(rvec)
			aruco_pos.append(tvec)
	return frame, aruco_pos
   
def main():
	# select the type of aruco used
	aruco_type = "DICT_6X6_50"
	# aruco marker size
	aruco_size = 0.08 # (m)

	arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
	arucoParams = cv2.aruco.DetectorParameters()

	# camera matrix and distortion vector obtained after calibration
	intrinsic_camera = np.array(((1.626e+03, 0, 9.351e+02),(0,1.612e+03, 5.145e+02),(0,0,1)))
	distortion = np.array((0.14321164, -0.37941193, -0.00400418, -0.00202883, -0.25072842))

	cap = cv2.VideoCapture("test_videos/Aruco2.MOV")

	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

	# plot
	plt.ion()
	fig, ax = plt.subplots(figsize=(10, 4))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title("Target trajectory")
	ax.set_xlabel('x [m]')
	ax.set_ylabel('y [m]')
	ax.set_zlabel('z [m]')
	ax.set_xlim(-2, 2)
	ax.set_ylim(-2, 2)
	ax.set_zlim(0, 4)
	ax.grid(True)
	line, = ax.plot([], [], 'b.')

	# vector containing the data
	x_pos = []
	y_pos = []
	z_pos = []

	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	for i in range(length):
		ret, img = cap.read()
		output, aruco_pos = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion, aruco_size)
		cv2.imshow('Estimated Pose', output)
		
		if len(aruco_pos)>0:
		# saving the first aruco position
			[x, y, z] = aruco_pos[0][0][0]
			x_pos.append(x)
			y_pos.append(y)
			z_pos.append(z)

			# updating the plot
			line.set_data(x_pos, y_pos)
			line.set_3d_properties(z_pos)
			ax.relim()
			ax.autoscale_view()
			plt.draw()
			plt.pause(0.001)

		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
	plt.show(block=True)


if __name__ == "__main__":
    main()