# pip install ultralytics --break-system-packages
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
# camera settings
fps = 25
f = 50 # focal length (mm)
rx = 1920 #resolution of the camera along x
d = 36/rx # physical size of a pixel (mm)
n_d = 255 # number of depth channels

def detect_human(frame):
    results = model(frame, verbose=False)
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
            label = model.names[class_id]

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

def target_estimation_stereo(x_target_L, x_target_R):
    T = 100 # sensor distance (mm)
    n1 = x_target_L # number of pixels from the sensor border
    n2 = x_target_R # number of pixels from the sensor border

    # relative position of the target in respect to the center of the two cameras (mm)
    x_relative = f*T/(d*(n1-n2))
    y_relative = -(n1+n2-rx)*T/(2*(n1-n2))
    return x_relative, y_relative

def target_estimation_RGBD(x1, x2, y1, y2, xc, depth):
    # angle in respect to the camera
    tan_theta = 1/f*(rx/2-xc)
    # distance from the depth image
    # using max value of the region
    region = depth[y1:y2, x1:x2]
    depth_val = np.max(region)
    D = -2970/n_d*depth_val + 3000

    # distances
    x_relative = D
    y_relative = D*tan_theta
    return x_relative, y_relative

def main():
    # Open the RGBD video
    cap = cv2.VideoCapture("Test_videos/RGB 3.mp4")
    cap_d = cv2.VideoCapture("Test_videos/Depth.mp4")

    delay = int(1000 / fps)
    
    # vector to contain all the estimated positions
    x_rel = []
    y_rel = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Target trajectory")
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.grid(True)
    line, = ax.plot([], [], 'b.')

    while True:
        # loading the current frame
        ret, frame = cap.read()
        ret_d, frame_d = cap_d.read()
        if not ret:
            print("ERROR IN CAPTURING")
            break

        # using YOLO to detect the target in the frame
        frame, x1, x2, y1, y2, xc = detect_human(frame)

        combined = np.hstack((frame, frame_d))
        combined_resized = cv2.resize(combined, (1280, 360))
        cv2.imshow("Person recognition", combined_resized)

        # estimated position of the target (if found)
        if xc != None:
            x_relative, y_relative = target_estimation_RGBD(x1, x2, y1, y2, xc, frame_d)
            x_rel.append(x_relative)
            y_rel.append(y_relative)

        line.set_xdata(x_rel)
        line.set_ydata(y_rel)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        if cv2.waitKey(delay) == ord('q') or not ret:
            break

    cap.release()
    cap_d.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


# CODE FOR STEREO CAMERAS
'''
def main():
    # Open the two stereo camera videos
    cap_L = cv2.VideoCapture("Left camera.mp4")
    cap_R = cv2.VideoCapture("Right camera.mp4")

    delay = int(1000 / fps)
    # filter smoothness
    alpha = 0.1
    x_filt_L = None
    x_filt_R = None
    
    # vector to contain all the estimated positions
    x_rel = []
    y_rel = []

    while True:
        # loading the current frames
        ret_L, frame_L = cap_L.read()
        ret_R, frame_R = cap_R.read()
        if not ret_L or not ret_R:
            print("ERROR IN CAPTURING")
            break

        # using YOLO to detect the target in the frame
        frame_L, x_target_L, y_target_L = detect_objects(frame_L)
        frame_R, x_target_R, y_target_R = detect_objects(frame_R)

        # filtering the obtained position
        if x_filt_L is None:
            x_filt_L = x_target_L
            x_filt_R = x_target_R
        else:
            x_filt_L = alpha * x_target_L + (1 - alpha) * x_filt_L
            x_filt_R = alpha * x_target_R + (1 - alpha) * x_filt_R


        combined = np.hstack((frame_L, frame_R))
        combined_resized = cv2.resize(combined, (1280, 360))
        cv2.imshow("Person recognition", combined_resized)

        # estimated position of the target
        x_relative, y_relative = target_estimation(x_filt_L, x_filt_R)
        x_rel.append(x_relative)
        y_rel.append(y_relative)
        if cv2.waitKey(delay) == ord('q') or not ret_L:
            break

    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()

    # target trajectory plot
    plt.figure(figsize=(10, 4))
    plt.title("Target trajectory")
    plt.plot([1300, 1700], [0, 0], 'r-', linewidth=2, label="Real position")
    plt.plot([1700, 1700], [0, 300], 'r-', linewidth=2,)
    plt.plot([1700, 1300], [300, 300], 'r-', linewidth=2,)
    plt.plot(x_rel, y_rel, 'b', linewidth=2, label="Estimated position")
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.grid(True)
    plt.legend()
    plt.show()
    '''