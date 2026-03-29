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


def detect_objects(frame):
    results = model(frame, verbose=False)
    detected_objects = []

    # we want the closest (biggest) person position
    x_target = None
    y_target = None
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
                    cv2.circle(frame, (x_c, y_c), 3, (0,0,255), 3)

    return frame, x_target, y_target

def target_estimation(x_target_L, x_target_R):
    f = 50 # focal length (mm)
    T = 100 # sensor distance (mm)
    n1 = x_target_L # number of pixels from the sensor border
    n2 = x_target_R # number of pixels from the sensor border
    d = 36/1920 # physical size of a pixel (mm)
    rx = 1920 #resolution of the camera along x

    # relative position of the target in respect to the center of the two cameras (mm)
    x_relative = f*T/(d*(n1-n2))
    y_relative = -(n1+n2-rx)*T/(2*(n1-n2))
    return x_relative, y_relative

def main():
    # Open the two stereo camera videos
    cap_L = cv2.VideoCapture("Left camera.mp4")
    cap_R = cv2.VideoCapture("Right camera.mp4")

    fps = 30
    delay = int(1000 / fps)

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

        combined = np.hstack((frame_L, frame_R))
        combined_resized = cv2.resize(combined, (1280, 360))
        cv2.imshow("Person recognition", combined_resized)

        # estimated position of the target
        x_relative, y_relative = target_estimation(x_target_L, x_target_R)
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


if __name__ == "__main__":
    main()