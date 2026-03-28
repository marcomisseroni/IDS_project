# pip install ultralytics --break-system-packages
import cv2
from ultralytics import YOLO
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")


def detect_objects(frame):
    results = model(frame, verbose=False)
    detected_objects = []

    # we want the closest (biggest) person position
    x_target = 0
    y_target = 0
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                # drawing a circle in the box centroid
                x_c = int((x1+x2)/2)
                y_c = int((y1+y2)/2)
                cv2.circle(frame, (x_c, y_c), 3, (255,0,0), 3)
                # finding the biggest box area
                area = (x2-x1)*(y2-y1)
                if area > biggest_area:
                    biggest_area = area
                    x_target = x_c
                    y_target = y_c
                    cv2.circle(frame, (x_c, y_c), 3, (0,0,255), 3)

    return frame, detected_objects, x_target, y_target


def main():
    # Open webcam (or a video to test)
    cap = cv2.VideoCapture("Person walking.mp4")
    #cap = cv2.VideoCapture("http://10.196.188.59:4747")
    print("Webcam opened")

    while True:
        # loading the current frame
        ret, frame = cap.read()
        if not ret:
            print("ERROR IN CAPTURING")
            break

        # using YOLO to detect the target in the frame
        frame, detected_objects, x_target, y_target = detect_objects(frame)
        if detected_objects:
            print("Targhet:", x_target, y_target)
        else:
            print("No target detected")

        cv2.imshow("Person recognition", frame)
        if cv2.waitKey(20) == ord('q') or not ret:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()