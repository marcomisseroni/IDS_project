import cv2
import sys
import os

def split_video(input_path, output_left, output_right):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        return

    # Ottieni proprietà del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Risoluzione: {width}x{height}, FPS: {fps}")

    half_width = width // 2

    # Codec (mp4 compatibile)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out_left = cv2.VideoWriter(output_left, fourcc, fps, (half_width, height))
    out_right = cv2.VideoWriter(output_right, fourcc, fps, (half_width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Split frame
        left_frame = frame[:, :half_width]
        right_frame = frame[:, half_width:]

        # Scrivi nei video
        out_left.write(left_frame)
        out_right.write(right_frame)

    cap.release()
    out_left.release()
    out_right.release()

    print("Operazione completata!")

if __name__ == "__main__":

    input_video = "test_videos/RGBD.MP4"
    base, ext = os.path.splitext(input_video)
    output_left = base + "_left.mp4"
    output_right = base + "_right.mp4"

    split_video(input_video, output_left, output_right)