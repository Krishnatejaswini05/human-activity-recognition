import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=5):

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    saved = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_skip == 0:

            frame = cv2.resize(frame,(224,224))

            filename = os.path.join(output_folder,f"frame_{saved}.jpg")

            cv2.imwrite(filename, frame)

            saved += 1

        frame_id += 1

    cap.release()