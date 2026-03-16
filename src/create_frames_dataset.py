import os
from frame_extraction import extract_frames

DATASET_PATH = "dataset"
OUTPUT_PATH = "dataset_frames"

classes = os.listdir(DATASET_PATH)

for activity in classes:

    input_folder = os.path.join(DATASET_PATH, activity)
    output_folder = os.path.join(OUTPUT_PATH, activity)

    os.makedirs(output_folder, exist_ok=True)

    videos = os.listdir(input_folder)

    for video in videos:

        video_path = os.path.join(input_folder, video)

        print("Processing:", video_path)

        extract_frames(video_path, output_folder)