import cv2
import os

def extract_frames(video_dir, output_dir):
    """
    Extract every frame from all videos in a directory.
    """

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory:", output_dir)

    # Get all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    if not video_files:
        print("No video files found in:", video_dir)
        return

    for video_file in video_files:

        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        print("Processing:", video_file)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error opening video:", video_file)
            continue

        frame_count = 0

        while True:

            ret, frame = cap.read()

            # Stop when video ends
            if not ret:
                break

            # Frame filename
            frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"

            frame_path = os.path.join(output_dir, frame_filename)

            # Save frame
            cv2.imwrite(frame_path, frame)

            frame_count += 1

        cap.release()

        print(f"Finished {video_file} → Extracted {frame_count} frames\n")

    print("All videos processed successfully!")



if __name__ == "__main__":

    INPUT_VIDEOS_FOLDER = "data/raw-videos-for-DL-model"
    OUTPUT_FRAMES_FOLDER = "data/extracted_frames"

    extract_frames(INPUT_VIDEOS_FOLDER, OUTPUT_FRAMES_FOLDER)