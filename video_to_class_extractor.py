import cv2
import os

def extract_videos_to_classes(raw_videos_base_dir, final_dataset_dir, frame_skip=3):
    """
    Automatically extracts frames from categorized video folders directly into CNN class folders.
    """
    classes = [
        '1_defender', '2_defenders', '3_defenders', 
        '4_defenders', '5_defenders', '6_defenders', '7_defenders'
    ]

    print("--- 🛡️ AUTOMATED DEFENDER DATASET BUILDER ---")

    for cls in classes:
        input_video_dir = os.path.join(raw_videos_base_dir, cls)
        output_image_dir = os.path.join(final_dataset_dir, cls)

        # Create the final output folder for this class
        os.makedirs(output_image_dir, exist_ok=True)

        # Skip if the raw video folder doesn't exist or is empty
        if not os.path.exists(input_video_dir):
            continue
            
        video_files = [f for f in os.listdir(input_video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        if not video_files:
            continue

        print(f"\n📁 Processing Class: {cls.upper()} ({len(video_files)} videos found)")
        total_saved_for_class = 0

        for video_file in video_files:
            video_path = os.path.join(input_video_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  ❌ Error opening: {video_file}")
                continue

            current_frame = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret: break
                
                # We use a small skip to prevent saving 300 identical frames of people standing still
                if current_frame % frame_skip == 0:
                    frame_filename = f"{video_name}_frame_{current_frame:04d}.jpg"
                    frame_path = os.path.join(output_image_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1
                    total_saved_for_class += 1

                current_frame += 1

            cap.release()
            print(f"  ✅ Extracted {saved_count} frames from {video_file}")

        print(f"📊 Total images generated for {cls}: {total_saved_for_class}")

    print("\n🚀 Dataset generation complete! Ready for PyTorch training.")

# --- Execution Block ---
if __name__ == "__main__":
    # Point this to where you sorted your raw MP4 clips
    RAW_VIDEOS_DIR = "data/raw_defender_clips"     
    
    # This will be your final dataset folder for CNN training
    FINAL_DATASET_DIR = "data/defender_counting_dataset"     
    
    # frame_skip=3 means it extracts ~10 frames per second (assuming 30fps video).
    # If you want EVERY single frame, change frame_skip=1.
    extract_videos_to_classes(RAW_VIDEOS_DIR, FINAL_DATASET_DIR, frame_skip=3)