from src.utils.video_utils import VideoProcessor

# Initialize
splitter = VideoProcessor()

# Run (Make sure the file path matches your file!)
print("Starting processing...")
splitter.detect_scenes("data/raw_videos/test_match3.mp4", "data/processed_clips")