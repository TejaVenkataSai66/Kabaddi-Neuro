import cv2
import os
import shutil

def setup_directories(base_dataset_dir):
    """Creates the strictly required PyTorch ImageFolder structure."""
    classes = ['active_raid', 'defensive_setup', 'skipped_trash']
    dirs = {}
    for cls in classes:
        path = os.path.join(base_dataset_dir, cls)
        os.makedirs(path, exist_ok=True)
        dirs[cls] = path
    return dirs

def rapid_annotate(input_frames_dir, base_dataset_dir):
    """
    A high-speed, keyboard-driven OpenCV GUI to sort frames into classes.
    """
    print("--- 🏆 KABADDI RAPID ANNOTATOR ---")
    print("CONTROLS:")
    print(" Press 'a' -> Classify as ACTIVE RAID")
    print(" Press 'd' -> Classify as DEFENSIVE SETUP")
    print(" Press 's' -> SKIP / TRASH (Blurry, empty, or transitional frame)")
    print(" Press 'q' -> QUIT and save progress")
    print("----------------------------------")

    dirs = setup_directories(base_dataset_dir)
    
    # Get all unannotated frames
    frames = [f for f in os.listdir(input_frames_dir) if f.endswith(('.jpg', '.png'))]
    total_frames = len(frames)
    
    if total_frames == 0:
        print("No frames found! Please run Task 1.1 (frame_extractor.py) first.")
        return

    annotated_count = 0

    for idx, frame_name in enumerate(frames):
        source_path = os.path.join(input_frames_dir, frame_name)
        
        # Load and resize image for viewing
        img = cv2.imread(source_path)
        if img is None: continue
        
        view_img = cv2.resize(img, (1080, 640)) 
        
        # Add UI Text overlay
        cv2.putText(view_img, f"Frame: {idx+1}/{total_frames} | Annotated: {annotated_count}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(view_img, "[A]ctive Raid | [D]efensive Setup | [S]kip | [Q]uit", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Kabaddi Neuro - Rapid Annotator", view_img)

        # Wait for user keyboard input
        key = cv2.waitKey(0) & 0xFF

        target_dir = None
        if key == ord('a'):
            target_dir = dirs['active_raid']
            print(f"[{annotated_count+1}] Moved to ACTIVE RAID: {frame_name}")
        elif key == ord('d'):
            target_dir = dirs['defensive_setup']
            print(f"[{annotated_count+1}] Moved to DEFENSIVE SETUP: {frame_name}")
        elif key == ord('s'):
            target_dir = dirs['skipped_trash']
            print(f"[{annotated_count+1}] SKIPPED: {frame_name}")
        elif key == ord('q'):
            print("\nAnnotation paused. Quitting...")
            break
        else:
            print("Invalid key. Skipping frame...")
            target_dir = dirs['skipped_trash']

        # Physically move the file to the new classification folder
        if target_dir:
            destination_path = os.path.join(target_dir, frame_name)
            shutil.move(source_path, destination_path)
            annotated_count += 1

    cv2.destroyAllWindows()
    print(f"\n✅ Session Complete! You annotated {annotated_count} frames.")
    
    # Print dataset balance stats
    raid_count = len(os.listdir(dirs['active_raid']))
    def_count = len(os.listdir(dirs['defensive_setup']))
    print(f"📊 Current Dataset Balance -> Active Raid: {raid_count} | Defensive Setup: {def_count}")

# --- Execution Block ---
if __name__ == "__main__":
    # The folder where Task 1.1 saved the images
    INPUT_FRAMES = "data/extracted_frames" 
    
    # The master folder for your PyTorch dataset
    DATASET_OUTPUT = "data/cnn_dataset"     
    
    rapid_annotate(INPUT_FRAMES, DATASET_OUTPUT)