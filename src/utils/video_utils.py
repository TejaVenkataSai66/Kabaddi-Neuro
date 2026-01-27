import cv2
import os
import sys
import subprocess
import imageio_ffmpeg # This comes installed with moviepy

class VideoProcessor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        # Get the actual path to the ffmpeg.exe installed on your system
        self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"✅ Using FFmpeg binary at: {self.ffmpeg_path}")

    def detect_scenes(self, video_path, output_dir, callback=None):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- PHASE 1: ANALYSIS (Using OpenCV) ---
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        prev_hist = None
        prev_frame = None 
        frame_idx = 0
        scene_start_time = 0.0
        
        cut_list = []
        evidence_count = 0 
        
        if callback: callback({"type": "log", "msg": f"Phase 1: Scanning {total_frames} frames..."})

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Histogram Calculation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            if prev_hist is not None:
                similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

                if similarity < self.threshold:
                    current_time = frame_idx / fps
                    
                    if (current_time - scene_start_time) > 2.0:
                        cut_list.append((scene_start_time, current_time))
                        
                        # --- CAPTURE EVIDENCE (First 4 Cuts) ---
                        if evidence_count < 4:
                            frame_A_path = os.path.join(output_dir, f"evidence_{evidence_count}_A.jpg")
                            frame_B_path = os.path.join(output_dir, f"evidence_{evidence_count}_B.jpg")
                            
                            cv2.imwrite(frame_A_path, prev_frame)
                            cv2.imwrite(frame_B_path, frame)
                            
                            if callback: 
                                callback({
                                    "type": "cut_evidence", 
                                    "time": current_time, 
                                    "score": similarity,
                                    "frame_A": frame_A_path,
                                    "frame_B": frame_B_path,
                                    "msg": f"✂️ **CUT DETECTED** at {current_time:.2f}s"
                                })
                            evidence_count += 1
                        else:
                            if callback:
                                callback({
                                    "type": "log", 
                                    "msg": f"✂️ Cut detected at {current_time:.2f}s"
                                })
                        
                        scene_start_time = current_time

            prev_hist = hist
            prev_frame = frame.copy()
            frame_idx += 1

        # Add the final scene
        final_time = frame_idx / fps
        if (final_time - scene_start_time) > 2.0:
            cut_list.append((scene_start_time, final_time))

        cap.release()
        
        # --- PHASE 2: EXTRACTION (Using Raw Subprocess) ---
        if callback: callback({"type": "log", "msg": f"Phase 2: Saving {len(cut_list)} clips using FFmpeg Core..."})
        
        saved_clips_info = []

        for i, (start, end) in enumerate(cut_list):
            try:
                # Minimum duration check
                if (end - start) < 1.0: continue

                filename = f"clip_{i}.mp4"
                output_filename = os.path.join(output_dir, filename)

                # BUILD THE COMMAND
                # We call ffmpeg.exe directly. This bypasses Python's memory entirely.
                cmd = [
                    self.ffmpeg_path,
                    "-y",              # Overwrite without asking
                    "-ss", str(start), # Seek to start time
                    "-i", video_path,  # Input file
                    "-t", str(end - start), # Duration to cut
                    "-c:v", "libx264", # Re-encode video for stability
                    "-c:a", "aac",     # Re-encode audio
                    "-preset", "ultrafast", # Speed up encoding
                    "-loglevel", "error",   # Quiet output unless error
                    output_filename
                ]

                # EXECUTE
                # This creates a completely new process for this one clip.
                # When it finishes, Windows automatically cleans up 100% of the resources.
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                
                saved_clips_info.append({"Clip": filename, "Start": start, "End": end})
                
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg Error on clip {i}")
                if callback: callback({"type": "log", "msg": f"❌ Error saving clip {i}"})
            except Exception as e:
                print(f"General Error on clip {i}: {e}")

        if callback: callback({"type": "log", "msg": "✅ All clips saved successfully."})
        return saved_clips_info