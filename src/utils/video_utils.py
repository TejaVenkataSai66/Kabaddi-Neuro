import os
import subprocess
import numpy as np
import librosa
import imageio_ffmpeg

class VideoProcessor:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"✅ Using FFmpeg binary at: {self.ffmpeg_path}")

    def extract_audio(self, video_path, audio_path):
        """Extracts audio from video for analysis using FFmpeg."""
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
            audio_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    def detect_whistle_timestamps(self, audio_path, callback=None):
        """
        Analyzes audio to find whistles and streams EXACT FREQUENCY values.
        """
        if callback: callback({"type": "log", "msg": "📊 Loading Audio for Frequency Analysis..."})
        
        y, sr = librosa.load(audio_path, sr=44100)
        
        # STFT
        D = np.abs(librosa.stft(y))
        frequencies = librosa.fft_frequencies(sr=sr)
        
        # Define Whistle Band (2000Hz - 5000Hz)
        whistle_band_mask = (frequencies > 2000) & (frequencies < 5000)
        band_frequencies = frequencies[whistle_band_mask]
        
        # Calculate Energy Profile
        whistle_energy = np.mean(D[whistle_band_mask, :], axis=0)
        background_energy = np.mean(D, axis=0) + 1e-6
        ratio = whistle_energy / background_energy
        ratio_smooth = np.convolve(ratio, np.ones(10)/10, mode='same')
        
        # --- NEW: DOMINANT FREQUENCY EXTRACTION ---
        # For every timeframe, find which Frequency (Hz) is the loudest
        band_magnitudes = D[whistle_band_mask, :]
        peak_freq_indices = np.argmax(band_magnitudes, axis=0)
        dominant_freqs = band_frequencies[peak_freq_indices]

        # Thresholding
        dynamic_thresh = max(3.0, np.mean(ratio_smooth) * 2)
        peaks = np.where(ratio_smooth > dynamic_thresh)[0]
        times = librosa.frames_to_time(peaks, sr=sr)
        
        detected_timestamps = []
        if len(times) > 0:
            last_time = times[0]
            detected_timestamps.append(last_time)
            
            for i, t in enumerate(times):
                frame_idx = peaks[i]
                current_freq = dominant_freqs[frame_idx] # The exact Hz
                current_conf = ratio_smooth[frame_idx]

                if (t - last_time) > 2.0: # Debounce
                    detected_timestamps.append(t)
                    last_time = t
                    
                    # STREAM LIVE NUMBERS (Step 1 Request)
                    if callback:
                        callback({
                            "type": "whistle_detected",
                            "time": t,
                            "freq": int(current_freq),  # e.g., 3450 Hz
                            "confidence": float(current_conf),
                            "msg": f"📢 Whistle: {int(current_freq)} Hz (Conf: {current_conf:.1f})"
                        })
        
        return detected_timestamps

    def detect_scenes(self, video_path, output_dir, callback=None):
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        temp_audio = os.path.join(output_dir, "temp_analysis_audio.wav")
        
        try:
            self.extract_audio(video_path, temp_audio)
            whistle_times = self.detect_whistle_timestamps(temp_audio, callback)
            
            cut_list = []
            start_time = 0.0
            
            for w_time in whistle_times:
                end_time = w_time + 5.0 
                cut_list.append((start_time, end_time))
                start_time = end_time 
            
            cut_list.append((start_time, start_time + 30.0)) 

            if callback: callback({"type": "log", "msg": f"💾 Saving {len(cut_list)} clips..."})
            
            saved_clips_info = []

            for i, (start, end) in enumerate(cut_list):
                duration = end - start
                if duration < 1.0: continue

                filename = f"clip_{i}.mp4"
                output_filename = os.path.join(output_dir, filename)

                cmd = [
                    self.ffmpeg_path, "-y",
                    "-ss", f"{start:.2f}",
                    "-i", video_path,
                    "-t", f"{duration:.2f}",
                    "-c:v", "libx264", "-c:a", "aac",
                    "-preset", "ultrafast",
                    "-loglevel", "error",
                    output_filename
                ]

                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                saved_clips_info.append({"Clip": filename, "Start": start, "End": end})
                
                if callback:
                    callback({"type": "progress", "clip": i, "status": "Saved"})

            if callback: callback({"type": "log", "msg": "✅ Whistle Segregation Complete."})
            return saved_clips_info

        except Exception as e:
            if callback: callback({"type": "log", "msg": f"❌ Error: {e}"})
            return []
        finally:
            if os.path.exists(temp_audio): os.remove(temp_audio)