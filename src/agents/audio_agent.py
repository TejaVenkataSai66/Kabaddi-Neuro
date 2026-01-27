import whisper
import os
import torch
import noisereduce as nr
import soundfile as sf
import librosa
import time
from moviepy import VideoFileClip

class AudioAgent:
    def __init__(self, model_size="small"): 
        print(f"Loading Whisper '{model_size}' model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)
        print(f"Model loaded on {device}.")
        
        self.referee_keywords = [
            "one point", "two points", "three points", "super tackle", 
            "all out", "review successful", "review unsuccessful", 
            "technical point", "bonus point", "green card", "yellow card",
            "jersey number", "do or die", "timeout", "raid over", "defender out",
            "whistle", "line out", "touch point", "raider out", "raider safe"
        ]

    def clean_audio(self, input_path, output_path):
        try:
            data, rate = librosa.load(input_path, sr=None)
            reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8, stationary=True)
            sf.write(output_path, reduced_noise, rate)
            return True
        except Exception as e:
            return False

    def extract_audio_from_video(self, video_path, temp_audio_path="temp_audio.wav"):
        try:
            video = VideoFileClip(video_path)
            if video.audio is None: return False
            video.audio.write_audiofile(temp_audio_path, logger=None)
            video.close()
            return True
        except Exception as e:
            return False

    # FIX: Added 'callback' parameter here
    def analyze_clip(self, video_path, callback=None):
        clip_name = os.path.basename(video_path)
        raw_audio = "temp_raw.wav"
        clean_audio = "temp_clean.wav"
        
        # 1. VISUALIZE EXTRACTION
        if callback: callback({"type": "log", "msg": f"🔊 Extracting audio track from {clip_name}..."})
        has_audio = self.extract_audio_from_video(video_path, raw_audio)
        
        if not has_audio:
            if callback: callback({"type": "log", "msg": f"❌ No audio found in {clip_name}"})
            return {"video_file": clip_name, "transcript": "", "referee_lines": []}

        # 2. VISUALIZE DENOISING
        if callback: callback({"type": "log", "msg": "📉 Applying Spectral Gating (Crowd Noise Reduction)..."})
        success = self.clean_audio(raw_audio, clean_audio)
        final_audio = clean_audio if success else raw_audio

        # 3. VISUALIZE TRANSCRIPTION
        if callback: callback({"type": "log", "msg": "🧠 Running Whisper Inference (Speech-to-Text)..."})
        
        prompt_text = "Referee announcements: One point, Jersey number, Super Tackle, Review."
        
        result = self.model.transcribe(
            final_audio, 
            fp16=False,
            initial_prompt=prompt_text,
            language="en"
        )
        
        full_transcript = result['text'].strip()
        referee_lines = []
        
        # 4. SEGMENT SCANNING (The "Intermediate Steps")
        for segment in result['segments']:
            text = segment['text'].strip()
            start = int(segment['start'])
            end = int(segment['end'])
            timestamp = f"[{start}s - {end}s]"
            
            # Check for keywords
            is_referee = False
            detected_kw = ""
            text_lower = text.lower()
            
            for kw in self.referee_keywords:
                if kw in text_lower:
                    is_referee = True
                    detected_kw = kw
                    referee_lines.append(f"{timestamp} {text}")
                    break 
            
            # SEND LIVE UPDATE TO UI
            if callback:
                callback({
                    "type": "segment",
                    "timestamp": timestamp,
                    "text": text,
                    "is_referee": is_referee,
                    "keyword": detected_kw
                })
                # Tiny sleep so you can see the text scrolling in the UI
                time.sleep(0.1)

        # Cleanup
        for f in [raw_audio, clean_audio]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass 

        return {
            "video_file": clip_name,
            "transcript": full_transcript,
            "referee_lines": referee_lines 
        }

    # FIX: Added 'callback' parameter here too
    def process_directory(self, input_dir, output_dir, callback=None):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        
        if callback: callback({"type": "log", "msg": f"Audio Agent: Found {len(files)} clips to listen to..."})

        for i, file in enumerate(files):
            if callback: callback({"type": "progress", "clip": file})
            
            # Pass the callback down to analyze_clip
            data = self.analyze_clip(os.path.join(input_dir, file), callback)
            
            save_path = os.path.join(output_dir, file.replace('.mp4', '.txt'))
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("--- FULL COMMENTARY ---\n")
                f.write(data['transcript'])
                f.write("\n\n--- REFEREE ANNOUNCEMENTS (Full Context) ---\n")
                if data['referee_lines']:
                    for line in data['referee_lines']:
                        f.write(f"{line}\n")
                else:
                    f.write("No referee announcements detected.\n")
                    
        if callback: callback({"type": "log", "msg": "✅ Audio Analysis Complete."})

if __name__ == "__main__":
    agent = AudioAgent()
    agent.process_directory("data/processed_clips", "data/transcripts")