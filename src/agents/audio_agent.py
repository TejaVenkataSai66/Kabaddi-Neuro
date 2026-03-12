import whisper
import os
import torch
import soundfile as sf
import librosa
import numpy as np
import time
import re
from moviepy import VideoFileClip

class AudioAgent:
    def __init__(self, model_size="medium"): 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)

    def clean_audio(self, input_path, output_path):
        try:
            # 1. Load Audio at Whisper's native sample rate
            data, rate = librosa.load(input_path, sr=16000)
            
            # 2. Advanced Vocal Isolation (Harmonic-Percussive Separation)
            # This strips away crowd noise, whistle peaks, and shoe squeaks
            S_full, phase = librosa.magphase(librosa.stft(data))
            S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=rate)))
            S_filter = np.minimum(S_full, S_filter)
            margin_v = 1.2
            power_v = 2
            mask_v = librosa.util.softmask(S_full - S_filter, margin_v * S_filter, power=power_v)
            S_vocal = mask_v * S_full
            vocal_audio = librosa.istft(S_vocal * phase)

            # 3. EXPERT SILENCE DETECTION (Anti-Hallucination Guard)
            # Calculate the acoustic energy. If it's too low, it's just noise.
            rms = librosa.feature.rms(y=vocal_audio)[0]
            mean_rms = np.mean(rms)
            
            # If the audio is practically silent, do not amplify it (which causes hallucinations)
            if mean_rms < 0.005: 
                normalized = vocal_audio
            else:
                max_val = np.max(np.abs(vocal_audio))
                normalized = vocal_audio / max_val
                
            sf.write(output_path, normalized, rate)
            return True, mean_rms
        except:
            return False, 0.0

    def extract_audio_from_video(self, video_path, temp_audio_path="temp_audio.wav"):
        try:
            video = VideoFileClip(video_path)
            if video.audio is None: return False
            video.audio.write_audiofile(temp_audio_path, logger=None, nbytes=2, ffmpeg_params=["-ac", "1"])
            video.close()
            return True
        except:
            return False

    def strict_referee_filter(self, raw_text):
        text = raw_text.lower()
        clean_text = re.sub(r'[^\w\s]', '', text)
        
        # --- EXPERT PHONETIC FILTER ---
        # Uses strict regex boundaries (\b) to prevent "on" from triggering "bonus"
        found = {
            "ceg": False, "actech": False, 
            "safe": False, "out": False, 
            "bonus": False, "point": False, 
            "super_tackle": False, "raider": False
        }
        
        # Team Names
        if re.search(r'\b(ceg|c e g|see gee|siege|seji|cg|cd|c d)\b', clean_text): found["ceg"] = True
        if re.search(r'\b(actech|ac tech|ak tech|ag tech|attack|active|arctic|altech|h tech)\b', clean_text): found["actech"] = True
        
        # Action States
        if re.search(r'\b(safe|save|survive|escape|say)\b', clean_text): found["safe"] = True
        if re.search(r'\b(out|catch|caught|tackle|fall|route|loud|dead)\b', clean_text): found["out"] = True
        
        # Scoring Events (Strictly curtailed to avoid false positives)
        if re.search(r'\b(bonus|bonas|ponus|bouns|monus)\b', clean_text): found["bonus"] = True
        if re.search(r'\b(point|one|won|score|coin|poin)\b', clean_text): found["point"] = True
        if re.search(r'\b(super tackle|super catch)\b', clean_text): found["super_tackle"] = True
        if re.search(r'\b(raider|rider|raid|player|red|right)\b', clean_text): found["raider"] = True

        # Resolve Logical Dependencies
        team = "CEG" if found["ceg"] else ("ACTECH" if found["actech"] else "")
        results = []
        
        if found["super_tackle"]:
            results.append(f"super tackle {team}".strip())
            
        if found["bonus"]:
            results.append(f"bonus point {team}".strip())
            
        action = ""
        if found["safe"]: action = "raider safe"
        elif found["out"]: action = "raider out"
        elif found["raider"] and not found["out"]: action = "raider safe"
        
        if found["point"] and team and not found["bonus"] and not found["super_tackle"]:
            if action:
                results.append(f"{action}, one point {team}")
            else:
                results.append(f"one point {team}")
        elif action and not found["bonus"] and not found["super_tackle"]:
            results.append(action)

        final_call = ", ".join(results)
        
        # Anti-Noise Filter: If no team was mentioned and the call is generic, it might be noise.
        # We only pass it through if it's a definitive match action.
        if not team and not action and not found["bonus"] and not found["super_tackle"]:
            return ""

        return final_call

    def analyze_clip(self, video_path, callback=None):
        clip_name = os.path.basename(video_path)
        raw_audio = "temp_raw.wav"
        clean_audio = "temp_clean.wav"
        
        has_audio = self.extract_audio_from_video(video_path, raw_audio)
        if not has_audio:
            return {"video_file": clip_name, "transcript": "", "referee_lines": []}

        success, mean_rms = self.clean_audio(raw_audio, clean_audio)
        
        # THE GOLDEN FIX: If acoustic energy is below human speech threshold, skip Whisper entirely!
        if mean_rms < 0.005:
            if callback: callback({"type": "deep_log", "msg": f"🔇 Silent clip detected. Bypassing AI translation."})
            return {"video_file": clip_name, "transcript": "", "referee_lines": []}

        final_audio = clean_audio if success else raw_audio

        # Prompt biases the neural network towards Kabaddi terms
        prompt_text = "Kabaddi referee calls: BONUS POINT CEG! BONUS POINT ACTECH! RAIDER SAFE, ONE POINT CEG! RAIDER OUT, ONE POINT ACTECH! SUPER TACKLE! EMPTY RAID!"
        
        # Expert Acoustic Thresholds applied
        result = self.model.transcribe(
            final_audio, 
            fp16=False,
            initial_prompt=prompt_text,
            language="en", 
            beam_size=5, 
            no_speech_threshold=0.6,          # Stricter rejection of crowd noise
            compression_ratio_threshold=2.4,  # Prevents AI looping/hallucinating text
            logprob_threshold=-1.0,
            condition_on_previous_text=False 
        )
        
        referee_lines = []
        unique_calls_set = set() 
        
        for segment in result['segments']:
            raw_text = segment['text'].strip()
            start = int(segment['start'])
            end = int(segment['end'])
            timestamp = f"[{start}s - {end}s]"
            
            strict_call = self.strict_referee_filter(raw_text)
            
            if strict_call and strict_call not in unique_calls_set:
                unique_calls_set.add(strict_call)
                formatted_line = f"{timestamp} {strict_call.upper()}"
                referee_lines.append(formatted_line)
                
                if callback:
                    callback({
                        "type": "segment",
                        "timestamp": timestamp,
                        "text": strict_call.upper(),
                        "is_referee": True,
                        "keyword": strict_call.upper()
                    })
                    time.sleep(0.05)

        for f in [raw_audio, clean_audio]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass 

        final_transcript = "\n".join(referee_lines)

        return {
            "video_file": clip_name,
            "transcript": final_transcript.lower(), 
            "referee_lines": referee_lines 
        }

    def process_directory(self, input_dir, output_dir, callback=None):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        
        for i, file in enumerate(files):
            if callback: callback({"type": "progress", "clip": file})
            data = self.analyze_clip(os.path.join(input_dir, file), callback)
            save_path = os.path.join(output_dir, file.replace('.mp4', '.txt'))
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("--- REFEREE ANNOUNCEMENTS (Strict Mode) ---\n")
                if data['referee_lines']:
                    for line in data['referee_lines']:
                        f.write(f"{line}\n")
                else:
                    f.write("No referee announcements detected.\n")

if __name__ == "__main__":
    agent = AudioAgent()
    agent.process_directory("data/processed_clips", "data/transcripts")