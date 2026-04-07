"""
audio_agent.py  –  WhisperX-based Kabaddi Referee Transcription Agent
State Sports Board Edition (Dynamic Teams Update)

Improvements:
  - Dynamically accepts ANY team names (e.g., team1="BULLS", team2="WARRIORS").
  - Dynamically updates Whisper's vocabulary prompt.
  - Accepts custom phonetic alias lists to correct AI stadium hallucinations.
  - Safely defaults to CEG/ACTECH for backward compatibility.
"""

import os
import gc
import re
import time
import logging

import torch
import soundfile as sf
import numpy as np

try:
    from moviepy import VideoFileClip as _VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip as _VideoFileClip

try:
    import whisperx
except ImportError as exc:
    raise ImportError(
        "whisperx is required.  Install it with:\n  pip install whisperx"
    ) from exc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AudioAgent")

class AudioAgent:
    def __init__(
        self,
        model_size: str = "small",
        language: str = "en",
        process_full_video: bool = True,
        tail_seconds: float = 10.0,
        team1: str = "CEG",             # ✨ NEW: Dynamic Team 1
        team2: str = "ACTECH",          # ✨ NEW: Dynamic Team 2
        team1_aliases: list = None,     # ✨ NEW: Custom misspellings for Team 1
        team2_aliases: list = None      # ✨ NEW: Custom misspellings for Team 2
    ):
        log.info(f"=== 🎤 Initialising WhisperX Kabaddi Agent ({team1} vs {team2}) ===")
        self.language = language
        self.process_full_video = process_full_video
        self.tail_seconds = tail_seconds

        # Store dynamic team names (lowercased for regex processing)
        self.team1 = team1.lower().strip()
        self.team2 = team2.lower().strip()

        # Handle Phonetic Aliases
        self.team1_aliases = team1_aliases or [self.team1, self.team1.replace(" ", ""), " ".join(self.team1)]
        self.team2_aliases = team2_aliases or [self.team2, self.team2.replace(" ", ""), " ".join(self.team2)]

        # Preserve the specific hardcoded misspellings for CEG/ACTECH if they are the active teams
        if team1.upper() == "CEG" and team1_aliases is None:
            self.team1_aliases = ["ceg", "c e g", "see gee", "seji", "cee gee", "cg", "c g"]
        if team2.upper() == "ACTECH" and team2_aliases is None:
            self.team2_aliases = ["actech", "ac tech", "ak tech", "ag tech", "altech", "a c tech", "attack", "active", "arctic", "ac", "a c"]

        # Sort aliases by length descending so longer phrases are replaced first
        self.team1_aliases.sort(key=len, reverse=True)
        self.team2_aliases.sort(key=len, reverse=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        log.info("Device: %s  |  Compute: %s", self.device, self.compute_type)

        # ✨ NEW: Dynamically build the prompt to bias Whisper's vocabulary
        self.initial_prompt = (
            f"Kabaddi referee announcement. {team1.upper()} versus {team2.upper()}. "
            "Vocabulary: kabaddi, raider safe, raider out, bonus point, bonus plus, "
            "super tackle, all out, all in, one point, two points, three points, "
            f"{team1.upper()}, {team2.upper()}, lobby, do or die raid, empty raid, tackle, touch."
        )

        log.info("Loading Faster-Whisper (%s) …", model_size)
        self.model = whisperx.load_model(
            model_size,
            self.device,
            compute_type=self.compute_type,
            language=language,
            asr_options={
                "initial_prompt": self.initial_prompt,
                "beam_size": 5,
                "best_of": 5,
            },
            vad_options={"vad_onset": 0.01, "vad_offset": 0.01}
        )

        log.info("Loading Wav2Vec2 alignment model …")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )

        log.info("✅  Agent ready!")

    def extract_audio(self, video_path: str, out_path: str = "temp_audio.wav") -> bool:
        try:
            video = _VideoFileClip(video_path)
        except Exception as exc:
            log.error("Could not open video %s: %s", video_path, exc)
            return False

        if video.audio is None:
            log.warning("No audio stream found in %s", video_path)
            video.close()
            return False

        if not self.process_full_video:
            duration = video.duration
            if duration > self.tail_seconds:
                try:
                    video = video.subclipped(duration - self.tail_seconds, duration)
                except AttributeError:
                    video = video.subclip(duration - self.tail_seconds, duration)
            log.info("Processing last %.1fs of clip", min(duration, self.tail_seconds))
        else:
            log.info("Processing full clip (%.2fs)", video.duration)

        ffmpeg_params = [
            "-ac", "1",
            "-ar", "16000",
        ]

        try:
            video.audio.write_audiofile(
                out_path,
                logger=None,
                nbytes=2,
                ffmpeg_params=ffmpeg_params,
            )
        except Exception as exc:
            log.error("Audio write failed: %s", exc)
            video.close()
            return False
        finally:
            video.close()

        return True

    def parse_referee_call(self, raw_text: str) -> str:
        if not raw_text or not raw_text.strip():
            return ""

        text = raw_text.lower().strip()
        clean = re.sub(r"[^\w\s\,]", " ", text)

        hallucinations = ["thank you", "thanks for", "subscribe", "mbn", "www", "bootball", "hey love"]
        if any(h in clean for h in hallucinations):
            return ""

        words = clean.replace(",", "").split()
        if len(words) > 5 and len(set(words)) == 1:
            return ""

        # ✨ NEW: Dynamic Phonetic Replacements for Team 1
        t1_pattern = r"\b(" + "|".join([re.escape(a) for a in self.team1_aliases]) + r")\b"
        clean = re.sub(t1_pattern, self.team1, clean)

        # ✨ NEW: Dynamic Phonetic Replacements for Team 2
        t2_pattern = r"\b(" + "|".join([re.escape(a) for a in self.team2_aliases]) + r")\b"
        clean = re.sub(t2_pattern, self.team2, clean)

        # Standard Kabaddi Terminology Replacements
        clean = re.sub(r"\b(save|say|seif)\b", "safe", clean)
        clean = re.sub(r"\b(aut|catch|caught|fall|route|loud|dead|ought|how|howd)\b", "out", clean)
        clean = re.sub(r"\b(super\s+catch)\b", "super tackle", clean)
        clean = re.sub(r"\b(bonas|ponus|bouns|monus)\b", "bonus", clean)
        clean = re.sub(r"\b(plas|pless|place|please)\b", "plus", clean)
        clean = re.sub(r"\b(tree)\b", "three", clean)
        clean = re.sub(r"\b(to|too)\b", "two", clean)
        clean = re.sub(r"\b(won|score|coin|poin|boy)\b", "point", clean)

        # ✨ NEW: Dynamic phrase combinations ("point [TEAM]")
        clean = re.sub(rf"\bpoint\s+({re.escape(self.team1)}|{re.escape(self.team2)})\b", r"one point \1", clean)
        
        if re.search(r"\bone\s+plus\s+two\b", clean) and not re.search(r"three\s+points?", clean):
            clean = re.sub(r"\bone\s+plus\s+two\b", "one plus two three points", clean)

        # ✨ NEW: Dynamic valid keyword verification
        valid_keywords = ["safe", "out", "bonus", "tackle", "point", "points", "all out", "all in", self.team1, self.team2, "kabaddi"]
        if not any(k in clean for k in valid_keywords):
            return ""

        clean = re.sub(r"\s+", " ", clean).strip()
        clean = re.sub(r"\s*,\s*", ", ", clean)
        clean = re.sub(r"^,|,$", "", clean).strip()

        return clean.upper()

    @staticmethod
    def resolve_conflicts(referee_lines: list[str]) -> list[str]:
        has_safe = any("SAFE" in ln for ln in referee_lines)
        has_out = any(
            "OUT" in ln and "ALL OUT" not in ln and "RAIDER OUT" not in ln.split(", ")[0]
            for ln in referee_lines
        ) or any("RAIDER OUT" in ln for ln in referee_lines)

        if not (has_safe and has_out):
            return referee_lines

        last_is_out = False
        for ln in reversed(referee_lines):
            if "RAIDER OUT" in ln:
                last_is_out = True
                break
            if "RAIDER SAFE" in ln:
                last_is_out = False
                break

        resolved = []
        for ln in referee_lines:
            if last_is_out and "RAIDER SAFE" in ln: continue
            if not last_is_out and "RAIDER OUT" in ln and "ALL OUT" not in ln: continue
            resolved.append(ln)
        return resolved

    def analyze_clip(self, video_path: str, callback=None) -> dict:
        clip_name = os.path.basename(video_path)
        temp_audio = f"_tmp_audio_{os.getpid()}.wav"

        log.info("[%s]  Extracting audio …", clip_name)
        if not self.extract_audio(video_path, temp_audio):
            log.warning("[%s]  No audio — skipping.", clip_name)
            return {"video_file": clip_name, "transcript": "", "referee_lines": []}

        try:
            audio = whisperx.load_audio(temp_audio)
        except Exception as exc:
            log.error("[%s]  whisperx.load_audio failed: %s", clip_name, exc)
            self._cleanup(temp_audio)
            return {"video_file": clip_name, "transcript": "", "referee_lines": []}

        log.info("[%s]  Running Faster-Whisper transcription …", clip_name)
        try:
            result = self.model.transcribe(
                audio,
                batch_size=4,          
                language=self.language,
                print_progress=False,
            )
        except Exception as exc:
            log.error("[%s]  Transcription failed: %s", clip_name, exc)
            self._cleanup(temp_audio)
            return {"video_file": clip_name, "transcript": "", "referee_lines": []}

        if not result.get("segments"):
            log.info("[%s]  No speech detected.", clip_name)
            self._cleanup(temp_audio)
            return {"video_file": clip_name, "transcript": "", "referee_lines": []}

        log.info("[%s]  Running Wav2Vec2 alignment …", clip_name)
        try:
            aligned = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            segments = aligned["segments"]
        except Exception as exc:
            log.warning(
                "[%s]  Alignment failed (%s) — using raw Whisper segments.", clip_name, exc
            )
            segments = result["segments"]

        log.info("[%s]  Filtering referee calls …", clip_name)
        referee_lines: list[str] = []
        raw_transcripts: list[str] = []

        start_time = 999.0
        end_time = 0.0

        for seg in segments:
            raw_text = seg.get("text", "").strip()
            if not raw_text: continue
            
            raw_transcripts.append(raw_text)
            s = round(seg.get("start", 0.0), 2)
            e = round(seg.get("end", s + 1.0), 2)
            
            start_time = min(start_time, s)
            end_time = max(end_time, e)

            if callback:
                callback({
                    "type": "segment",
                    "timestamp": f"[{s}s - {e}s]",
                    "text": raw_text,
                    "is_referee": False,
                    "keyword": raw_text,
                })
                time.sleep(0.05)

        full_raw_text = " ".join(raw_transcripts).strip()
        
        if full_raw_text:
            if start_time == 999.0: start_time = 0.0
            if end_time == 0.0: end_time = 6.0
            
            ts = f"[{round(start_time, 1)}s - {round(end_time, 1)}s]"
            call = self.parse_referee_call(full_raw_text)
            
            if call:
                formatted = f"{ts} {call.upper()}"
                referee_lines.append(formatted)
                log.info("  CALL  %s", formatted)
                
                if callback:
                    callback({
                        "type": "segment",
                        "timestamp": ts,
                        "text": call.upper(),
                        "is_referee": True,
                        "keyword": call.upper(),
                    })
                    time.sleep(0.05)

        referee_lines = self.resolve_conflicts(referee_lines)
        self._cleanup(temp_audio)

        log.info("[%s]  Done — %d referee call(s) detected.", clip_name, len(referee_lines))
        
        return {
            "video_file": clip_name,
            "transcript": full_raw_text.lower(),
            "referee_lines": referee_lines,
        }

    def process_directory(self, input_dir: str, output_dir: str, callback=None) -> None:
        os.makedirs(output_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".mp4"))

        if not files:
            log.warning("No .mp4 files found in %s", input_dir)
            return

        log.info("Processing %d clip(s) …", len(files))

        for i, fname in enumerate(files, start=1):
            log.info("─── Clip %d / %d : %s ───", i, len(files), fname)

            if callback:
                callback({"type": "progress", "clip": fname, "index": i, "total": len(files)})

            data = self.analyze_clip(os.path.join(input_dir, fname), callback)

            out_path = os.path.join(output_dir, fname.replace(".mp4", ".txt"))
            self._write_result(out_path, data)

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

    @staticmethod
    def _write_result(out_path: str, data: dict) -> None:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("╔══════════════════════════════════════════╗\n")
            fh.write("║   REFEREE ANNOUNCEMENTS  (Strict Mode)   ║\n")
            fh.write("╚══════════════════════════════════════════╝\n\n")
            if data["referee_lines"]:
                for line in data["referee_lines"]:
                    fh.write(f"  {line}\n")
            else:
                fh.write("  No distinct referee calls detected.\n")
            fh.write("\n\n── FULL RAW TRANSCRIPT ──────────────────────\n")
            fh.write(data["transcript"] or "(empty)" )
            fh.write("\n")
        log.info("Result saved → %s", out_path)

    @staticmethod
    def _cleanup(path: str) -> None:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

if __name__ == "__main__":
    # Example usage for dynamic teams
    agent = AudioAgent(team1="PATNA", team2="MUMBA")
    agent.process_directory("../../data/processed_clips", "../../data/transcripts")