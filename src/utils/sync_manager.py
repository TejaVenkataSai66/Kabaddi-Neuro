import os
import json
import time

class SyncManager:
    def __init__(self):
        self.base_dir = "data"
        self.dirs = {
            "vision": os.path.join(self.base_dir, "metadata"),     # Raw Vision JSONs
            "audio": os.path.join(self.base_dir, "transcripts"),   # Raw Audio TXTs
            "output": os.path.join(self.base_dir, "unified_data")  # The Merged Result
        }
        
        if not os.path.exists(self.dirs["output"]):
            os.makedirs(self.dirs["output"])

    def load_json(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f: return json.load(f)
        return {}

    def load_txt(self, path):
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f: return f.read()
        return ""

    def parse_audio_file(self, content):
        data = {"full_transcript": "", "referee_lines": []}
        parts = content.split("--- REFEREE ANNOUNCEMENTS")
        
        raw_trans = parts[0].replace("--- FULL COMMENTARY ---", "").strip()
        data["full_transcript"] = raw_trans
        
        if len(parts) > 1:
            lines = parts[1].strip().split('\n')
            calls = [line.strip() for line in lines if line.strip() and not line.startswith("(")]
            data["referee_lines"] = calls
            
        return data

    def sync_all(self, callback=None):
        clips = [f for f in os.listdir(self.dirs["vision"]) if f.endswith('.json')]
        
        if callback: callback({"type": "log", "msg": f"🔄 Sync Manager: Found {len(clips)} visual records to align."})
        
        successful_syncs = 0
        
        for i, file in enumerate(clips):
            clip_id = file.replace('.json', '') 
            
            # 1. PATHS
            path_vision = os.path.join(self.dirs["vision"], file)
            path_audio = os.path.join(self.dirs["audio"], file.replace('.json', '.txt'))
            
            # 2. DEEP DIVE LOG: Loading
            if callback: callback({"type": "deep_log", "msg": f"[{clip_id}] Loading Vision Metadata..."})
            vision_data = self.load_json(path_vision)
            
            if os.path.exists(path_audio):
                if callback: callback({"type": "deep_log", "msg": f"[{clip_id}] Loading Audio Transcript..."})
                raw_audio = self.load_txt(path_audio)
                audio_parsed = self.parse_audio_file(raw_audio)
                audio_status = "✅ Audio Linked"
            else:
                if callback: callback({"type": "deep_log", "msg": f"[{clip_id}] ⚠️ No Audio File Found (Silent Clip)"})
                audio_parsed = {"full_transcript": "", "referee_lines": []}
                audio_status = "❌ No Audio"

            # 3. ALIGNMENT LOGIC
            classification = vision_data.get("classification", {})
            stats = vision_data.get("stats", {})
            zonal = vision_data.get("zonal_analysis", {}) # Fetch the Zonal block
            
            # Deep Dive: Show the fusion
            if callback: 
                fusion_msg = (
                    f"   >>> FUSING: Scene='{classification.get('scene_type')}' | "
                    f"Raid={classification.get('is_raid_likely')} | "
                    f"AudioLen={len(audio_parsed['full_transcript'])} chars"
                )
                callback({"type": "deep_log", "msg": fusion_msg})

            unified_object = {
                "clip_id": clip_id,
                "timestamp_sync": time.strftime("%Y-%m-%d %H:%M:%S"),
                "visual_context": {
                    "scene_class": classification.get("scene_type", "Unknown"),
                    "is_raid": classification.get("is_raid_likely", False),
                    "max_players": vision_data.get("max_players_visible", 0),
                    "tactical_metrics": stats,
                    "zonal_analysis": zonal # Make sure Zonal gets synced to output
                },
                "audio_context": {
                    "transcript": audio_parsed["full_transcript"],
                    "referee_events": audio_parsed["referee_lines"]
                },
                "status": "raw_ingestion"
            }
            
            # 4. SAVE
            save_path = os.path.join(self.dirs["output"], file)
            with open(save_path, 'w') as f:
                json.dump(unified_object, f, indent=4)
                
            if callback: 
                callback({"type": "progress", "clip": clip_id, "status": "Synced"})
                # Tiny pause for visual effect
                time.sleep(0.05)
                
            successful_syncs += 1

        if callback: callback({"type": "log", "msg": f"✅ Sync Complete. {successful_syncs} unified objects created."})

if __name__ == "__main__":
    syncer = SyncManager()
    syncer.sync_all()