from ultralytics import YOLO
import cv2
import json
import os
import numpy as np
from collections import Counter

class VisionAgent:
    def __init__(self, frame_skip=5):
        print("Loading YOLOv8 Pose model...")
        self.model = YOLO('yolov8n-pose.pt') 
        self.frame_skip = frame_skip

    # --- POSTURE PHYSICS ---
    def get_posture_details(self, keypoints, person_id):
        if len(keypoints) == 0: return "Unknown", "No Data"
        try:
            lx, ly = keypoints[5][:2] # Shoulders
            rx, ry = keypoints[6][:2]
            l_hip_y, r_hip_y = keypoints[11][1], keypoints[12][1]

            shoulder_mid_y = (ly + ry) / 2
            hip_mid_y = (l_hip_y + r_hip_y) / 2
            
            torso_height = abs(hip_mid_y - shoulder_mid_y)
            shoulder_width = abs(lx - rx)
            if shoulder_width < 5: shoulder_width = 10 

            ratio = torso_height / shoulder_width
            posture = "Bending" if ratio < 1.1 else "Standing"
            
            return posture, f"P{person_id}: Ratio={ratio:.2f} -> {posture}"
        except:
            return "Unknown", "Error"

    def analyze_clip(self, video_path, output_dir, callback=None, deep_dive=False):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        clip_name = os.path.basename(video_path)
        
        # --- STATS CONTAINERS ---
        # Basic
        max_players = 0
        best_evidence_frame = None
        frame_player_counts = [] 
        all_actions = []         
        confidences = []         
        
        # Advanced Spatial
        raid_likelihood_scores = []
        defender_counts = []        
        formation_densities = []    
        raider_positions = []       # "Left Flank", "Right Flank", "Center"
        
        # New Future-Proof Metrics
        raider_x_history = []       # To calc movement intensity
        defense_widths = []         # To calc spread

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue
            
            if deep_dive and callback:
                callback({"type": "deep_log", "msg": f"--- Frame {frame_idx} Analysis ---"})

            results = self.model(frame, conf=0.4, verbose=False) 
            current_players = 0
            frame_x_positions = [] 
            annotated_frame = frame.copy()

            for result in results:
                annotated_frame = result.plot()
                if result.keypoints is not None:
                    kpts_batch = result.keypoints.data.cpu().numpy()
                    boxes = result.boxes

                    for i, kpts in enumerate(kpts_batch):
                        current_players += 1
                        
                        # 1. Posture
                        posture, log = self.get_posture_details(kpts, i)
                        all_actions.append(posture)
                        
                        # 2. Confidence
                        if boxes is not None:
                            confidences.append(float(boxes.conf[i]))

                        # 3. Spatial Anchor (Hip Center X)
                        hip_center_x = (kpts[11][0] + kpts[12][0]) / 2
                        frame_x_positions.append(hip_center_x)

                        if deep_dive and callback:
                            callback({"type": "deep_log", "msg": f"   > {log}"})

            frame_player_counts.append(current_players)

            # --- REFINED SPATIAL LOGIC (Attack Vector) ---
            scene_verdict = "Neutral"
            est_defenders = 0
            current_density = 0.0
            vector = "None"
            
            if current_players >= 3:
                frame_x_positions.sort()
                gaps = [frame_x_positions[j+1] - frame_x_positions[j] for j in range(len(frame_x_positions)-1)]
                max_gap = max(gaps) if gaps else 0
                frame_width = frame.shape[1]
                
                # Gap Threshold > 15% of width implies separation
                if max_gap > (frame_width * 0.15):
                    split_idx = gaps.index(max_gap) + 1
                    
                    group_a = frame_x_positions[:split_idx]
                    group_b = frame_x_positions[split_idx:]
                    
                    # Identify Raider (Smaller Group) vs Defense (Larger Group)
                    if len(group_a) < len(group_b):
                        raider_grp = group_a
                        defense_grp = group_b
                    else:
                        raider_grp = group_b
                        defense_grp = group_a

                    # Metrics
                    est_defenders = len(defense_grp)
                    raider_x = np.mean(raider_grp)
                    defense_min = np.min(defense_grp)
                    defense_max = np.max(defense_grp)
                    defense_w = defense_max - defense_min
                    
                    # --- ATTACK VECTOR LOGIC ---
                    if raider_x < defense_min:
                        vector = "Left Flank"
                    elif raider_x > defense_max:
                        vector = "Right Flank"
                    else:
                        # Raider is horizontally within the bounds of the defense
                        vector = "Center"
                        
                    # Store Metrics
                    raider_positions.append(vector)
                    raider_x_history.append(raider_x)
                    defense_widths.append(defense_w)
                    if len(defense_grp) > 1:
                        current_density = np.std(defense_grp)
                        formation_densities.append(current_density)
                    
                    scene_verdict = f"Active Raid ({vector})"
                    raid_likelihood_scores.append(1)
                else:
                    scene_verdict = "Defensive Setup"
                    raid_likelihood_scores.append(0)
                    est_defenders = current_players
                    if current_players > 1:
                        defense_widths.append(max(frame_x_positions) - min(frame_x_positions))
            else:
                raid_likelihood_scores.append(0)
            
            if est_defenders > 0: defender_counts.append(est_defenders)
            
            if deep_dive and callback and current_players > 0:
                callback({"type": "deep_log", "msg": f"   >>> Scene: {scene_verdict} | Vector: {vector}"})

            if current_players > max_players:
                max_players = current_players
                best_evidence_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            if callback and len(frame_player_counts) % 5 == 0:
                callback({"type": "progress", "clip": clip_name, "players": current_players, "frame": frame_idx})

            frame_idx += 1
            
        cap.release()

        # --- AGGREGATION & TYPE SAFEGUARDING ---
        if not frame_player_counts:
            median_players = 0; avg_conf = 0.0; bending_pct = 0.0; avg_defenders = 0
            avg_spread = 0.0; move_intensity = 0.0
        else:
            median_players = int(np.median(frame_player_counts))
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            
            action_cnt = Counter(all_actions)
            tot = sum(action_cnt.values())
            bending_pct = round((action_cnt.get("Bending", 0) / tot * 100), 1) if tot > 0 else 0.0
            
            # Defense Stats
            avg_defenders = int(np.median(defender_counts)) if defender_counts else 0
            if avg_defenders > 7: avg_defenders = 7 # Restriction
            
            # New Stats: Spread & Intensity
            avg_spread = float(np.mean(defense_widths)) if defense_widths else 0.0
            move_intensity = float(np.std(raider_x_history)) if len(raider_x_history) > 2 else 0.0

        avg_raid_score = float(np.mean(raid_likelihood_scores)) if raid_likelihood_scores else 0.0
        avg_density = float(np.mean(formation_densities)) if formation_densities else 0.0
        
        dominant_vector = "None"
        if raider_positions:
            dominant_vector = Counter(raider_positions).most_common(1)[0][0]

        final_scene = "Active Raid" if avg_raid_score > 0.4 else "Defensive Setup"
        
        if best_evidence_frame is not None and callback:
            callback({
                "type": "result", "clip": clip_name, 
                "players": int(median_players), "action": final_scene, 
                "evidence_frame": best_evidence_frame
            })

        # --- JSON OUTPUT (STRICT TYPES) ---
        return {
            "video_file": str(clip_name),
            "max_players_visible": int(max_players),
            "stats": {
                # Existing Fields
                "player_count_mode": int(median_players),
                "player_count_max": int(max_players),
                "bending_percentage": float(bending_pct),
                "avg_confidence": float(round(avg_conf, 2)),
                "defender_pack_size": int(avg_defenders),
                "formation_density_index": float(round(avg_density, 2)),
                "raid_confidence": float(round(avg_raid_score, 2)),
                
                # REFINED FIELD
                "attack_vector": str(dominant_vector), # Left Flank / Right Flank / Center
                
                # NEW EXTRA FIELDS
                "raider_movement_intensity": float(round(move_intensity, 2)), # High = Agility
                "defense_spread_avg": float(round(avg_spread, 2)) # in pixels
            },
            "classification": {
                "is_raid_likely": bool(avg_raid_score > 0.4),
                "scene_type": str(final_scene)
            }
        }

    def empty_response(self, path):
        return {
            "video_file": os.path.basename(path),
            "max_players_visible": 0,
            "stats": {"player_count_mode": 0},
            "classification": {"is_raid_likely": False}
        }

    def process_directory(self, input_dir, output_dir, callback=None):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        
        if callback: callback({"type": "log", "msg": f"Vision Agent: Found {len(files)} clips..."})

        for i, file in enumerate(files):
            is_deep_dive = (i < 2) 
            data = self.analyze_clip(os.path.join(input_dir, file), output_dir, callback, deep_dive=is_deep_dive)
            json_name = file.replace('.mp4', '.json')
            with open(os.path.join(output_dir, json_name), 'w') as f:
                json.dump(data, f, indent=4)
            
        if callback: callback({"type": "log", "msg": "✅ Vision Analysis Complete."})

if __name__ == "__main__":
    pass