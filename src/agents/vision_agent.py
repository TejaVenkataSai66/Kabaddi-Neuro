import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import cv2
from PIL import Image
import json
import os
import numpy as np
from collections import Counter

class VisionAgent:
    def __init__(self, frame_skip=5):
        print("--- 🧠 INITIALIZING TRI-MODEL VISION AGENT ---")
        self.frame_skip = frame_skip
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.pose_model = YOLO('yolov8n-pose.pt') 
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.scene_cnn = models.resnet18(weights=None)
        self.scene_cnn.fc = nn.Linear(self.scene_cnn.fc.in_features, 2)
        self.scene_cnn.load_state_dict(torch.load('scene_classifier.pth', map_location=self.device, weights_only=True))
        self.scene_cnn = self.scene_cnn.to(self.device)
        self.scene_cnn.eval()

        self.counter_cnn = models.resnet18(weights=None)
        self.counter_cnn.fc = nn.Linear(self.counter_cnn.fc.in_features, 7)
        self.counter_cnn.load_state_dict(torch.load('defender_counter_cnn.pth', map_location=self.device, weights_only=True))
        self.counter_cnn = self.counter_cnn.to(self.device)
        self.counter_cnn.eval()

    def predict_cnns(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            scene_out = self.scene_cnn(input_tensor)
            _, predicted_scene = torch.max(scene_out.data, 1)
            scene_class = "Active Raid" if predicted_scene.item() == 0 else "Defensive Setup"

            count_out = self.counter_cnn(input_tensor)
            _, predicted_count = torch.max(count_out.data, 1)
            defenders_count = predicted_count.item() + 1 

        return scene_class, defenders_count

    def analyze_clip(self, video_path, output_dir, callback=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        midline_x = frame_width / 2.0
        
        frame_idx = 0
        ui_frames_shown = 0 
        clip_name = os.path.basename(video_path)
        
        yolo_exact_counts = []
        cnn_scene_tags = []
        cnn_defender_counts = [] 
        raider_x_history = []
        raw_timeline = []
        
        frame_player_counts = []
        confidences = []
        formation_densities = []
        defense_widths = []
        raider_positions = []
        
        raid_likelihood_scores = []
        all_actions = []
        
        current_zone, current_action = "Center", "Scanning"
        deepest_px, past_baulk_frames = 0.0, 0
        zone_counts = {"left_corner": 0, "left_in": 0, "center": 0, "right_in": 0, "right_corner": 0}

        # ✨ TRACKING LOCK
        tracked_raider_box = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            cnn_scene, cnn_defenders = self.predict_cnns(frame)
            cnn_scene_tags.append(cnn_scene)
            cnn_defender_counts.append(cnn_defenders)

            results = self.pose_model(frame, conf=0.4, verbose=False)
            player_data = [] 
            player_kpts_map = {} 
            player_boxes = {}
            noses_on_defense_side = 0
            frame_conf = []

            for result in results:
                if result.keypoints is not None:
                    kpts_batch = result.keypoints.data.cpu().numpy()
                    boxes = result.boxes.data.cpu().numpy()
                    for i, kpts in enumerate(kpts_batch):
                        if len(kpts) > 0:
                            if i < len(boxes): 
                                frame_conf.append(boxes[i][4])
                                box = boxes[i][:4] 
                            
                            nose_x = kpts[0][0]
                            hip_center_x = (kpts[11][0] + kpts[12][0]) / 2
                            
                            if nose_x > 0:
                                player_data.append(hip_center_x)
                                player_kpts_map[hip_center_x] = kpts
                                player_boxes[hip_center_x] = box
                                
                                if (cnn_scene == "Active Raid" and nose_x > midline_x) or (cnn_scene == "Defensive Setup"):
                                    noses_on_defense_side += 1

            if frame_conf: confidences.append(np.mean(frame_conf))
            frame_player_counts.append(len(player_data))
            if noses_on_defense_side > 0: yolo_exact_counts.append(min(noses_on_defense_side, 7))

            action_type = current_action
            current_raider_x = None  
            current_raider_kpts = None 
            
            if len(player_data) >= 3:
                # ✨ 100% LOCKED RAIDER LOGIC: Anchor to the Far Left Person
                if tracked_raider_box is None:
                    raider_x = min(player_data) # Finds the leftmost person entering the screen
                    tracked_raider_box = player_boxes[raider_x]
                
                best_match_score = float('inf')
                best_x = None
                
                tracked_center_x = (tracked_raider_box[0] + tracked_raider_box[2]) / 2
                tracked_center_y = (tracked_raider_box[1] + tracked_raider_box[3]) / 2

                # Euclidean Center-of-Mass Tracking
                for px in player_data:
                    curr_box = player_boxes[px]
                    curr_center_x = (curr_box[0] + curr_box[2]) / 2
                    curr_center_y = (curr_box[1] + curr_box[3]) / 2
                    
                    dist = ((tracked_center_x - curr_center_x)**2 + (tracked_center_y - curr_center_y)**2)**0.5
                    
                    if dist < best_match_score:
                        best_match_score = dist
                        best_x = px
                
                if best_x is not None and best_match_score < (frame_width * 0.15):
                    tracked_raider_box = player_boxes[best_x]
                    raider_x = best_x
                else:
                    raider_x = min(player_data)
                    tracked_raider_box = player_boxes[raider_x]

                defense_grp = [x for x in player_data if x != raider_x]
                
                if defense_grp:
                    defense_min, defense_max = min(defense_grp), max(defense_grp)
                    
                    current_raider_x = raider_x
                    current_raider_kpts = player_kpts_map.get(raider_x, None)
                    
                    defense_widths.append(abs(defense_max - defense_min))
                    formation_densities.append(abs(defense_max - defense_min) / len(defense_grp))

                    dir_mult = 1 if raider_x < defense_min else -1 if raider_x > defense_max else 0
                    raider_x_history.append(raider_x)

                    p = raider_x / frame_width
                    z = "left_corner" if p < 0.2 else "left_in" if p < 0.4 else "center" if p < 0.6 else "right_in" if p < 0.8 else "right_corner"
                    zone_counts[z] += 1
                    raider_positions.append("Left Flank" if p < 0.5 else "Right Flank")
                    
                    action_type = "Scanning"
                    if len(raider_x_history) >= 2 and dir_mult != 0:
                        velocity = (raider_x_history[-1] - raider_x_history[-2]) * dir_mult
                        engage_threshold = frame_width * 0.015 
                        retreat_threshold = -(frame_width * 0.015)

                        if velocity > engage_threshold: action_type = "Engaging"
                        elif velocity < retreat_threshold: action_type = "Retreating"
                        
                    raid_likelihood_scores.append(0.8 if action_type == "Engaging" else 0.2)
                    all_actions.append(action_type)

                    penetration = (raider_x - midline_x) if dir_mult == 1 else (midline_x - raider_x)
                    if penetration > deepest_px: deepest_px = penetration
                    if (dir_mult == 1 and raider_x > frame_width*0.75) or (dir_mult == -1 and raider_x < frame_width*0.25):
                        past_baulk_frames += 1

                    current_zone, current_action = z, action_type

            current_time_sec = round(frame_idx / fps, 1)
            raw_timeline.append({
                "time_sec": current_time_sec, 
                "zone": current_zone.replace("_", " ").title(), 
                "action": current_action
            })
            
            if callback and frame_idx % (self.frame_skip * 2) == 0:
                callback({"type": "progress", "players": noses_on_defense_side})
                
                if cnn_scene == "Active Raid" and ui_frames_shown < 15:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    overlay = rgb_frame.copy()
                        
                    if current_raider_kpts is not None:
                        skeleton_edges = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
                        for p1, p2 in skeleton_edges:
                            if p1 < len(current_raider_kpts) and p2 < len(current_raider_kpts):
                                x1, y1 = int(current_raider_kpts[p1][0]), int(current_raider_kpts[p1][1])
                                x2, y2 = int(current_raider_kpts[p2][0]), int(current_raider_kpts[p2][1])
                                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                    cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 0), 4)
                        
                        raider_y_min = rgb_frame.shape[0]
                        for kpt in current_raider_kpts:
                            x, y = int(kpt[0]), int(kpt[1])
                            if x > 0 and y > 0:
                                cv2.circle(overlay, (x, y), 6, (255, 0, 0), -1)
                                if y < raider_y_min: raider_y_min = y

                        if current_raider_x is not None:
                            r_text_x = int(current_raider_x) - 40
                            r_text_y = max(50, raider_y_min - 20)
                            cv2.putText(overlay, "RAIDER", (r_text_x, r_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                            cv2.putText(overlay, "RAIDER", (r_text_x, r_text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

                    cv2.addWeighted(overlay, 0.6, rgb_frame, 0.4, 0, rgb_frame)
                    action_display = f"Tracking Raider: {current_action} in {current_zone.title()} (Time: {current_time_sec}s)"
                    callback({"type": "result", "action": action_display, "evidence_frame": rgb_frame})
                    ui_frames_shown += 1

                callback({"type": "deep_log", "msg": f"Time {current_time_sec}s: Zone={current_zone}, Action={current_action}, Def={noses_on_defense_side}"})

            frame_idx += 1
            
        cap.release()

        final_scene = Counter(cnn_scene_tags).most_common(1)[0][0] if cnn_scene_tags else "Unknown"
        final_cnn_defenders = np.median(cnn_defender_counts) if cnn_defender_counts else 0
        is_raid_likely = True if final_scene == "Active Raid" else False
        
        avg_density = np.mean(formation_densities) if formation_densities else 0
        avg_spread = np.mean(defense_widths) if defense_widths else 0

        timeline_segments = []
        if raw_timeline:
            curr_state = raw_timeline[0]
            start_time = curr_state["time_sec"]
            
            for t in raw_timeline[1:]:
                if t['zone'] != curr_state['zone'] or t['action'] != curr_state['action']:
                    end_time = t["time_sec"]
                    timeline_segments.append({
                        "duration": f"{start_time}s - {end_time}s",
                        "zone": curr_state["zone"],
                        "action": curr_state["action"]
                    })
                    curr_state = t
                    start_time = t["time_sec"]
            
            final_time = raw_timeline[-1]["time_sec"]
            timeline_segments.append({
                "duration": f"{start_time}s - {final_time}s",
                "zone": curr_state["zone"],
                "action": curr_state["action"]
            })

        total_frames = sum(zone_counts.values())
        zone_pct = {k: round((v/total_frames)*100, 1) for k, v in zone_counts.items()} if total_frames > 0 else {}
        default_zones = {"left_corner": 0, "left_in": 0, "center": 100, "right_in": 0, "right_corner": 0}

        return {
            "clip_id": clip_name,
            "max_players_visible": int(max(frame_player_counts)) if frame_player_counts else 0,
            "stats": {
                "defender_pack_size": int(final_cnn_defenders), 
                "formation_density_index": float(round(avg_density, 2)),
                "raider_movement_intensity": float(round(np.mean(confidences), 2)) if confidences else 0,
                "attack_vector": Counter(raider_positions).most_common(1)[0][0] if raider_positions else "Center",
                "defense_spread_px_avg": float(round(avg_spread, 2)),
                "number_of_defenders": int(final_cnn_defenders)
            },
            "classification": {
                "is_raid_likely": is_raid_likely,
                "scene_type": str(final_scene)
            },
            "zonal_analysis": {
                "court_distribution_percentages": zone_pct if sum(zone_pct.values()) > 0 else default_zones,
                "raider_trajectory_timeline": timeline_segments,
                "baulk_line_proximity": {
                    "time_spent_past_baulk_line_sec": round((past_baulk_frames * self.frame_skip) / fps, 1),
                    "deepest_penetration_px": int(deepest_px)
                }
            }
        }

    def process_directory(self, input_dir, output_dir, callback=None):
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
        for file in files:
            if callback: callback({"type": "deep_log", "msg": f"Starting processing for {file}..."})
            data = self.analyze_clip(os.path.join(input_dir, file), output_dir, callback)
            with open(os.path.join(output_dir, file.replace('.mp4', '.json')), 'w') as f:
                json.dump(data, f, indent=4)