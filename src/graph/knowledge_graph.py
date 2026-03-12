import networkx as nx
import json
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter
import time

class KabaddiGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.clips_data = []

    def load_data(self, unified_dir, callback=None):
        if callback: callback({"type": "log", "msg": "📊 Loading Unified Data for Topology..."})
        files = sorted(glob.glob(os.path.join(unified_dir, "*.json")))
        self.clips_data = []
        for f in files:
            with open(f, 'r') as file:
                self.clips_data.append(json.load(file))
        if callback: callback({"type": "log", "msg": f"✅ Graph Loader: Loaded {len(self.clips_data)} clips."})

    def build_graph(self, callback=None):
        self.G.clear()
        previous_clip_id = None
        
        if callback: callback({"type": "log", "msg": "🕸️ Constructing Tactical Network (Deep Analysis)..."})

        for i, clip in enumerate(self.clips_data):
            clip_id = clip.get("clip_id")
            visuals = clip.get("visual_context", {})
            metrics = visuals.get("tactical_metrics", {})
            audio = clip.get("audio_context", {})
            zonal = visuals.get("zonal_analysis", {})
            
            # --- 1. BLUE NODE (Clip) ---
            self.G.add_node(clip_id, type="clip", time_index=i)
            if callback:
                callback({
                    "type": "deep_log", 
                    "msg": f"🔵 [CLIP NODE] Created: {clip_id} (Time Index: {i})"
                })

            # --- 2. TEMPORAL LINK ---
            if previous_clip_id:
                self.G.add_edge(previous_clip_id, clip_id, relation="NEXT_PLAY")
                if callback:
                     callback({
                        "type": "deep_log", 
                        "msg": f"   ↳ ⛓️ [LINK] Temporal Flow: {previous_clip_id} --> {clip_id}"
                    })
            previous_clip_id = clip_id
            
            # --- 3. ORANGE NODE (Tactic) ---
            vector = metrics.get("attack_vector", "None")
            if vector and vector != "None":
                node_name = f"Tactic: {vector}"
                self.G.add_node(node_name, type="tactic")
                self.G.add_edge(clip_id, node_name, relation="USES_STRATEGY")
                if callback: 
                    callback({
                        "type": "deep_log", 
                        "msg": f"   ↳ 🟠 [TACTIC NODE] Identified Strategy: '{vector}' -> Linked to {clip_id}"
                    })

            # --- 4. GREEN NODE (State) ---
            def_count = metrics.get("defender_pack_size", 0)
            if def_count > 0:
                if def_count <= 3: 
                    state_name = "State: Super Tackle Opp"
                    reason = "Defenders <= 3 (High Stakes)"
                elif def_count == 7: 
                    state_name = "State: Full Defense"
                    reason = "Defenders == 7 (Max Strength)"
                else: 
                    state_name = f"State: {def_count} Defenders"
                    reason = f"Standard Defense ({def_count})"
                
                self.G.add_node(state_name, type="state")
                self.G.add_edge(clip_id, state_name, relation="GAME_STATE")
                if callback: 
                    callback({
                        "type": "deep_log", 
                        "msg": f"   ↳ 🟢 [STATE NODE] Context: '{state_name}' (Reason: {reason})"
                    })

            # --- 5. RED NODE (Event) ---
            ref_calls = audio.get("referee_events", [])
            for call in ref_calls:
                call_clean = call.split(']')[-1].strip()
                node_name = f"Ref: {call_clean}"
                self.G.add_node(node_name, type="event")
                self.G.add_edge(clip_id, node_name, relation="OCCURRED")
                if callback: 
                    callback({
                        "type": "deep_log", 
                        "msg": f"   ↳ 🔴 [EVENT NODE] Outcome Detected: '{call_clean}' -> Linking Result"
                    })
                    
            # --- 6. YELLOW NODE (Zonal) ---
            if zonal:
                court_dist = zonal.get("court_distribution_percentages", {})
                if court_dist:
                    highest_zone = max(court_dist, key=court_dist.get)
                    zonal_node = f"Zone: {highest_zone.replace('_', ' ').title()}"
                    self.G.add_node(zonal_node, type="zone")
                    self.G.add_edge(clip_id, zonal_node, relation="PRIMARY_ATTACK_ZONE")
                    if callback: 
                        callback({
                            "type": "deep_log", 
                            "msg": f"   ↳ 🟡 [ZONE NODE] Attack Zone: '{zonal_node}' -> Linked to {clip_id}"
                        })
            
            # Pacing for UI
            time.sleep(0.1)

    def get_tactical_insights(self):
        insights = {}
        tactic_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("type") == "tactic"]
        if tactic_nodes:
            degrees = self.G.degree(tactic_nodes)
            top_tactic = max(degrees, key=lambda x: x[1])
            insights["dominant_strategy"] = top_tactic[0]
        else:
            insights["dominant_strategy"] = "None"

        super_tackle_nodes = [n for n in self.G.predecessors("State: Super Tackle Opp")] if "State: Super Tackle Opp" in self.G else []
        insights["super_tackle_scenarios"] = len(super_tackle_nodes)
        
        zone_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("type") == "zone"]
        if zone_nodes:
            z_degrees = self.G.degree(zone_nodes)
            top_zone = max(z_degrees, key=lambda x: x[1])
            insights["frequent_attack_zone"] = top_zone[0]
        else:
            insights["frequent_attack_zone"] = "None"
            
        return insights

    def visualize_match_topology(self, output_path="data/match_graph.png"):
        try:
            plt.figure(figsize=(10, 6))
            pos = nx.spring_layout(self.G, k=0.5, iterations=50)
            color_map = []
            for node in self.G:
                ntype = self.G.nodes[node].get("type", "unknown")
                if ntype == "clip": color_map.append('#87CEFA')   
                elif ntype == "tactic": color_map.append('#FFA500') 
                elif ntype == "state": color_map.append('#90EE90')  
                elif ntype == "event": color_map.append('#FF6347')  
                elif ntype == "zone": color_map.append('#FFFF00')  
                else: color_map.append('grey')
            nx.draw(self.G, pos, node_color=color_map, with_labels=True, font_size=8, node_size=800, alpha=0.9, edge_color='gray')
            plt.title("Kabaddi Match Tactical Topology")
            plt.savefig(output_path)
            plt.close()
            return output_path
        except Exception as e:
            return None

if __name__ == "__main__":
    pass