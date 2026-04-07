import networkx as nx
import json
import os
import glob

# Ensure Streamlit does not crash on Windows when plotting
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

            def_count = metrics.get("number_of_defenders", 0)
            attack_vec = metrics.get("attack_vector", "Unknown")
            is_raid = visuals.get("is_raid", False)
            
            ref_calls = audio.get("referee_events", [])
            outcome = "neutral"
            for call in ref_calls:
                c = call.lower()
                if "safe" in c or "bonus" in c: outcome = "raid_success"
                if "out" in c or "super tackle" in c: outcome = "defense_success"

            self.G.add_node(clip_id, type="clip", layer=0, timestamp=clip.get("timestamp_sync"))
            
            state_node = f"State_{def_count}_Defenders"
            self.G.add_node(state_node, type="state", layer=1)
            self.G.add_edge(clip_id, state_node, relation="occurred_in_state")

            scene_node = "Active_Raid" if is_raid else "Defensive_Setup"
            self.G.add_node(scene_node, type="scene", layer=1)
            self.G.add_edge(clip_id, scene_node, relation="has_scene")
            
            # ✨ FIXED: The Attack Vector (Tactic Node) is now mapped universally!
            # Removed the `if is_raid:` gate. Whether it is an Active Raid or Defensive Setup,
            # the raider's flank position is always recorded in the Knowledge Graph.
            tactic_node = f"Attack_{attack_vec}"
            self.G.add_node(tactic_node, type="tactic", layer=2)
            self.G.add_edge(clip_id, tactic_node, relation="used_tactic")
                
            if zonal and "court_distribution_percentages" in zonal:
                dist = zonal["court_distribution_percentages"]
                if dist:
                    top_zone = max(dist, key=dist.get)
                    zone_node = f"Zone_{top_zone}"
                    self.G.add_node(zone_node, type="zone", layer=3)
                    self.G.add_edge(clip_id, zone_node, relation="dominant_zone")

            outcome_node = f"Outcome_{outcome}"
            self.G.add_node(outcome_node, type="outcome", layer=4)
            self.G.add_edge(clip_id, outcome_node, relation="resulted_in")

            if previous_clip_id:
                self.G.add_edge(previous_clip_id, clip_id, relation="followed_by")
            previous_clip_id = clip_id
            
            if callback:
                inter_path = self.visualize_match_topology(f"data/intermediate_graph_step_{i}.png")
                callback({
                    "type": "intermediate_plot", 
                    "msg": f"   > Mapped {clip_id}: {scene_node}, {def_count} Def -> {outcome_node}",
                    "path": inter_path
                })

        if callback: callback({"type": "log", "msg": f"✅ Graph Complete: {self.G.number_of_nodes()} Nodes, {self.G.number_of_edges()} Edges."})

    def get_tactical_insights(self):
        insights = {}
        
        tactic_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("type") == "tactic"]
        if tactic_nodes:
            degrees = self.G.degree(tactic_nodes)
            top_tactic = max(degrees, key=lambda x: x[1])
            insights["dominant_strategy"] = top_tactic[0]
        else:
            insights["dominant_strategy"] = "None"
            
        super_tackle_state = "State_3_Defenders"
        if super_tackle_state in self.G:
            insights["super_tackle_scenarios"] = self.G.degree(super_tackle_state)
        else:
            insights["super_tackle_scenarios"] = 0
            
        zone_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("type") == "zone"]
        if zone_nodes:
            z_degrees = self.G.degree(zone_nodes)
            top_zone = max(z_degrees, key=lambda x: x[1])
            insights["frequent_attack_zone"] = top_zone[0]
        else:
            insights["frequent_attack_zone"] = "None"
            
        return insights

    def get_clip_outcomes(self):
        outcomes_list = []
        for clip in self.clips_data:
            clip_id = clip.get("clip_id")
            audio = clip.get("audio_context", {})
            ref_calls = audio.get("referee_events", [])
            full_audio = " ".join(ref_calls).lower()
            
            outcome = "Neutral / Empty Raid"
            
            if "safe" in full_audio or "bonus" in full_audio:
                if "ceg" in full_audio:
                    outcome = "CEG Raider Success"
                elif "actech" in full_audio:
                    outcome = "ACTECH Raider Success"
            elif "out" in full_audio or "super tackle" in full_audio:
                if "ceg" in full_audio:
                    outcome = "ACTECH Raider Failed"
                elif "actech" in full_audio:
                    outcome = "CEG Raider Failed"

            outcomes_list.append({"Clip": clip_id, "Outcome": outcome})
        return outcomes_list

    def get_advanced_analytics(self):
        zones = {"Left Corner": 0, "Left In": 0, "Center": 0, "Right In": 0, "Right Corner": 0}
        intensities = []
        for clip in self.clips_data:
            zonal = clip.get("visual_context", {}).get("zonal_analysis", {})
            dist = zonal.get("court_distribution_percentages", {})
            if dist:
                top_zone = max(dist, key=dist.get)
                clean_zone = top_zone.replace("_", " ").title()
                if clean_zone in zones:
                    zones[clean_zone] += 1
                else:
                    zones[clean_zone] = 1
            
            metrics = clip.get("visual_context", {}).get("tactical_metrics", {})
            intensities.append(metrics.get("raider_movement_intensity", 0.0))
            
        return {
            "attack_zones": zones,
            "intensities": intensities
        }

    def visualize_match_topology(self, output_path="data/match_graph.png"):
        try:
            plt.figure(figsize=(16, 9)) 
            
            pos = nx.multipartite_layout(self.G, subset_key="layer", align="horizontal")
            
            color_map = []
            for node in self.G:
                ntype = self.G.nodes[node].get("type", "unknown")
                if ntype == "clip": color_map.append('#87CEFA')        
                elif ntype == "tactic": color_map.append('#FFA500')    
                elif ntype == "state": color_map.append('#90EE90')     
                elif ntype == "outcome": color_map.append('#FF6347')   
                elif ntype == "zone": color_map.append('#FFFF00')      
                elif ntype == "scene": color_map.append('#DA70D6')     
                else: color_map.append('#D3D3D3')                      

            nx.draw_networkx_edges(self.G, pos, edge_color='gray', arrows=True, arrowsize=15, alpha=0.4)
            nx.draw_networkx_nodes(self.G, pos, node_color=color_map, node_size=1800, edgecolors='black', linewidths=1.2)
            
            labels = {node: str(node).replace("_", "\n") for node in self.G.nodes()}
            nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, font_weight="bold", font_color="black")
            
            legend_handles = [
                mpatches.Patch(color='#87CEFA', label='Video Clip'),
                mpatches.Patch(color='#DA70D6', label='Scene Type'),
                mpatches.Patch(color='#90EE90', label='Court State (Defenders)'),
                mpatches.Patch(color='#FFA500', label='Raid Tactic (Attack Vector)'),
                mpatches.Patch(color='#FFFF00', label='Dominant Zone'),
                mpatches.Patch(color='#FF6347', label='Match Outcome')
            ]
            plt.legend(handles=legend_handles, loc='upper right', fontsize=10, title="Node Hierarchy", title_fontsize=12, framealpha=0.9)
            
            plt.title("Kabaddi Tactical Knowledge Graph (Hierarchical Layout)", fontsize=18, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            
            plt.savefig(output_path, bbox_inches='tight', dpi=200) 
            plt.close()
            
            return output_path
        except Exception as e:
            print(f"Graph Plot Error: {e}")
            return None