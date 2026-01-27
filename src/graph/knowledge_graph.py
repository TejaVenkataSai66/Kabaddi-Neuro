import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import re

class KabaddiGraph:
    def __init__(self, config_path="match_config.json"):
        self.graph = nx.DiGraph()
        self.config = self.load_config(config_path)
        
        # 1. SETUP MATCH & TEAMS
        match_id = self.config.get("match_id", "Final_Match")
        self.root_node = match_id
        self.graph.add_node(match_id, type="Match", label="PKL Final 2025", layer=0)
        
        # Load Teams
        teams = self.config.get("teams", {})
        self.team_a = teams.get("team_A", {}).get("name", "Team A")
        self.team_b = teams.get("team_B", {}).get("name", "Team B")
        
        # Create Team Nodes (Layer 1)
        self.graph.add_node(self.team_a, type="Team", color="orange", layer=1)
        self.graph.add_node(self.team_b, type="Team", color="blue", layer=1)
        
        self.graph.add_edge(match_id, self.team_a, relation="FEATURED_TEAM")
        self.graph.add_edge(match_id, self.team_b, relation="FEATURED_TEAM")

    def load_config(self, path):
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return {}

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def add_clip_logic(self, clip_id, vision_data, transcript, previous_clip_id):
        # --- DATA EXTRACTION ---
        player_count = vision_data.get('max_players_visible', 0)
        is_raid_vision = vision_data.get('is_raid_likely', False)
        text_lower = transcript.lower()
        
        # Node: Video Clip (Layer 2)
        self.graph.add_node(
            clip_id, 
            type="VideoClip", 
            transcript=transcript[:40]+"...",
            layer=2
        )
        
        # CHRONOLOGY LINK
        if previous_clip_id:
            self.graph.add_edge(previous_clip_id, clip_id, relation="NEXT_MOMENT")
        else:
            self.graph.add_edge(self.root_node, clip_id, relation="KICKOFF")

        # --- LOGIC LAYER 1: DEFENSIVE CONTEXT ---
        # Connect clip to the number of defenders (Tactical State)
        # 6-7 Players = Full House (Safe)
        # 1-3 Players = Super Tackle Opportunity (High Risk/High Reward)
        situation_node = f"Sit_{player_count}_Players"
        situation_label = f"{player_count} Defenders"
        
        if player_count <= 3 and player_count > 0:
            situation_node += "_SuperTackle_On"
            situation_label += " (Super Tackle ON)"
            self.graph.add_node(situation_node, type="Context", color="red", layer=3)
        else:
            self.graph.add_node(situation_node, type="Context", color="gray", layer=3)
            
        self.graph.add_edge(clip_id, situation_node, relation="DEFENSE_STATE")

        # --- LOGIC LAYER 2: EVENT CLASSIFICATION ---
        event_node = None
        event_label = "Unknown"
        
        # Keywords
        has_point = any(x in text_lower for x in ["point", "touch", "hand", "toe"])
        has_tackle = any(x in text_lower for x in ["tackle", "caught", "dash", "block"])
        has_bonus = "bonus" in text_lower
        has_review = "review" in text_lower or "challenge" in text_lower
        
        if has_review:
            event_label = "DRS Review"
            event_node = f"Evt_{clip_id}_Review"
        elif "super tackle" in text_lower:
            event_label = "SUPER TACKLE (+2)"
            event_node = f"Evt_{clip_id}_SuperTackle"
        elif "all out" in text_lower:
            event_label = "ALL OUT (+2)"
            event_node = f"Evt_{clip_id}_AllOut"
        elif has_tackle:
            event_label = "Successful Tackle"
            event_node = f"Evt_{clip_id}_Tackle"
        elif is_raid_vision:
            if has_point:
                event_label = "Scoring Raid"
                event_node = f"Evt_{clip_id}_RaidScore"
            elif has_bonus:
                event_label = "Bonus Point"
                event_node = f"Evt_{clip_id}_Bonus"
            else:
                event_label = "Empty Raid"
                event_node = f"Evt_{clip_id}_EmptyRaid"
                
        # --- LOGIC LAYER 3: TEAM ATTRIBUTION ---
        # Check who is mentioned in the commentary
        involved_team = None
        if self.team_a.lower() in text_lower:
            involved_team = self.team_a
        elif self.team_b.lower() in text_lower:
            involved_team = self.team_b
            
        # --- GRAPH CONSTRUCTION ---
        if event_node:
            self.graph.add_node(event_node, type="Event", label=event_label, layer=4)
            self.graph.add_edge(clip_id, event_node, relation="HAS_EVENT")
            
            # Link Event to Concept (Hub)
            concept = f"Concept_{event_label.split()[0]}"
            self.graph.add_node(concept, type="Concept", layer=5)
            self.graph.add_edge(event_node, concept, relation="INSTANCE_OF")
            
            # Link Event to Team (if known)
            if involved_team:
                self.graph.add_edge(involved_team, event_node, relation="EXECUTED_BY")

    def build_from_directories(self, json_dir, txt_dir):
        files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        files.sort(key=self.natural_sort_key)
        
        print(f"Constructing Deep Tactical Graph from {len(files)} clips...")
        prev_clip = None
        
        for file in files:
            json_path = os.path.join(json_dir, file)
            txt_path = os.path.join(txt_dir, file.replace('.json', '.txt'))
            
            with open(json_path, 'r') as f: vision = json.load(f)
            
            transcript = ""
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f: transcript = f.read().strip()
                
            clip_id = vision['video_file']
            self.add_clip_logic(clip_id, vision, transcript, prev_clip)
            prev_clip = clip_id
            
        print(f"Deep Graph Complete! Nodes: {self.graph.number_of_nodes()} | Edges: {self.graph.number_of_edges()}")

    def visualize(self, output_file="kabaddi_graph_deep.png"):
        plt.figure(figsize=(20, 14))
        
        # Define Layers for Layout (Hierarchical)
        pos = nx.spring_layout(self.graph, k=2, iterations=100)
        
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node, attr in self.graph.nodes(data=True):
            n_type = attr.get("type", "Unknown")
            label = attr.get("label", node)
            
            # Custom Styling
            if n_type == "Match": 
                node_colors.append("#FFD700") # Gold
                node_sizes.append(3000)
                labels[node] = label
            elif n_type == "Team": 
                node_colors.append("#FFA500") # Orange
                node_sizes.append(2500)
                labels[node] = label
            elif n_type == "VideoClip": 
                node_colors.append("#87CEEB") # Sky Blue
                node_sizes.append(1000)
                labels[node] = node[:7] # Short label
            elif n_type == "Context": 
                node_colors.append("#FF6961") # Red/Gray
                node_sizes.append(1200)
                labels[node] = label
            elif n_type == "Event": 
                node_colors.append("#90EE90") # Light Green
                node_sizes.append(1500)
                labels[node] = label
            else: 
                node_colors.append("#D3D3D3")
                node_sizes.append(500)
                labels[node] = ""

        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, edgecolors='white')
        nx.draw_networkx_edges(self.graph, pos, edge_color='#888888', arrows=True, width=1.5, alpha=0.6)
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=9, font_weight="bold")
        
        plt.title("Kabaddi-Neuro: Deep Tactical Knowledge Graph", fontsize=16)
        plt.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
        plt.close()

if __name__ == "__main__":
    kg = KabaddiGraph()
    kg.build_from_directories("data/metadata", "data/transcripts")
    kg.visualize()