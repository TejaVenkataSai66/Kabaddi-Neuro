from google import genai
import os
import json

class ReasoningAgent:
    def __init__(self, api_key):
        # Initialize the client with the new SDK
        self.client = genai.Client(api_key=api_key)

    def construct_prompt(self, clip_data, user_query):
        visuals = clip_data.get("visual_context", {})
        stats = visuals.get("tactical_metrics", {})
        audio = clip_data.get("audio_context", {})
        
        context = f"""
        You are an expert Kabaddi Coach and Analyst. 
        Analyze the following match data for a specific video clip:
        
        --- MATCH CONTEXT ---
        Video ID: {clip_data.get('clip_id')}
        Scene Type: {visuals.get('scene_class')}
        Active Raid: {visuals.get('is_raid')}
        
        --- TACTICAL DATA (Vision) ---
        - Defender Count: {stats.get('defender_pack_size')} (7 is full team, <4 is Super Tackle opp)
        - Attack Flank: {stats.get('attack_vector')}
        - Defense Formation Density: {stats.get('formation_density_index')} (Low=Tight Huddle, High=Spread Out)
        - Defense Spread Width: {stats.get('defense_spread_avg')} pixels
        - Raider Agility Score: {stats.get('raider_movement_intensity')}
        
        --- AUDIO LOGS (Referee & Commentary) ---
        - Transcript: "{audio.get('transcript')}"
        - Referee Calls: {audio.get('referee_events')}
        
        --- USER QUESTION ---
        {user_query}
        
        INSTRUCTIONS:
        1. Answer the user's question based strictly on the data provided.
        2. If the data explains *why* something happened (e.g., "Raider failed because defense was tight"), explicitly mention the metric.
        3. Keep the tone professional, analytical, and sporty.
        """
        return context

    def ask_strategy(self, clip_data, user_query):
        prompt = self.construct_prompt(clip_data, user_query)
        
        # UPDATED MODEL LIST based on your check_models.py output
        models_to_try = [
            'gemini-2.0-flash',        # Primary choice (Fast & Smart)
            'gemini-2.0-flash-lite',   # Backup
            'gemini-flash-latest',     # Generic alias
            'gemini-pro-latest'        # Slow but powerful fallback
        ]
        
        last_error = ""
        
        for model_name in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=model_name, 
                    contents=prompt
                )
                return response.text
            except Exception as e:
                last_error = str(e)
                continue # Try next model
        
        return f"❌ AI Reasoning Error: All models failed. Last error: {last_error}"

if __name__ == "__main__":
    pass