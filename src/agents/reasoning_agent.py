from google import genai
import os
import json

class ReasoningAgent:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)

    def ask_strategy(self, clips_data_list, user_query, match_outcomes=None):
        """
        The Advanced 'Kabaddi-Neuro' Brain.
        This uses your full multimodal data pipeline and Knowledge Graph.
        """
        context = f"""
        You are an expert Kabaddi Coach and Analyst powered by the 'Kabaddi-Neuro' architecture.
        You have been provided with rich, multimodal tactical data extracted from match video clips.
        
        USER QUESTION: "{user_query}"
        
        CANDIDATE CLIPS:
        """
        
        for clip_data in clips_data_list:
            if isinstance(clip_data, list): clip_data = clip_data[0] if len(clip_data) > 0 else {}
            
            visuals = clip_data.get("visual_context", {}) if isinstance(clip_data, dict) else {}
            if isinstance(visuals, list): visuals = visuals[0] if len(visuals) > 0 else {}
            
            stats = visuals.get("tactical_metrics", {}) if isinstance(visuals, dict) else {}
            if isinstance(stats, list): stats = stats[0] if len(stats) > 0 else {}
            
            audio = clip_data.get("audio_context", {}) if isinstance(clip_data, dict) else {}
            if isinstance(audio, list): audio = audio[0] if len(audio) > 0 else {}
            
            clip_id = clip_data.get('clip_id', 'Unknown') if isinstance(clip_data, dict) else 'Unknown'
            
            context += f"""
            --- CLIP ID: {clip_id} ---
            Scene Type: {visuals.get('scene_class', 'Unknown')}
            Defender Count: {stats.get('number_of_defenders', 'Unknown')}
            Attack Flank: {stats.get('attack_vector', 'Unknown')}
            Formation Density: {stats.get('formation_density_index', 'Unknown')}
            Raider Agility: {stats.get('raider_movement_intensity', 'Unknown')}
            Referee Calls: {audio.get('referee_events', 'Unknown')}
            """
            
        if match_outcomes:
            context += f"\n--- GLOBAL MATCH OUTCOMES (All Clips in the Match) ---\n"
            context += json.dumps(match_outcomes)
            context += "\n"
            
        context += """
        INSTRUCTIONS:
        1. Evaluate the candidate clips and determine which ONE clip best answers the user's query.
        2. Write a professional, highly-specific tactical explanation answering the user's question. You MUST explicitly reference the 'Formation Density', 'Defender Count', or 'Attack Flank' data provided to you to prove you are using the advanced metrics.
        3. You MUST return your answer as a valid JSON object.
        
        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
        {
            "selected_clip_id": "clip_X.mp4",
            "analysis": "Your detailed tactical explanation goes here..."
        }
        """
        
        try:
            # Primary model for Kabaddi-Neuro analysis
            response = self.client.models.generate_content(
                model='gemini-flash-lite-latest', 
                contents=context
            )
            return response.text
        except Exception as e:
            print(f"Primary LLM Error: {e}")
            return e 