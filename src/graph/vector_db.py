import chromadb
from chromadb.utils import embedding_functions
import os
import json
from pypdf import PdfReader
import re
import time

class KabaddiVectorDB:
    def __init__(self, db_path="data/chroma_db"):
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name="kabaddi_knowledge",
            embedding_function=self.embedding_fn
        )

    def reset_database(self, callback=None):
        if callback: callback({"type": "log", "msg": "🧹 Purging existing Knowledge Graph..."})
        try:
            self.client.delete_collection("kabaddi_knowledge")
        except ValueError:
            pass
        
        self.collection = self.client.get_or_create_collection(
            name="kabaddi_knowledge",
            embedding_function=self.embedding_fn
        )
        if callback: callback({"type": "log", "msg": "✅ Database Reset. Ready for ingestion."})

    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def chunk_text_by_sentence(self, text):
        text = text.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+', text)
        valid_sentences = []
        for s in sentences:
            clean_s = s.strip()
            if len(clean_s) > 15: 
                valid_sentences.append(clean_s)
        return valid_sentences

    def ingest_rulebook(self, pdf_path, callback=None):
        if not os.path.exists(pdf_path): return

        if callback: callback({"type": "log", "msg": f"📘 Ingesting Rulebook: {os.path.basename(pdf_path)}..."})
        
        try:
            reader = PdfReader(pdf_path)
            total_sentences = 0
            
            for i, page in enumerate(reader.pages):
                text = self.clean_text(page.extract_text())
                if len(text) < 50: continue
                
                sentences = self.chunk_text_by_sentence(text)
                
                for j, sentence in enumerate(sentences):
                    doc_id = f"rule_page_{i+1}_sent_{j}"
                    
                    if j == 0 and callback:
                        callback({"type": "deep_log", "msg": f"   > Indexing Page {i+1}: '{sentence[:50]}...'"})

                    self.collection.upsert(
                        documents=[sentence],
                        metadatas=[{"type": "rule", "source": "official_pdf", "page": i+1}],
                        ids=[doc_id]
                    )
                    total_sentences += 1
            
            if callback: callback({"type": "log", "msg": f"✅ Rulebook Indexed: {total_sentences} constraints stored."})
            
        except Exception as e:
            if callback: callback({"type": "log", "msg": f"❌ PDF Error: {e}"})

    def add_unified_clip(self, json_path, callback=None):
        if not os.path.exists(json_path): return

        with open(json_path, 'r') as f:
            data = json.load(f)

        clip_id = data.get("clip_id", "unknown")
        visuals = data.get("visual_context", {})
        audio = data.get("audio_context", {})
        metrics = visuals.get("tactical_metrics", {})
        zonal = visuals.get("zonal_analysis", {})
        
        scene_desc = visuals.get('scene_class', 'Unknown Scene')
        is_raid_str = "This is an active raid." if visuals.get('is_raid') else "This is a defensive setup."
        
        # ✨ FIXED: Using strictly 'number_of_defenders' from JSON
        def_count = metrics.get('number_of_defenders', 0)
        def_str = f"There are {def_count} defenders active on the court."
        if def_count == 7:
            def_str += " The defense is at full strength (7 players)."
        elif def_count < 4:
            def_str += " This is a Super Tackle opportunity (less than 4 defenders)."
            
        attack_vec = metrics.get('attack_vector', 'None')
        attack_str = f"The raider is attacking from the {attack_vec}."
        
        transcript = audio.get('transcript', "")
        ref_events = ", ".join(audio.get('referee_events', []))
        full_audio = f"{ref_events} {transcript}".lower()
        
        outcome_contexts = []
        if "raider safe" in full_audio or "safe" in full_audio:
            if "ceg" in full_audio:
                outcome_contexts.append("CEG raider performed successful raid. ACTECH's defense failed. ACTECH's defense failure. CEG's raid success.")
            elif "actech" in full_audio:
                outcome_contexts.append("ACTECH raider performed successful raid. CEG's defense failed. CEG's defense failure. ACTECH's raid success.")
                
        elif "raider out" in full_audio or "out" in full_audio:
            if "ceg" in full_audio:
                outcome_contexts.append("CEG's defense succeeded. ACTECH's raid failed. ACTECH's raider failure. CEG's defense success.")
            elif "actech" in full_audio:
                outcome_contexts.append("ACTECH's defense succeeded. CEG's raid failed. CEG's raider failure. ACTECH's defense success.")
                
        if "bonus" in full_audio:
            if "ceg" in full_audio:
                outcome_contexts.append("CEG scored a bonus point. ACTECH's defense failed. ACTECH's defense failure.")
            elif "actech" in full_audio:
                outcome_contexts.append("ACTECH scored a bonus point. CEG's defense failed. CEG's defense failure.")

        outcome_str = " ".join(outcome_contexts) if outcome_contexts else "Outcome is standard or unclear."

        zonal_str = ""
        if zonal:
            baulk_prox = zonal.get("baulk_line_proximity", {})
            court_dist = zonal.get("court_distribution_percentages", {})
            timeline = zonal.get("raider_trajectory_timeline", [])
            
            zonal_str = f"Raider spent {baulk_prox.get('time_spent_past_baulk_line_sec', 0)} seconds past the baulk line reaching a depth of {baulk_prox.get('deepest_penetration_px', 0)} pixels. "
            if court_dist:
                highest_zone = max(court_dist, key=court_dist.get)
                zonal_str += f"The primary attacking zone was the {highest_zone.replace('_', ' ')}. "
            if timeline:
                actions = ", ".join([f"{t.get('action', 'moving')} in {t.get('zone', 'zone')}" for t in timeline])
                zonal_str += f"Trajectory sequence: {actions}."

        searchable_text = (
            f"Clip ID: {clip_id}.\n"
            f"{scene_desc}. {is_raid_str}\n"
            f"{def_str}\n"
            f"{attack_str}\n"
            f"Explicit Match Outcome: {outcome_str}\n" 
            f"Tactical Data: Formation Density is {metrics.get('formation_density_index')}. "
            f"Raider Intensity is {metrics.get('raider_movement_intensity')}.\n"
            f"Zonal Stats: {zonal_str}\n"
            f"Commentary: {transcript}.\n"
            f"Referee Calls: {ref_events}."
        )
        
        if callback:
            callback({
                "type": "deep_log", 
                "msg": f"🧠 [Combined DB Context] {clip_id}: \"{def_str} | {attack_str} | {outcome_str}\""
            })

        self.collection.upsert(
            documents=[searchable_text],
            metadatas=[{
                "type": "gameplay_clip", 
                "filename": clip_id,
                "defenders": def_count,
                "raid": visuals.get('is_raid'),
                "has_bonus": "bonus" in full_audio
            }],
            ids=[clip_id]
        )
        
        time.sleep(0.05) 

    def build_db(self, unified_data_dir, rulebook_path=None, callback=None):
        if os.path.exists(unified_data_dir):
            files = [f for f in os.listdir(unified_data_dir) if f.endswith('.json')]
            if callback: callback({"type": "log", "msg": f"🧠 Knowledge Graph: Integrating {len(files)} Unified Memories..."})
            
            for file in files:
                self.add_unified_clip(os.path.join(unified_data_dir, file), callback)

        if rulebook_path:
            self.ingest_rulebook(rulebook_path, callback)
            
        if callback: callback({"type": "log", "msg": "✅ Vector Database Optimized & Ready."})

    def search(self, query, category="all", n_results=3):
        try:
            where_clause = None
            if category == "rule":
                where_clause = {"type": "rule"}
            elif category == "match":
                where_clause = {"type": "gameplay_clip"}

            query_params = {
                "query_texts": [query],
                "n_results": n_results
            }
            if where_clause:
                query_params["where"] = where_clause

            results = self.collection.query(**query_params)
            
            structured_results = []
            if results and 'documents' in results and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    meta = results['metadatas'][0][i]
                    structured_results.append({
                        "text": doc,
                        "metadata": meta
                    })
            return structured_results
        except Exception as e:
            print(f"Vector DB Search Error: {e}")
            return []

if __name__ == "__main__":
    vdb = KabaddiVectorDB()
    vdb.reset_database()
    vdb.build_db("data/unified_data", "data/kabaddi_rules.pdf")