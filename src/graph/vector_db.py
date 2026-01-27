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
        # Suppressed init print for cleaner UI logs
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
        if not os.path.exists(pdf_path):
            if callback: callback({"type": "log", "msg": f"❌ Rulebook not found: {pdf_path}"})
            return

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
                        metadatas=[{
                            "type": "rule", 
                            "source": "official_pdf", 
                            "page": i+1,
                            "chunk_id": j
                        }],
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
        
        # --- 1. NATURAL LANGUAGE TRANSLATION ---
        # Instead of just storing raw numbers, we create sentences that match 
        # how a human would ask questions.
        
        scene_desc = visuals.get('scene_class', 'Unknown Scene')
        is_raid_str = "This is an active raid." if visuals.get('is_raid') else "This is a defensive setup."
        
        # Defender Count Logic (The Fix)
        def_count = metrics.get('defender_pack_size', 0)
        def_str = f"There are {def_count} defenders active on the court."
        if def_count == 7:
            def_str += " The defense is at full strength (7 players)."
        elif def_count < 4:
            def_str += " This is a Super Tackle opportunity (less than 4 defenders)."
            
        # Attack Logic
        attack_vec = metrics.get('attack_vector', 'None')
        attack_str = f"The raider is attacking from the {attack_vec}."
        
        # Audio
        transcript = audio.get('transcript', "")
        ref_events = ", ".join(audio.get('referee_events', []))
        
        # Construct the FINAL Searchable Text
        searchable_text = (
            f"Clip ID: {clip_id}.\n"
            f"{scene_desc}. {is_raid_str}\n"
            f"{def_str}\n"  # <--- This sentence is what your query will match against
            f"{attack_str}\n"
            f"Tactical Data: Formation Density is {metrics.get('formation_density_index')}. "
            f"Raider Intensity is {metrics.get('raider_movement_intensity')}.\n"
            f"Commentary: {transcript}.\n"
            f"Referee Calls: {ref_events}."
        )
        
        # Deep Dive Log
        if callback:
            callback({
                "type": "deep_log", 
                "msg": f"🧠 Encoding [{clip_id}]: \"{def_str} | {attack_str}\""
            })

        self.collection.upsert(
            documents=[searchable_text],
            metadatas=[{
                "type": "gameplay_clip", 
                "filename": clip_id,
                "defenders": def_count, # Stored as int for filtering
                "raid": visuals.get('is_raid')
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
        # Default search behavior
        pass

if __name__ == "__main__":
    vdb = KabaddiVectorDB()
    vdb.reset_database()
    vdb.build_db("data/unified_data", "data/kabaddi_rules.pdf")