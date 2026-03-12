import streamlit as st
import os
import time
import shutil
import json

# --- IMPORT YOUR BACKEND LOGIC ---
from src.utils.video_utils import VideoProcessor
from src.agents.vision_agent import VisionAgent
from src.agents.audio_agent import AudioAgent
from src.utils.sync_manager import SyncManager
from src.graph.vector_db import KabaddiVectorDB
from src.graph.knowledge_graph import KabaddiGraph
from src.agents.reasoning_agent import ReasoningAgent

# ==========================================
# 🔑 SYSTEM CREDENTIALS (HIDDEN FROM USER)
# ==========================================
# PASTE YOUR KEY INSIDE THE QUOTES BELOW
GEMINI_API_KEY = "AIzaSyDrcndeFrz2Wh6_eR-z9NG2HBJvjpkVZfk" 

# --- CONFIGURATION ---
BASE_DIR = "data"
DIRS = {
    "raw": os.path.join(BASE_DIR, "raw_videos"),
    "clips": os.path.join(BASE_DIR, "processed_clips"),
    "metadata": os.path.join(BASE_DIR, "metadata"),
    "transcripts": os.path.join(BASE_DIR, "transcripts"),
    "unified": os.path.join(BASE_DIR, "unified_data"),
    "chroma": os.path.join(BASE_DIR, "chroma_db")
}

for d in DIRS.values():
    if not os.path.exists(d): os.makedirs(d)

st.set_page_config(page_title="Kabaddi Neuro Demo", layout="wide")
st.title("🏆 Kabaddi Neuro: AI Referee & Analyst")
st.markdown("**Tamil Nadu State Kabaddi Board - Automated Analysis System**")

with st.sidebar:
    st.header("1. Upload Match Footage")
    uploaded_file = st.file_uploader("Choose an MP4 file", type="mp4")
    process_btn = st.button("🚀 Start Full Processing", type="primary")
    st.divider()
    
    st.info("System Pipeline:\n1. Whistle Detection\n2. Vision Agent\n3. Audio Agent\n4. Sync Manager\n5. Vector DB\n6. Knowledge Graph\n7. LLM Reasoning")

if uploaded_file is not None:
    video_path = os.path.join(DIRS["raw"], uploaded_file.name)
    with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())
    st.video(video_path)
    st.success(f"Video uploaded: {uploaded_file.name}")

    if process_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- PHASE 1: AUDIO SEGREGATION (WHISTLE) ---
        st.markdown("### 📢 Phase 1: Audio Event Detection (Whistle)")
        st.info("Detecting Referee Whistles (2kHz-5kHz). Live Frequency Analysis:")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown("**📜 Event Log**")
            whistle_log = st.empty()
        with col2:
            st.markdown("**🔊 Live Frequency**")
            freq_metric = st.empty()
        with col3:
            st.markdown("**📊 Confidence**")
            conf_metric = st.empty()
            
        cut_status = st.empty()
        whistle_buffer = []

        def ui_callback(data):
            if data["type"] == "whistle_detected":
                # UPDATE LIVE NUMBERS
                freq_metric.metric("Dominant Freq", f"{data['freq']} Hz")
                conf_metric.metric("Energy Ratio", f"{data['confidence']:.1f}x")
                
                whistle_buffer.append(f"🟢 {data['msg']}")
                if len(whistle_buffer) > 8: whistle_buffer.pop(0)
                whistle_log.markdown("\n\n".join(whistle_buffer))
                time.sleep(0.05)
                
            elif data["type"] == "progress":
                cut_status.info(f"Saving Clip {data['clip']}...")

        splitter = VideoProcessor(threshold=3.0)
        if os.path.exists(DIRS["clips"]): shutil.rmtree(DIRS["clips"])
        os.makedirs(DIRS["clips"])
        
        clips_data = splitter.detect_scenes(video_path, DIRS["clips"], callback=ui_callback)
        st.dataframe(clips_data)
        progress_bar.progress(15)

        # --- PHASE 2: VISION ANALYSIS ---
        status_text.markdown("### 👁️ Phase 2: Computer Vision")
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1: vision_image_box = st.empty() 
        with col_v2: metric_players = st.empty(); metric_action = st.empty()
        
        with st.expander("📝 Vision Logic Logs", expanded=True):
            deep_dive_logs = st.empty()
            log_buffer = []

        def vision_callback(data):
            if data["type"] == "progress":
                metric_players.metric("Players", f"{data['players']}")
            elif data["type"] == "deep_log":
                log_buffer.append(data["msg"])
                if len(log_buffer) > 10: log_buffer.pop(0)
                deep_dive_logs.code("\n".join(log_buffer), language="bash")
                time.sleep(0.01)
            elif data["type"] == "result":
                if "evidence_frame" in data:
                    vision_image_box.image(data["evidence_frame"], caption=data['action'], use_container_width=True)
                metric_action.info(f"{data['action']}")

        vision_bot = VisionAgent(frame_skip=5)
        vision_bot.process_directory(DIRS["clips"], DIRS["metadata"], callback=vision_callback)
        progress_bar.progress(30)

        # --- PHASE 3: AUDIO ---
        status_text.markdown("### 👂 Phase 3: Audio Intelligence")
        col_a1, col_a2 = st.columns([1, 1])
        with col_a1: transcript_box = st.empty()
        with col_a2: referee_box = st.empty()
        
        tr_hist = []; ref_hist = []
        def audio_callback(data):
            if data["type"] == "segment":
                ts = data["timestamp"]; txt = data["text"]
                if data["is_referee"]:
                    ref_hist.append(f"🟢 {ts} {txt}")
                    tr_hist.append(f"🟢 {ts} {txt}")
                else:
                    tr_hist.append(f"⚪ {ts} {txt}")
                if len(tr_hist) > 8: tr_hist.pop(0)
                transcript_box.markdown("\n\n".join(tr_hist))
                if ref_hist: referee_box.markdown("\n\n".join(ref_hist))

        audio_bot = AudioAgent(model_size="small")
        audio_bot.process_directory(DIRS["clips"], DIRS["transcripts"], callback=audio_callback)
        progress_bar.progress(45)

        # --- PHASE 4: SYNC ---
        status_text.markdown("### 🔄 Phase 4: Data Fusion")
        with st.expander("🔗 Data Fusion Terminal", expanded=True):
            sync_log_box = st.empty()
            sync_buffer = []

        def sync_callback(data):
            if data["type"] == "deep_log":
                sync_buffer.append(data["msg"])
                if len(sync_buffer) > 12: sync_buffer.pop(0)
                sync_log_box.code("\n".join(sync_buffer), language="yaml")
                time.sleep(0.02)

        syncer = SyncManager()
        syncer.sync_all(callback=sync_callback)
        progress_bar.progress(60)

        # --- PHASE 5: VECTOR DB ---
        status_text.markdown("### 🧠 Phase 5: Vector Encoding")
        with st.expander("💾 Vector Encoding Terminal", expanded=True):
            db_log_box = st.empty()
            db_buffer = []
        def db_callback(data):
            if data["type"] == "deep_log":
                db_buffer.append(data["msg"])
                if len(db_buffer) > 10: db_buffer.pop(0)
                db_log_box.code("\n".join(db_buffer), language="bash")
        vdb = KabaddiVectorDB()
        vdb.reset_database(callback=db_callback)
        rulebook_path = "data/kabaddi_rules.pdf" 
        if not os.path.exists(rulebook_path): rulebook_path = None
        vdb.build_db(DIRS["unified"], rulebook_path, callback=db_callback)
        progress_bar.progress(80)

        # --- PHASE 6: KNOWLEDGE GRAPH ---
        status_text.markdown("### 🕸️ Phase 6: Tactical Knowledge Graph")
        st.info("Linking Entities: Clips (Blue) -> Tactics (Orange) -> States (Green) -> Outcomes (Red)")

        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            st.markdown("**🏗️ Graph Construction Logic**")
            kg_log_box = st.empty()
            kg_buffer = []
        with col_g2:
            st.markdown("**🗺️ Tactical Topology**")
            graph_plot_box = st.empty()

        def kg_callback(data):
            if data["type"] == "deep_log":
                kg_buffer.append(data["msg"])
                if len(kg_buffer) > 15: kg_buffer.pop(0)
                kg_log_box.code("\n".join(kg_buffer), language="bash")

        kg = KabaddiGraph()
        kg.load_data(DIRS["unified"], callback=kg_callback)
        kg.build_graph(callback=kg_callback)
        
        plot_path = kg.visualize_match_topology()
        if plot_path and os.path.exists(plot_path):
            graph_plot_box.image(plot_path, caption="Match Topology", use_container_width=True)
            
        insights = kg.get_tactical_insights()
        st.markdown("#### 📊 Tactical Insights")
        
        # Changed to 3 columns to display the new Zonal Intelligence
        i_col1, i_col2, i_col3 = st.columns(3)
        i_col1.metric("Dominant Strategy", insights.get("dominant_strategy", "None"))
        i_col2.metric("Super Tackle Opps", f"{insights.get('super_tackle_scenarios', 0)}")
        i_col3.metric("Frequent Attack Zone", insights.get("frequent_attack_zone", "None"))

        st.success("✅ System Pipeline Complete.")
        st.balloons()
        progress_bar.progress(100)

# --- SEARCH & REASONING LAYER ---
st.divider()
st.header("🧠 Ask the AI Referee")

# 1. Search Interface
query = st.text_input("Ask a question about the match (e.g., 'Why did the CEG raider fail?') or rules (e.g., 'Can players wear ornaments?'):")

if query:
    vdb = KabaddiVectorDB()
    
    # Smarter category detection
    category = "match"
    rule_keywords = ["rule", "allowed", "can a player", "ornament", "penalty", "foul", "how many", "points for", "what is"]
    if any(k in query.lower() for k in rule_keywords): 
        category = "rule"
        
    st.markdown(f"**Searching Knowledge Base ({category.upper()})...**")
    
    where_filter = None
    fetch_count = 1  # Default to 1 for video clips
    
    if category == "rule": 
        where_filter = {"type": "rule"}
        fetch_count = 3  # Ask ChromaDB for 3 sentences for rules
    elif category == "match": 
        where_filter = {"type": "gameplay_clip"}
    
    try:
        # Dynamically pass where filter to prevent ChromaDB crashing
        query_params = {"query_texts": [query], "n_results": fetch_count}
        if where_filter:
            query_params["where"] = where_filter
            
        results = vdb.collection.query(**query_params)
        
        if results and 'documents' in results and len(results['documents'][0]) > 0:
            
            # Scenario A: It's a Gameplay Clip
            if category == "match" and results['metadatas'][0][0].get('type') == 'gameplay_clip':
                doc = results['documents'][0][0]
                meta = results['metadatas'][0][0]
                clip_id = meta.get('filename')
                
                st.success(f"✅ Found relevant match clip: {clip_id}")
                
                clip_path = os.path.join(DIRS["clips"], str(clip_id) + ".mp4")
                if os.path.exists(clip_path):
                    st.video(clip_path)
                    
                json_path = os.path.join(DIRS["unified"], str(clip_id) + ".json")
                if os.path.exists(json_path) and GEMINI_API_KEY and "PASTE" not in GEMINI_API_KEY:
                    with open(json_path, 'r') as f:
                        clip_data = json.load(f)
                    
                    st.markdown("### 🤖 Coach's Analysis")
                    with st.spinner("Analyzing tactics with Gemini..."):
                        brain = ReasoningAgent(api_key=GEMINI_API_KEY)
                        strategy_explanation = brain.ask_strategy(clip_data, query)
                    st.info(strategy_explanation)
                elif "PASTE" in GEMINI_API_KEY:
                    st.error("⚠️ API Key not detected. Please hardcode your 'GEMINI_API_KEY' in demo_ui.py.")
                    
            # Scenario B: It's a Rulebook Extract
            elif category == "rule":
                st.success("✅ Found relevant guidelines in the Official Rulebook.")
                
                # Loop through and display the 3 retrieved sentences
                for idx, doc in enumerate(results['documents'][0]):
                    st.info(f"**Rule Extract {idx + 1}:** {doc}")
                    
        else:
            st.warning("No relevant information found in the database. Please try rephrasing.")
    except Exception as e:
        st.error(f"Search failed: {e}")