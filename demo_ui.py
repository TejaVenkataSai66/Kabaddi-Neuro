import streamlit as st
import os
import time
import shutil

# --- IMPORT YOUR BACKEND LOGIC ---
from src.utils.video_utils import VideoProcessor
from src.agents.vision_agent import VisionAgent
from src.agents.audio_agent import AudioAgent
from src.utils.sync_manager import SyncManager
from src.graph.vector_db import KabaddiVectorDB

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
    st.info("System Pipeline:\n1. Video Segregation (OpenCV)\n2. Player Tracking (YOLOv8)\n3. Audio Analysis (Whisper)\n4. Data Sync\n5. Vector Knowledge Base")

if uploaded_file is not None:
    video_path = os.path.join(DIRS["raw"], uploaded_file.name)
    with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())
    st.video(video_path)
    st.success(f"Video uploaded: {uploaded_file.name}")

    if process_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- PHASE 1: VISUAL SEGREGATION ---
        st.markdown("### 🎬 Phase 1: Histogram Comparison (Cut Detection)")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown("**Frame N (Scene End)**")
            img_A = st.empty()
        with col2:
            st.markdown("**Similarity**")
            metric_score = st.empty()
            metric_time = st.empty()
        with col3:
            st.markdown("**Frame N+1 (New Scene)**")
            img_B = st.empty()
        log_box = st.empty()

        def ui_callback(data):
            if data["type"] == "cut_evidence":
                img_A.image(data["frame_A"], caption="Previous Histogram", use_container_width=True)
                img_B.image(data["frame_B"], caption="Current Histogram", use_container_width=True)
                metric_score.metric("Similarity", f"{data['score']:.3f}", delta="-Low (Cut)", delta_color="inverse")
                metric_time.info(f"Time: {data['time']:.2f}s")
                log_box.markdown(f"🔴 {data['msg']}")
                time.sleep(0.5)
            elif data["type"] == "log":
                log_box.markdown(f"ℹ️ {data['msg']}")

        splitter = VideoProcessor(threshold=0.5)
        if os.path.exists(DIRS["clips"]): shutil.rmtree(DIRS["clips"])
        os.makedirs(DIRS["clips"])
        
        clips_data = splitter.detect_scenes(video_path, DIRS["clips"], callback=ui_callback)
        if clips_data:
            st.success(f"✅ Segregation Complete: {len(clips_data)} clips created.")
            st.dataframe(clips_data)
        progress_bar.progress(20)

        # --- PHASE 2: VISION ANALYSIS ---
        status_text.markdown("### 👁️ Phase 2: Computer Vision (YOLOv8 Pose Estimation)")
        st.info("The AI scans clips. First 2 clips show 'Deep Dive' math logs.")
        
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            st.markdown("**🤖 AI Perception (Live Evidence)**")
            vision_image_box = st.empty() 
        with col_v2:
            st.markdown("**Live Telemetry**")
            metric_players = st.empty()
            metric_action = st.empty()
        
        # DEEP DIVE TERMINAL
        with st.expander("📝 Live Logic Logs (Detailed Math)", expanded=True):
            deep_dive_logs = st.empty()
            log_buffer = []

        def vision_callback(data):
            if data["type"] == "progress":
                metric_players.metric("Players Visible", f"{data['players']}")
            elif data["type"] == "deep_log":
                log_buffer.append(data["msg"])
                if len(log_buffer) > 12: log_buffer.pop(0)
                deep_dive_logs.code("\n".join(log_buffer), language="bash")
                time.sleep(0.02) 
            elif data["type"] == "result":
                # DISPLAY IMAGE FROM MEMORY
                if "evidence_frame" in data:
                    vision_image_box.image(
                        data["evidence_frame"], 
                        caption=f"Analysis: {data['clip']} ({data['action']})", 
                        use_container_width=True
                    )
                metric_players.metric("Confirmed Players", f"{data['players']}", delta="Median Count")
                metric_action.info(f"Verdict: {data['action']}")
                time.sleep(1.0) 

        vision_bot = VisionAgent(frame_skip=5)
        vision_bot.process_directory(DIRS["clips"], DIRS["metadata"], callback=vision_callback)
        st.success("✅ Vision Analysis Complete.")
        progress_bar.progress(40)

        # ==========================================
        # PHASE 3: AUDIO ANALYSIS (Whisper)
        # ==========================================
        status_text.markdown("### 👂 Phase 3: Audio Intelligence (Whisper)")
        st.info("The AI extracts audio, applies noise reduction, and filters for Referee Keywords.")
        
        col_a1, col_a2 = st.columns([1, 1])
        
        with col_a1:
            st.markdown("**📜 Live Transcript Stream**")
            transcript_box = st.empty()
            
        with col_a2:
            st.markdown("**📢 Detected Referee Calls**")
            referee_box = st.empty()
            
        audio_log = st.empty()
        
        # Buffers for UI
        transcript_history = []
        referee_history = []
        
        def audio_callback(data):
            if data["type"] == "log":
                audio_log.caption(f"⚙️ {data['msg']}")
                
            elif data["type"] == "progress":
                audio_log.info(f"🎙️ Listening to: **{data['clip']}**")
                
            elif data["type"] == "segment":
                # Add to scrolling transcript
                ts = data["timestamp"]
                txt = data["text"]
                
                if data["is_referee"]:
                    # Highlight Referee calls
                    line = f"🟢 {ts} **{txt}** (Detected: '{data['keyword']}')"
                    referee_history.append(line)
                    transcript_history.append(f"🟢 {ts} {txt}")
                else:
                    # Normal commentary
                    transcript_history.append(f"⚪ {ts} {txt}")
                
                # Keep buffers manageable
                if len(transcript_history) > 8: transcript_history.pop(0)
                
                # Update UI
                transcript_box.markdown("\n\n".join(transcript_history))
                
                if referee_history:
                    referee_box.markdown("\n\n".join(referee_history))
                else:
                    referee_box.caption("No referee commands detected yet...")

        audio_bot = AudioAgent(model_size="small")
        audio_bot.process_directory(DIRS["clips"], DIRS["transcripts"], callback=audio_callback)
        
        st.success("✅ Audio Analysis Complete.")
        progress_bar.progress(60)

        # ==========================================
        # PHASE 4: SYNC MANAGER (DEEP DIVE UI)
        # ==========================================
        status_text.markdown("### 🔄 Phase 4: Data Fusion & Synchronization")
        st.info("Merging Vision Metrics (JSON) with Audio Events (TXT) into Unified Objects.")
        
        # NEW TERMINAL FOR SYNC
        with st.expander("🔗 Data Fusion Terminal", expanded=True):
            sync_log_box = st.empty()
            sync_buffer = []

        def sync_callback(data):
            if data["type"] == "deep_log":
                sync_buffer.append(data["msg"])
                if len(sync_buffer) > 12: sync_buffer.pop(0)
                sync_log_box.code("\n".join(sync_buffer), language="yaml") # YAML styling looks good for logs
                time.sleep(0.02)
            elif data["type"] == "log":
                sync_buffer.append(f"ℹ️ {data['msg']}")
                sync_log_box.code("\n".join(sync_buffer), language="yaml")

        syncer = SyncManager()
        syncer.sync_all(callback=sync_callback)
        st.success("✅ Synchronization Complete.")
        progress_bar.progress(80)

        # ==========================================
        # PHASE 5: VECTOR DB (DEEP DIVE UI)
        # ==========================================
        status_text.markdown("### 🧠 Phase 5: Knowledge Graph Construction")
        st.info("Generating Vector Embeddings for Search Engine...")

        # NEW TERMINAL FOR DB
        with st.expander("💾 Vector Encoding Terminal", expanded=True):
            db_log_box = st.empty()
            db_buffer = []

        def db_callback(data):
            if data["type"] == "deep_log":
                # Only show last 3 detailed logs to keep it readable but fast
                db_buffer.append(data["msg"])
                if len(db_buffer) > 10: db_buffer.pop(0)
                db_log_box.code("\n".join(db_buffer), language="bash")
            elif data["type"] == "log":
                db_buffer.append(f"ℹ️ {data['msg']}")
                db_log_box.code("\n".join(db_buffer), language="bash")

        vdb = KabaddiVectorDB()
        vdb.reset_database(callback=db_callback)
        
        rulebook_path = "data/kabaddi_rules.pdf" 
        if not os.path.exists(rulebook_path): rulebook_path = None
        
        vdb.build_db(DIRS["unified"], rulebook_path, callback=db_callback)
        
        st.success("✅ System Ready!")
        st.balloons()
        progress_bar.progress(100)

# --- SEARCH ---
st.divider()
st.header("🧠 Ask the AI Referee")
query = st.text_input("Ask a question about the match or rules:")

if query:
    vdb = KabaddiVectorDB()
    category = "all"
    if "rule" in query.lower() or "what is" in query.lower(): category = "rule"
    elif "show" in query.lower() or "clip" in query.lower(): category = "match"
        
    st.markdown(f"**Searching Knowledge Base ({category.upper()})...**")
    
    where_filter = None
    if category == "rule": where_filter = {"type": "rule"}
    elif category == "match": where_filter = {"type": "gameplay_clip"}
    
    results = vdb.collection.query(query_texts=[query], n_results=3, where=where_filter)
    
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            source = meta.get('type', 'Unknown').upper()
            with st.expander(f"Result {i+1} [{source}]"):
                st.markdown(f"**Content:** {doc}")
                if source == "GAMEPLAY_CLIP":
                    clip_name = meta.get('filename') + ".mp4"
                    clip_path = os.path.join(DIRS["clips"], clip_name)
                    if os.path.exists(clip_path): st.video(clip_path)
    else:
        st.error("No relevant information found.")