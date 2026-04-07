import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# --- 🚨 FIX FOR WINDOWS PYTORCH/STREAMLIT THREAD DEADLOCK 🚨 ---
if hasattr(torch.library, "register_fake"):
    _original_register_fake = torch.library.register_fake
    def _mock_register_fake(*args, **kwargs):
        def decorator(func): return func
        return decorator
    torch.library.register_fake = _mock_register_fake

if hasattr(torch.library, "impl"):
    _original_impl = torch.library.impl
    def _mock_impl(*args, **kwargs):
        def decorator(func): return func
        return decorator
    torch.library.impl = _mock_impl

import torchvision

if hasattr(torch.library, "register_fake"):
    torch.library.register_fake = _original_register_fake
if hasattr(torch.library, "impl"):
    torch.library.impl = _original_impl
# ---------------------------------------------------------------

import streamlit as st
import time
import shutil
import json
import pandas as pd
import plotly.express as px

# --- IMPORT YOUR BACKEND LOGIC ---
from src.utils.video_utils import VideoProcessor
from src.agents.vision_agent import VisionAgent
from src.agents.audio_agent import AudioAgent
from src.utils.sync_manager import SyncManager
from src.graph.vector_db import KabaddiVectorDB
from src.graph.knowledge_graph import KabaddiGraph
from src.agents.reasoning_agent import ReasoningAgent
from dotenv import load_dotenv

# --- LOAD SECRET API KEY ---
load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ API Key not found! Please check your .env file.")

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

st.set_page_config(page_title="Kabaddi Neuro Demo", layout="wide", page_icon="🏆")
st.title("🏆 Kabaddi Neuro: AI Referee & Analyst")
st.markdown("**Tamil Nadu State Kabaddi Board - Automated Analysis System**")

# --- ✨ PERSISTENT UI CACHE ---
if "ui_cache" not in st.session_state:
    st.session_state.ui_cache = {
        "ran": False,
        "w_log": [], "w_freq": "-", "w_conf": "-", "w_df": None,
        "v_img": None, "v_play": "-", "v_act": "Waiting...", "v_log": [],
        "a_trans": [], "a_ref": [],
        "s_log": [],
        "db_log": [],
        "kg_log": [], "kg_img": None, "insights": None,
        "team1_name": "CEG", "team2_name": "ACTECH" # Added to cache
    }

with st.sidebar:
    # --- ✨ NEW: DYNAMIC TEAM CONFIGURATION ---
    st.header("1. Match Configuration")
    team1_input = st.text_input("Team 1 Name", value=st.session_state.ui_cache["team1_name"])
    team2_input = st.text_input("Team 2 Name", value=st.session_state.ui_cache["team2_name"])
    
    st.divider()
    
    st.header("2. Upload Match Footage")
    uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])
    process_btn = st.button("🚀 Start Full Processing", type="primary")
    st.divider()
    
    st.info("System Pipeline:\n1. Whistle Detection\n2. Vision Agent\n3. Audio Agent\n4. Sync Manager\n5. Vector DB\n6. Knowledge Graph\n7. Kabaddi-Neuro LLM")

if uploaded_file is not None:
    video_path = os.path.join(DIRS["raw"], uploaded_file.name)
    with open(video_path, "wb") as f: f.write(uploaded_file.getbuffer())
    st.video(video_path)
    st.success(f"Video uploaded: {uploaded_file.name}")

    if process_btn:
        st.session_state.ui_cache["ran"] = True
        st.session_state.ui_cache["team1_name"] = team1_input.upper()
        st.session_state.ui_cache["team2_name"] = team2_input.upper()
        
        for k in ["w_log", "v_log", "a_trans", "a_ref", "s_log", "db_log", "kg_log"]:
            st.session_state.ui_cache[k] = []

        progress_bar = st.progress(0)
        
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

        def ui_callback(data):
            if data["type"] == "whistle_detected":
                st.session_state.ui_cache["w_freq"] = f"{data['freq']} Hz"
                st.session_state.ui_cache["w_conf"] = f"{data['confidence']:.1f}x"
                freq_metric.metric("Dominant Freq", st.session_state.ui_cache["w_freq"])
                conf_metric.metric("Energy Ratio", st.session_state.ui_cache["w_conf"])
                
                st.session_state.ui_cache["w_log"].append(f"🟢 {data['msg']}")
                if len(st.session_state.ui_cache["w_log"]) > 8: st.session_state.ui_cache["w_log"].pop(0)
                whistle_log.markdown("\n\n".join(st.session_state.ui_cache["w_log"]))
                time.sleep(0.05)
                
            elif data["type"] == "progress":
                cut_status.info(f"Saving Clip {data['clip']}...")

        splitter = VideoProcessor(threshold=3.0)
        if os.path.exists(DIRS["clips"]): shutil.rmtree(DIRS["clips"])
        os.makedirs(DIRS["clips"])
        
        clips_data = splitter.detect_scenes(video_path, DIRS["clips"], callback=ui_callback)
        st.session_state.ui_cache["w_df"] = clips_data
        st.dataframe(clips_data)
        progress_bar.progress(15)

        # --- PHASE 2: VISION ANALYSIS ---
        st.markdown("### 👁️ Phase 2: Computer Vision")
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1: vision_image_box = st.empty() 
        with col_v2: metric_players = st.empty(); metric_action = st.empty()
        
        with st.expander("📝 Vision Logic Logs", expanded=True):
            deep_dive_logs = st.empty()

        def vision_callback(data):
            if data["type"] == "progress":
                st.session_state.ui_cache["v_play"] = f"{data['players']}"
                metric_players.metric("Players", st.session_state.ui_cache["v_play"])
            elif data["type"] == "deep_log":
                st.session_state.ui_cache["v_log"].append(data["msg"])
                if len(st.session_state.ui_cache["v_log"]) > 10: st.session_state.ui_cache["v_log"].pop(0)
                deep_dive_logs.code("\n".join(st.session_state.ui_cache["v_log"]), language="bash")
                time.sleep(0.01)
            elif data["type"] == "result":
                if "evidence_frame" in data:
                    st.session_state.ui_cache["v_img"] = data["evidence_frame"]
                    st.session_state.ui_cache["v_act"] = data['action']
                    vision_image_box.image(data["evidence_frame"], caption=data['action'], use_container_width=True)
                metric_action.info(data['action'])

        vision_bot = VisionAgent(frame_skip=5)
        vision_bot.process_directory(DIRS["clips"], DIRS["metadata"], callback=vision_callback)
        progress_bar.progress(30)

        # --- PHASE 3: AUDIO ---
        st.markdown("### 👂 Phase 3: Audio Intelligence")
        col_a1, col_a2 = st.columns([1, 1])
        with col_a1: 
            st.markdown("**🎙️ Audio Agent Logs**")
            transcript_box = st.empty()
        with col_a2: 
            st.markdown("**🏁 Detected Referee Calls**")
            referee_box = st.empty()
        
        def audio_callback(data):
            if data["type"] == "segment":
                ts = data["timestamp"]; txt = data["text"]
                line = f"🟢 {ts} {txt}" if data.get("is_referee") else f"⚪ {ts} {txt}"
                
                st.session_state.ui_cache["a_trans"].append(line)
                if data.get("is_referee"):
                    st.session_state.ui_cache["a_ref"].append(line)
                    
                if len(st.session_state.ui_cache["a_trans"]) > 8: st.session_state.ui_cache["a_trans"].pop(0)
                transcript_box.markdown("\n\n".join(st.session_state.ui_cache["a_trans"]))
                if st.session_state.ui_cache["a_ref"]: referee_box.markdown("\n\n".join(st.session_state.ui_cache["a_ref"]))
            elif data["type"] == "progress":
                st.session_state.ui_cache["a_trans"].append(f"⚙️ Processing: {data.get('clip')}")
                transcript_box.markdown("\n\n".join(st.session_state.ui_cache["a_trans"]))

        # ✨ NEW: Pass the dynamic UI team names directly into the Audio Agent!
        audio_bot = AudioAgent(
            model_size="small", 
            team1=st.session_state.ui_cache["team1_name"], 
            team2=st.session_state.ui_cache["team2_name"]
        )
        audio_bot.process_directory(DIRS["clips"], DIRS["transcripts"], callback=audio_callback)
        
        for file in os.listdir(DIRS["transcripts"]):
            if file.endswith('.txt'):
                filepath = os.path.join(DIRS["transcripts"], file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                transcript_mode = False
                ref_lines, comm_lines = [], []
                for line in lines:
                    if "── FULL RAW TRANSCRIPT" in line:
                        transcript_mode = True
                        continue
                    if not transcript_mode:
                        clean_line = line.strip()
                        if clean_line.startswith('['): ref_lines.append(clean_line)
                    else:
                        comm_lines.append(line.strip())

                comm_part = " ".join(comm_lines).strip()
                if comm_part == "(empty)": comm_part = ""

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"--- FULL COMMENTARY ---\n{comm_part}\n\n--- REFEREE ANNOUNCEMENTS ---\n")
                    for r_line in ref_lines: f.write(f"{r_line}\n")

        progress_bar.progress(45)

        # --- PHASE 4: SYNC ---
        st.markdown("### 🔄 Phase 4: Data Fusion")
        with st.expander("🔗 Data Fusion Terminal", expanded=True):
            sync_log_box = st.empty()

        def sync_callback(data):
            if data["type"] == "deep_log":
                st.session_state.ui_cache["s_log"].append(data["msg"])
                if len(st.session_state.ui_cache["s_log"]) > 12: st.session_state.ui_cache["s_log"].pop(0)
                sync_log_box.code("\n".join(st.session_state.ui_cache["s_log"]), language="yaml")
                time.sleep(0.02)

        syncer = SyncManager()
        syncer.sync_all(callback=sync_callback)
        progress_bar.progress(60)

        # --- PHASE 5: VECTOR DB ---
        st.markdown("### 🧠 Phase 5: Vector Encoding")
        with st.expander("💾 Vector Encoding Terminal", expanded=True):
            db_log_box = st.empty()
            
        def db_callback(data):
            if data.get("type") in ["deep_log", "log"]:
                st.session_state.ui_cache["db_log"].append(data.get("msg", ""))
                if len(st.session_state.ui_cache["db_log"]) > 10: st.session_state.ui_cache["db_log"].pop(0)
                db_log_box.code("\n".join(st.session_state.ui_cache["db_log"]), language="bash")
                
        vdb = KabaddiVectorDB()
        vdb.reset_database(callback=db_callback)
        rulebook_path = "data/kabaddi_rules.pdf" 
        if not os.path.exists(rulebook_path): rulebook_path = None
        vdb.build_db(DIRS["unified"], rulebook_path, callback=db_callback)
        progress_bar.progress(80)

        # --- PHASE 6: KNOWLEDGE GRAPH ---
        st.markdown("### 🕸️ Phase 6: Tactical Knowledge Graph")
        st.info("Linking Entities: Clips (Blue) -> Scenes (Purple) -> Tactics (Orange) -> States (Green) -> Outcomes (Red)")

        col_g1, col_g2 = st.columns([1, 2])
        with col_g1:
            st.markdown("**🏗️ Graph Construction Logic**")
            kg_log_box = st.empty()
        with col_g2:
            st.markdown("**🗺️ Tactical Topology**")
            graph_plot_box = st.empty()

        def kg_callback(data):
            if data["type"] in ["deep_log", "intermediate_plot"]:
                st.session_state.ui_cache["kg_log"].append(data["msg"])
                if len(st.session_state.ui_cache["kg_log"]) > 15: st.session_state.ui_cache["kg_log"].pop(0)
                kg_log_box.code("\n".join(st.session_state.ui_cache["kg_log"]), language="bash")
                
                if data.get("path") and os.path.exists(data["path"]):
                    st.session_state.ui_cache["kg_img"] = data["path"]
                    graph_plot_box.image(data["path"], caption="Dynamically constructing topology...", use_container_width=True)
                time.sleep(0.5)

        kg = KabaddiGraph()
        kg.load_data(DIRS["unified"], callback=kg_callback)
        kg.build_graph(callback=kg_callback)
        
        plot_path = kg.visualize_match_topology()
        if plot_path and os.path.exists(plot_path):
            st.session_state.ui_cache["kg_img"] = plot_path
            graph_plot_box.image(plot_path, caption="Final Match Topology", use_container_width=True)
            
        insights = kg.get_tactical_insights()
        st.session_state.ui_cache["insights"] = insights
        
        st.markdown("#### 📊 Tactical Insights")
        i_col1, i_col2, i_col3 = st.columns(3)
        i_col1.metric("Dominant Strategy", insights.get("dominant_strategy", "None"))
        i_col2.metric("Super Tackle Opps", f"{insights.get('super_tackle_scenarios', 0)}")
        i_col3.metric("Frequent Attack Zone", insights.get("frequent_attack_zone", "None"))

        st.success("✅ System Pipeline Complete.")
        st.balloons()
        progress_bar.progress(100)

# --- ✨ RENDER STATIC CACHE IF PIPELINE ALREADY RAN ---
elif st.session_state.ui_cache["ran"]:
    st.markdown("### 📢 Phase 1: Audio Event Detection (Whistle)")
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown("**📜 Event Log**\n\n" + "\n\n".join(st.session_state.ui_cache["w_log"]))
    c2.metric("Dominant Freq", st.session_state.ui_cache["w_freq"])
    c3.metric("Energy Ratio", st.session_state.ui_cache["w_conf"])
    if st.session_state.ui_cache["w_df"] is not None: st.dataframe(st.session_state.ui_cache["w_df"])

    st.markdown("### 👁️ Phase 2: Computer Vision")
    cv1, cv2 = st.columns([2, 1])
    if st.session_state.ui_cache["v_img"] is not None:
        cv1.image(st.session_state.ui_cache["v_img"], caption=st.session_state.ui_cache["v_act"], use_container_width=True)
    cv2.metric("Players", st.session_state.ui_cache["v_play"])
    cv2.info(st.session_state.ui_cache["v_act"])
    with st.expander("📝 Vision Logic Logs", expanded=False):
        st.code("\n".join(st.session_state.ui_cache["v_log"]), language="bash")

    st.markdown("### 👂 Phase 3: Audio Intelligence")
    ca1, ca2 = st.columns([1, 1])
    with ca1:
        st.markdown("**🎙️ Audio Agent Logs**")
        st.markdown("\n\n".join(st.session_state.ui_cache["a_trans"]))
    with ca2:
        st.markdown("**🏁 Detected Referee Calls**")
        st.markdown("\n\n".join(st.session_state.ui_cache["a_ref"]))

    st.markdown("### 🔄 Phase 4: Data Fusion")
    with st.expander("🔗 Data Fusion Terminal", expanded=False):
        st.code("\n".join(st.session_state.ui_cache["s_log"]), language="yaml")

    st.markdown("### 🧠 Phase 5: Vector Encoding")
    with st.expander("💾 Vector Encoding Terminal", expanded=False):
        st.code("\n".join(st.session_state.ui_cache["db_log"]), language="bash")

    st.markdown("### 🕸️ Phase 6: Tactical Knowledge Graph")
    cg1, cg2 = st.columns([1, 2])
    with cg1:
        st.markdown("**🏗️ Graph Construction Logic**")
        st.code("\n".join(st.session_state.ui_cache["kg_log"]), language="bash")
    with cg2:
        st.markdown("**🗺️ Tactical Topology**")
        if st.session_state.ui_cache["kg_img"]: st.image(st.session_state.ui_cache["kg_img"], use_container_width=True)

    if st.session_state.ui_cache["insights"]:
        st.markdown("#### 📊 Tactical Insights")
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Dominant Strategy", st.session_state.ui_cache["insights"].get("dominant_strategy", "None"))
        ic2.metric("Super Tackle Opps", f"{st.session_state.ui_cache['insights'].get('super_tackle_scenarios', 0)}")
        ic3.metric("Frequent Attack Zone", st.session_state.ui_cache["insights"].get("frequent_attack_zone", "None"))

# --- SEARCH & REASONING LAYER ---
st.divider()
st.header("🧠 Ask the AI Referee")

query = st.text_input("Ask a question about the match (e.g., 'Why did the raider fail?') or rules (e.g., 'Can players wear ornaments?'):")

if query:
    vdb = KabaddiVectorDB()
    
    st.markdown("### ⚙️ Query Processing Pipeline")
    step1, step2, step3 = st.columns(3)

    category = "match"
    rule_keywords = ["rule", "allowed", "can a player", "ornament", "penalty", "foul", "how many", "points for", "what is"]
    if any(k in query.lower() for k in rule_keywords): 
        category = "rule"

    with step1:
        with st.spinner("Analyzing intent..."):
            time.sleep(1)
        st.success(f"**1. Intent Segregation**\n\nIdentified Category: `{category.upper()}`\n\n*Determined via linguistic intent matching.*")
        
    fetch_count = 2 if category == "match" else 3

    with step2:
        with st.spinner("Fetching Data..."):
            time.sleep(1)
        if category == "match":
            st.success(f"**2. Active Retrieval Engine**\n\nEngine: `Vector Database (ChromaDB)`\n\nStrategy: Semantic Similarity\n\nFetching: Top {fetch_count} Match Clips\n\n*(Note: Knowledge Graph is used for global match topology)*")
        else:
            st.success(f"**2. Active Retrieval Engine**\n\nEngine: `Vector Database (ChromaDB)`\n\nStrategy: Document Text Search\n\nFetching: Top {fetch_count} PDF Guidelines")

    try:
        results = vdb.search(query, category=category, n_results=fetch_count)
        
        if results:
            if category == "match":
                with step3:
                    with st.spinner("Evaluating LLM Candidate..."):
                        time.sleep(1)
                    st.success(f"**3. LLM Candidate Evaluation**\n\nAgent: Gemini Flash\n\nAction: Executing Kabaddi-Neuro reasoning to pick the best match video.")
                
                candidate_data = []
                for res in results:
                    meta = res.get("metadata", {}) if isinstance(res, dict) else {}
                    if isinstance(meta, list): meta = meta[0] if len(meta) > 0 else {}
                    clip_id = meta.get('filename', 'unknown')
                    
                    json_path = os.path.join(DIRS["unified"], str(clip_id) + ".json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            data_obj = json.load(f)
                            if isinstance(data_obj, list): data_obj = data_obj[0] if len(data_obj) > 0 else {}
                            candidate_data.append(data_obj)
                
                if candidate_data and GEMINI_API_KEY and "PASTE" not in GEMINI_API_KEY:
                    
                    kg = KabaddiGraph()
                    match_outcomes = []
                    if os.path.exists(DIRS["unified"]):
                        kg.load_data(DIRS["unified"])
                        match_outcomes = kg.get_clip_outcomes()

                    with st.spinner("LLM generating tactical analysis..."):
                        brain = ReasoningAgent(api_key=GEMINI_API_KEY)
                        llm_response = brain.ask_strategy(candidate_data, query, match_outcomes)
                        
                        try:
                            cleaned_response = llm_response.strip().removeprefix('```json').removesuffix('```').strip()
                            parsed_response = json.loads(cleaned_response)
                            
                            if isinstance(parsed_response, list):
                                parsed_response = parsed_response[0] if len(parsed_response) > 0 else {}
                                
                            best_clip_id = parsed_response.get("selected_clip_id", candidate_data[0].get('clip_id', 'unknown'))
                            advanced_analysis = parsed_response.get("analysis", "Analysis generated.")
                        except:
                            best_clip_id = candidate_data[0].get('clip_id', 'unknown')
                            advanced_analysis = llm_response

                    st.markdown("---")
                    
                    col_res1, col_res2 = st.columns([1, 1])
                    
                    with col_res1:
                        st.success(f"✅ **LLM Selected Best Match: {best_clip_id}**")
                        st.markdown("### 📹 Visual Evidence")
                        
                        best_clip_path = os.path.join(DIRS["clips"], best_clip_id)
                        if not best_clip_path.endswith(".mp4"): best_clip_path += ".mp4"
                            
                        if os.path.exists(best_clip_path):
                            st.video(best_clip_path)
                            
                        st.markdown("### 🤖 Coach's Analysis")
                        st.info(advanced_analysis)
                        
                    with col_res2:
                        # --- 1. ZONAL CHART ---
                        st.markdown("### 🏟️ Raider Zonal Distribution")
                        zonal_data = None
                        
                        best_id_clean = best_clip_id.replace(".mp4", "")
                        for c in candidate_data:
                            if isinstance(c, dict):
                                c_id_clean = str(c.get("clip_id", "")).replace(".mp4", "")
                                if c_id_clean == best_id_clean:
                                    vis_ctx = c.get("visual_context")
                                    if isinstance(vis_ctx, dict):
                                        zon_an = vis_ctx.get("zonal_analysis")
                                        if isinstance(zon_an, dict):
                                            zonal_data = zon_an.get("court_distribution_percentages", {})
                                    break
                        
                        if zonal_data and isinstance(zonal_data, dict):
                            zonal_filtered = {k.replace("_", " ").title(): v for k, v in zonal_data.items() if v > 0}
                            if zonal_filtered:
                                df_zonal = pd.DataFrame(list(zonal_filtered.items()), columns=["Zone", "Percentage"])
                                fig_zonal = px.pie(df_zonal, names="Zone", values="Percentage", hole=0.4)
                                st.plotly_chart(fig_zonal, use_container_width=True, key="zonal_chart")
                            else:
                                st.info("No distinct zonal movement recorded for this clip.")
                        else:
                            st.info("No zonal distribution available. (Data structure missing).")
                            
                        # --- 2. OUTCOMES CHART ---
                        # Dynamically use the names provided by the user in the UI configuration
                        t1_name = st.session_state.ui_cache.get("team1_name", "Team 1").upper()
                        t2_name = st.session_state.ui_cache.get("team2_name", "Team 2").upper()
                        
                        st.markdown("### 📊 Global Match Outcomes (Team Performance)")
                        if match_outcomes:
                            df_outcomes = pd.DataFrame(match_outcomes)
                            df_filtered = df_outcomes[df_outcomes["Outcome"] != "Neutral / Empty Raid"]
                            
                            if not df_filtered.empty:
                                df_agg = df_filtered.groupby("Outcome").size().reset_index(name="Count")
                                
                                # Dynamic color mapping based on whatever teams the user typed!
                                dynamic_colors = {
                                    f"{t1_name} Raider Success": "#00CC96", 
                                    f"{t1_name} Raider Failed": "#EF553B", 
                                    f"{t2_name} Raider Success": "#636EFA", 
                                    f"{t2_name} Raider Failed": "#FFA15A"
                                }
                                
                                fig_dyn = px.bar(
                                    df_agg, 
                                    x="Outcome", 
                                    y="Count", 
                                    color="Outcome",
                                    color_discrete_map=dynamic_colors
                                )
                                fig_dyn.update_layout(yaxis=dict(tickmode='linear', tick0=0, dtick=1))
                                st.plotly_chart(fig_dyn, use_container_width=True, key="dyn_chart")
                            else:
                                st.info("No definitive raid outcomes detected across the match clips yet.")
                        else:
                            st.info("Global match outcomes not available.")
                        
                elif "PASTE" in GEMINI_API_KEY:
                    st.error("⚠️ API Key not detected. Please hardcode your 'GEMINI_API_KEY' in demo_ui.py.")
                    
            elif category == "rule":
                with step3:
                    with st.spinner("Bypassing LLM..."):
                        time.sleep(1)
                    st.success(f"**3. Document Parsing**\n\nExtracted relevant constraints directly from official documentation. (Bypassed Video Retrieval & LLM Generation)")
                
                st.markdown("---")
                st.success("✅ Found relevant guidelines in the Official Rulebook.")
                for idx, res in enumerate(results):
                    st.info(f"**Rule Extract {idx + 1}:** {res['text']}")
                    
        else:
            st.warning("No relevant information found in the database. Please try rephrasing.")
    except Exception as e:
        st.error(f"Search failed: {e}")