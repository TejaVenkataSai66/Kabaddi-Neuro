# 🏆 Kabaddi-Neuro: Agentic Multimodal Hybrid-RAG System for Tactical Sports Analytics

## 📌 Overview
**Kabaddi-Neuro** is an advanced, automated tactical analytics platform designed for the sport of Kabaddi. Moving beyond traditional manual video review, this system utilizes an autonomous, multi-agent AI architecture to extract granular physical metrics, synchronize audio-visual events, and generate professional-grade coaching insights using Large Language Models (LLMs).

By fusing **Computer Vision, Audio Intelligence, and a Hybrid-RAG (Retrieval-Augmented Generation) engine**, the system acts as an interactive "AI Referee & Coach," allowing users to query match footage using natural language and receive mathematically backed, visual tactical breakdowns.

---

## ✨ Key Features
- **👁️ Vision Agent (Spatial Intelligence):** Utilizes YOLOv8-Pose and fine-tuned ResNet-18 models to track skeletal joints, calculate defensive formation density, track raider attack vectors, and determine zonal court distribution.
- **👂 Audio Agent (Temporal Intelligence):** Leverages Faster-Whisper and intelligent phonetic replacement pipelines to isolate referee whistles, filter stadium noise, and perfectly transcribe match outcomes and points.
- **🔗 Hybrid-RAG Architecture:** Combines semantic search via a Vector Database (**ChromaDB**) with a deterministic, hierarchical **Knowledge Graph** to understand both specific clips and global match topology.
- **🧠 Agentic Reasoning:** Employs an LLM (Google Gemini) instructed via "Agent-of-Thoughts" to cross-reference visual JSON data with global match stats, eliminating AI hallucinations and providing accurate, data-driven answers.
- **📊 Dynamic Visualizations:** Natively renders interactive Plotly charts like Zonal Distribution, Global Match Outcomes.

---

## 🏗️ System Architecture
The data pipeline processes raw broadcast footage through 7 distinct phases:
1. **Audio Event Detection:** Isolates referee whistles (2kHz-5kHz) to segment continuous footage into logical engagement clips.
2. **Computer Vision:** Extracts player coordinates, calculates distances, and classifies scenes (Active Raid vs. Defensive Setup).
3. **Audio Intelligence:** Transcribes referee calls and maps them to specific teams.
4. **Data Fusion (Sync Manager):** Merges visual and audio timelines into a unified JSON metadata object.
5. **Vector Encoding:** Stores semantic embeddings of the tactical data into ChromaDB.
6. **Knowledge Graph:** Maps spatial, temporal, and logical relationships (Clips → States → Tactics → Zones → Outcomes).
7. **LLM Reasoning Engine:** Retrieves candidate clips, evaluates them against the Knowledge Graph, and dynamically generates textual and visual analyses.

---

## 🛠️ Technology Stack
* **Language:** Python
* **Computer Vision:** PyTorch, Ultralytics (YOLOv8-Pose), torchvision (ResNet-18), OpenCV
* **Audio Processing:** WhisperX / Faster-Whisper, FFmpeg, Librosa, Soundfile
* **Databases & Data Structures:** ChromaDB (Vector DB), NetworkX (Knowledge Graph), Pandas
* **AI / LLM:** Google GenAI SDK (Gemini 2.0 Flash / 1.5 Flash)
* **Frontend UI:** Streamlit, Plotly Express

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- [FFmpeg](https://ffmpeg.org/download.html) installed and added to system PATH.
- A valid Google Gemini API Key.

## **Constraints**
1. The referee's whistle should be there after every raid which is used to segregate the entire match video into individual raids
2. The camera angle should be static for the whole match



**Create a Virtual Environment:**
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

**Install Dependencies:**
pip install -r requirements.txt

**Configure Environment Variables:**
Create a .env file in the root directory and add your Google Gemini API Key:
GEMINI_API_KEY=your_api_key_here

**Run the Application:**
streamlit run demo_ui.py

**💻 Usage Guide**
Match Configuration: Open the sidebar in the UI and enter the names of the competing teams (e.g., "PUNERI PALTAN" and "JAIPUR PINK PANTHERS"). The Audio Agent will dynamically adjust its vocabulary to listen for these teams.

Upload Footage: Upload an .mp4 file of a Kabaddi match.

Start Processing: Click the Start Full Processing button. The dashboard will live-stream the logs of the Whistle Detection, Vision Agent, Audio Agent, and Graph Generation.

Ask the AI Coach: Once processing is complete, use the search bar at the bottom to ask tactical questions (e.g., "Why did the raider fail on the left flank?" or "Analyze the defensive formation density."). The system will return the relevant video clip, a textual analysis, and dynamic Plotly charts.

📂 Project Structure
Plaintext
Kabaddi-Neuro/
├── data/                       # Directory for raw videos, processed clips, and unified JSONs
├── src/
│   ├── agents/
│   │   ├── vision_agent.py     # YOLOv8 and ResNet logic
│   │   ├── audio_agent.py      # Whisper transcription and phonetic filtering
│   │   └── reasoning_agent.py  # LLM prompt engineering and response generation
│   ├── graph/
│   │   ├── vector_db.py        # ChromaDB setup and search logic
│   │   └── knowledge_graph.py  # NetworkX topology and global outcome tracking
│   └── utils/
│       ├── video_utils.py      # FFmpeg whistle detection and clipping
│       └── sync_manager.py     # Merges audio and visual data into Unified JSON
├── demo_ui.py                  # Main Streamlit Dashboard
├── requirements.txt            # Python dependencies
└── .env                        # API keys and environment variables
