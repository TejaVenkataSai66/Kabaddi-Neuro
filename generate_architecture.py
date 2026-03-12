import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure
fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Helper function to draw components (Fixed to use FancyBboxPatch for rounded corners)
def draw_component(x, y, w, h, text, color='#FFFFFF', edge='#000000'):
    # FancyBboxPatch creates the rounded box effect
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                  linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', wrap=True)

# Helper for Arrows
def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1.5))

# --- LAYER 4: APPLICATION LAYER (Top) ---
# Background Box (Standard Rectangle is fine for background)
rect = patches.Rectangle((5, 78), 90, 18, linewidth=2, edgecolor='#333333', facecolor='#E3F2FD', alpha=0.3)
ax.add_patch(rect)
ax.text(50, 93, "LAYER 4: APPLICATION & INTERACTION", ha='center', fontsize=12, fontweight='bold', color='#1565C0')

# Components
draw_component(15, 81, 30, 8, "Web Dashboard\n(Streamlit UI)", color='#BBDEFB')
draw_component(55, 81, 30, 8, "Strategic Response Engine\n(Gemini 1.5 Flash)", color='#BBDEFB')

# --- LAYER 3: KNOWLEDGE & REASONING LAYER ---
# Background Box
rect = patches.Rectangle((5, 54), 90, 18, linewidth=2, edgecolor='#333333', facecolor='#E8F5E9', alpha=0.3)
ax.add_patch(rect)
ax.text(50, 69, "LAYER 3: KNOWLEDGE & REASONING", ha='center', fontsize=12, fontweight='bold', color='#2E7D32')

# Components
draw_component(15, 57, 30, 8, "Vector Database\n(ChromaDB)\n[Semantic Search]", color='#C8E6C9')
draw_component(55, 57, 30, 8, "Tactical Knowledge Graph\n(NetworkX)\n[Pattern Recognition]", color='#C8E6C9')

# --- LAYER 2: PERCEPTION & INTELLIGENCE LAYER ---
# Background Box
rect = patches.Rectangle((5, 30), 90, 18, linewidth=2, edgecolor='#333333', facecolor='#FFF3E0', alpha=0.3)
ax.add_patch(rect)
ax.text(50, 45, "LAYER 2: PERCEPTION & INTELLIGENCE", ha='center', fontsize=12, fontweight='bold', color='#EF6C00')

# Components
draw_component(10, 33, 22, 8, "Visual Intelligence\n(YOLOv8 Pose)", color='#FFE0B2')
draw_component(39, 33, 22, 8, "Sync & Alignment\n(Temporal Logic)", color='#FFCC80')
draw_component(68, 33, 22, 8, "Audio Intelligence\n(Whisper ASR)", color='#FFE0B2')

# --- LAYER 1: INGESTION LAYER (Bottom) ---
# Background Box
rect = patches.Rectangle((5, 6), 90, 18, linewidth=2, edgecolor='#333333', facecolor='#F3E5F5', alpha=0.3)
ax.add_patch(rect)
ax.text(50, 21, "LAYER 1: DATA INGESTION", ha='center', fontsize=12, fontweight='bold', color='#6A1B9A')

# Components
draw_component(35, 9, 30, 8, "Video Processor\n(FFmpeg / OpenCV)\n[Whistle Detection & Cutting]", color='#E1BEE7')

# --- DATA FLOW ARROWS ---
# Ingestion -> Perception
draw_arrow(50, 17, 21, 33) # To Vision
draw_arrow(50, 17, 50, 33) # To Sync
draw_arrow(50, 17, 79, 33) # To Audio

# Perception -> Knowledge
draw_arrow(21, 41, 50, 44) # Vision to Sync (Conceptual)
draw_arrow(79, 41, 50, 44) # Audio to Sync (Conceptual)
draw_arrow(50, 41, 30, 57) # Sync to Vector DB
draw_arrow(50, 41, 70, 57) # Sync to Graph

# Knowledge -> Application
draw_arrow(30, 65, 30, 81) # Vector DB to Dashboard
draw_arrow(70, 65, 70, 81) # Graph to LLM
draw_arrow(30, 65, 70, 81) # Vector DB to LLM (Context)
draw_arrow(70, 89, 45, 85) # LLM to Dashboard

plt.title("Kabaddi Neuro: Proposed System Architecture", fontsize=16, fontweight='bold', y=0.98)
plt.savefig("Kabaddi_Neuro_Architecture.png", dpi=300, bbox_inches='tight')
print("✅ Architecture diagram saved as 'Kabaddi_Neuro_Architecture.png'")