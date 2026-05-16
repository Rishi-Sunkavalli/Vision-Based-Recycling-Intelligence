"""
architecture.py — Generate architecture diagram as PNG using matplotlib.
Run standalone: python architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

def box(x, y, w, h, label, sublabel="", color="#1e2130", text_color="#ccd6f6",
        border="#64ffda", fontsize=10):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=border, linewidth=1.5)
    ax.add_patch(rect)
    cx, cy = x + w / 2, y + h / 2
    if sublabel:
        ax.text(cx, cy + 0.18, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold")
        ax.text(cx, cy - 0.22, sublabel, ha="center", va="center",
                color="#8892b0", fontsize=fontsize - 2)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold")

def arrow(x1, y1, x2, y2, color="#64ffda"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

# Title
ax.text(8, 8.6, "RecycleVision AI — Architecture", ha="center", va="center",
        color="#64ffda", fontsize=16, fontweight="bold")

# ── Layer 0: Input ──────────────────────────────────────────────────────
ax.text(2, 7.9, "INPUT", ha="center", color="#8892b0", fontsize=8)
box(0.3, 6.6, 1.6, 0.9, "Image Input", "JPG/PNG/WEBP", border="#57a6e8")
box(2.1, 6.6, 1.6, 0.9, "Video Input", "MP4/MOV/AVI", border="#57a6e8")

# ── Layer 1: Ingestion ─────────────────────────────────────────────────
ax.text(2, 6.0, "INGESTION", ha="center", color="#8892b0", fontsize=8)
box(0.3, 5.0, 3.4, 0.85, "File Handler", "load_image_bytes()\nextract_video_frames()", border="#f5a623")

# ── Layer 2: Pre-processing ────────────────────────────────────────────
ax.text(2, 4.45, "PRE-PROCESS", ha="center", color="#8892b0", fontsize=8)
box(0.3, 3.5, 1.5, 0.7, "Resize", "max 1568px", border="#f5a623")
box(2.2, 3.5, 1.5, 0.7, "Base64\nEncode", "", border="#f5a623")

# ── Layer 3: Vision Model ──────────────────────────────────────────────
ax.text(7.5, 7.9, "AI BACKBONE", ha="center", color="#8892b0", fontsize=8)
box(5.5, 6.2, 4.0, 1.4, "Claude Opus 4\n(Vision Language Model)",
    "claude-opus-4-5 via Anthropic API", color="#1a3a2a", border="#64ffda", fontsize=11)

# ── Layer 4: Post-processing ───────────────────────────────────────────
ax.text(7.5, 5.55, "POST-PROCESS", ha="center", color="#8892b0", fontsize=8)
box(5.5, 4.6, 1.8, 0.75, "JSON Parser", "strip fences, validate", border="#f5a623")
box(7.7, 4.6, 1.8, 0.75, "Frame Merger", "avg / worst-case", border="#f5a623")

# ── Layer 5: Analysis outputs ──────────────────────────────────────────
ax.text(7.5, 3.95, "ANALYSIS OUTPUTS", ha="center", color="#8892b0", fontsize=8)
outputs = [
    ("Bottle\nCount", 5.0),
    ("PET\nComp.", 6.3),
    ("Colour\nDist.", 7.6),
    ("Review\nFlag", 8.9),
    ("Batch\nGrade", 10.2),
]
for label, x in outputs:
    box(x, 3.0, 1.1, 0.75, label, "", border="#57a6e8", fontsize=8)

# ── Layer 6: Dashboard ─────────────────────────────────────────────────
ax.text(7.5, 2.35, "PRESENTATION", ha="center", color="#8892b0", fontsize=8)
box(5.0, 1.4, 2.0, 0.75, "Streamlit\nDashboard", "", color="#1a2a3a", border="#64ffda")
box(7.3, 1.4, 1.7, 0.75, "Text\nReport", "", color="#1a2a3a", border="#64ffda")
box(9.3, 1.4, 1.4, 0.75, "JSON\nExport", "", color="#1a2a3a", border="#64ffda")
box(11.0, 1.4, 1.4, 0.75, "CSV\nExport", "", color="#1a2a3a", border="#64ffda")

# ── Right panel: MLOps ─────────────────────────────────────────────────
ax.text(13.5, 7.9, "MLOps / FUTURE", ha="center", color="#8892b0", fontsize=8)
mlops = [
    ("Human Review\nQueue", 7.1), ("Label Store\n(S3 + CVAT)", 6.1),
    ("Fine-tuned Model\n(future)", 5.1), ("Model Registry\n& Versioning", 4.1),
    ("Monitoring\n& Drift Alerts", 3.1),
]
for label, y in mlops:
    box(12.0, y, 3.0, 0.75, label, "", color="#1e2130", border="#8892b0", fontsize=9)

# ── Arrows ─────────────────────────────────────────────────────────────
# Input → ingestion
arrow(1.1, 6.6, 1.1, 5.85)
arrow(2.9, 6.6, 2.9, 5.85)
# Ingestion → preprocessing
arrow(1.7, 5.0, 1.0, 4.2)
arrow(2.5, 5.0, 2.8, 4.2)
# Preprocessing → Claude
arrow(3.7, 3.85, 5.5, 6.6)
# Claude → parser
arrow(7.0, 6.2, 6.5, 5.35)
# Parser → merger
arrow(7.3, 4.97, 7.9, 4.97)  # horizontal
# Merger → outputs
for x in [5.55, 6.85, 8.15, 9.45, 10.75]:
    arrow(8.6, 4.6, x, 3.75)
# Outputs → dashboard
for x in [5.55, 6.85, 8.15, 9.45, 10.75]:
    arrow(x, 3.0, 8.0, 2.15)

# ── Legend ─────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color="#64ffda", label="Core AI / Output"),
    mpatches.Patch(color="#f5a623", label="Data Processing"),
    mpatches.Patch(color="#57a6e8", label="I/O Interfaces"),
    mpatches.Patch(color="#8892b0", label="MLOps (Planned)"),
]
ax.legend(handles=patches, loc="lower left", framealpha=0,
          labelcolor="#ccd6f6", fontsize=9, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig("architecture.png", dpi=150, bbox_inches="tight", facecolor="#0f1117")
print("Saved architecture.png")
