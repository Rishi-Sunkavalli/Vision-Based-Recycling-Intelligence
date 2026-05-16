"""
app.py — Streamlit Dashboard
Vision-Based Recycling Intelligence Prototype
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from pathlib import Path
from backend import analyze_image, analyze_video, generate_report

import os

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY missing. Add it in Streamlit Secrets.")
    st.stop()
# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="♻️ RecycleVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0f1117; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #1e2130;
    border: 1px solid #2e3450;
    border-radius: 12px;
    padding: 16px 20px;
  }
  div[data-testid="metric-container"] label { color: #8892b0 !important; font-size: 0.78rem !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #ccd6f6 !important; font-size: 1.6rem !important; font-weight: 700;
  }

  /* Grade badge */
  .grade-badge {
    display: inline-block;
    font-size: 3rem;
    font-weight: 900;
    border-radius: 50%;
    width: 80px; height: 80px;
    line-height: 80px;
    text-align: center;
    margin: 0 auto;
  }
  .grade-A { background: #1a3a2a; color: #64ffda; border: 3px solid #64ffda; }
  .grade-B { background: #1a2a3a; color: #57a6e8; border: 3px solid #57a6e8; }
  .grade-C { background: #3a2e1a; color: #f5a623; border: 3px solid #f5a623; }
  .grade-D { background: #3a1a1a; color: #ff6b6b; border: 3px solid #ff6b6b; }

  /* Review flag */
  .review-flag {
    background: #3a2000;
    border: 1px solid #f5a623;
    border-radius: 10px;
    padding: 12px 20px;
    color: #f5a623;
    font-weight: 600;
  }

  /* Section headers */
  .section-header {
    color: #64ffda;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid #2e3450;
    padding-bottom: 6px;
    margin-bottom: 12px;
  }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #151821; border-right: 1px solid #2e3450; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ♻️ RecycleVision AI")
    st.caption("Vision-Based Recycling Intelligence")
    st.divider()

    st.markdown("### Upload Batch Media")
    uploaded_file = st.file_uploader(
        "Drop image or video here",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi"],
        help="Supported: JPG, PNG, WEBP, MP4, MOV, AVI"
    )

    max_frames = 4
    if uploaded_file and uploaded_file.name.lower().endswith(("mp4", "mov", "avi")):
        max_frames = st.slider("Frames to sample (video)", 2, 6, 4)

    analyze_btn = st.button("🔍 Analyse Batch", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "This prototype uses **Llama 4 Scout Vision** via the **Groq API** to analyse recycling batches. "
        "It detects bottle count, PET composition, colour & quality distribution, "
        "contamination risk, and estimates batch value."
    )

    st.divider()
    st.markdown("### Tech Stack")
    st.caption("• Groq API — llama-4-scout-17b (VLM)\n• OpenCV (video frame extraction)\n• Streamlit (dashboard)\n• Plotly (charts)")


# ─── Main area ────────────────────────────────────────────────────────────────
st.markdown("# ♻️ Recycling Batch Intelligence Dashboard")
st.caption("Upload an image or video of incoming plastic bottles to get instant AI analysis.")

# ── Placeholder / welcome state ──
if "result" not in st.session_state:
    st.session_state.result = None
if "report_text" not in st.session_state:
    st.session_state.report_text = None

# ── Run analysis ──
if analyze_btn and uploaded_file:
    with st.spinner("Analysing batch with Groq Vision (llama-4-scout)…"):
        try:
            uploaded_file.seek(0)
            ext = Path(uploaded_file.name).suffix.lower()
            if ext in (".mp4", ".mov", ".avi"):
                result = analyze_video(uploaded_file, max_frames=max_frames)
            else:
                result = analyze_image(uploaded_file)

            st.session_state.result = result
            st.session_state.report_text = generate_report(result)
            st.success("✅ Analysis complete!")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
elif analyze_btn and not uploaded_file:
    st.warning("Please upload an image or video first.")


# ─── Display results ──────────────────────────────────────────────────────────
result = st.session_state.result

if result is None:
    # Welcome placeholder
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📤 **Step 1** — Upload a batch image or video using the sidebar.")
    with col2:
        st.info("🔍 **Step 2** — Click **Analyse Batch** to run AI vision analysis.")
    with col3:
        st.info("📊 **Step 3** — Review the dashboard, download report or JSON.")
    st.divider()
    st.markdown("""
    **What this system detects:**
    - 🍶 Bottle count & estimated weight
    - 🔵 PET vs non-PET composition
    - 🎨 Colour distribution (clear, green, blue, brown, other)
    - 🧹 Quality & contamination assessment
    - 💰 Batch grade (A–D) and estimated market value
    - ⚠️ Human-review flag for uncertain or high-risk batches
    """)
else:
    # ── Human review alert ──
    if result.get("human_review_flag"):
        st.markdown(
            f'<div class="review-flag">⚠️ HUMAN REVIEW REQUIRED — {result.get("human_review_reason", "")}</div>',
            unsafe_allow_html=True
        )
        st.write("")

    # ── Top KPI strip ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Batch Overview</p>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Bottle Count", result.get("bottle_count", "–"))
    k2.metric("Est. Weight", f"{result.get('estimated_weight_kg', 0):.1f} kg")
    k3.metric("PET Content", f"{result.get('pet_percentage', 0):.0f}%")
    k4.metric("Contamination", result.get("contamination_risk", "–").upper())
    k5.metric("Confidence", f"{result.get('confidence', 0)*100:.0f}%")
    k6.metric("Est. Value", f"${result.get('estimated_total_value', 0):.2f}")

    st.write("")

    # ── Grade card + Composition donut ────────────────────────────────────
    st.markdown('<p class="section-header">Grade & Composition</p>', unsafe_allow_html=True)
    left, mid, right = st.columns([1, 2, 2])

    with left:
        grade = result.get("batch_grade", "C")
        grade_class = f"grade-{grade}"
        val_per_kg = result.get("estimated_value_per_kg", 0)
        st.markdown(
            f'<div style="text-align:center">'
            f'<div class="grade-badge {grade_class}">{grade}</div>'
            f'<br><span style="color:#8892b0;font-size:0.8rem;">Batch Grade</span><br>'
            f'<span style="color:#ccd6f6;font-weight:600;">${val_per_kg:.3f}/kg</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with mid:
        # PET vs Non-PET donut
        pet = result.get("pet_percentage", 50)
        non_pet = result.get("non_pet_percentage", 50)
        fig_donut = go.Figure(go.Pie(
            labels=["PET", "Non-PET"],
            values=[pet, non_pet],
            hole=0.55,
            marker_colors=["#64ffda", "#ff6b6b"],
            textfont_color="white",
        ))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(font_color="#ccd6f6", bgcolor="rgba(0,0,0,0)"),
            height=220,
            title=dict(text="Material Composition", font_color="#ccd6f6", x=0.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with right:
        # Colour distribution bar
        color_data = result.get("color_distribution", {})
        color_map = {
            "clear": "#e0f7fa", "green": "#43a047",
            "blue": "#1e88e5", "brown": "#8d6e63", "other": "#78909c"
        }
        fig_color = go.Figure(go.Bar(
            x=list(color_data.values()),
            y=[k.capitalize() for k in color_data.keys()],
            orientation="h",
            marker_color=[color_map.get(k, "#aaa") for k in color_data.keys()],
            text=[f"{v:.0f}%" for v in color_data.values()],
            textposition="outside",
            textfont_color="#ccd6f6",
        ))
        fig_color.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(color="#ccd6f6")),
            margin=dict(t=30, b=10, l=10, r=40),
            height=220,
            title=dict(text="Colour Distribution", font_color="#ccd6f6", x=0),
        )
        st.plotly_chart(fig_color, use_container_width=True)

    # ── Quality distribution ───────────────────────────────────────────────
    st.markdown('<p class="section-header">Quality Distribution</p>', unsafe_allow_html=True)
    quality_data = result.get("quality_distribution", {})
    q_labels = [k.replace("_", " ").title() for k in quality_data.keys()]
    q_values = list(quality_data.values())
    q_colors = ["#64ffda", "#57a6e8", "#f5a623", "#ff6b6b"]

    fig_quality = go.Figure(go.Bar(
        x=q_labels,
        y=q_values,
        marker_color=q_colors[: len(q_labels)],
        text=[f"{v:.0f}%" for v in q_values],
        textposition="outside",
        textfont_color="#ccd6f6",
    ))
    fig_quality.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            showgrid=True, gridcolor="#2e3450", zeroline=False,
            ticksuffix="%", tickfont=dict(color="#8892b0"), range=[0, 110]
        ),
        xaxis=dict(tickfont=dict(color="#ccd6f6")),
        margin=dict(t=20, b=10),
        height=260,
    )
    st.plotly_chart(fig_quality, use_container_width=True)

    # ── Observations & Recommendations ────────────────────────────────────
    st.markdown('<p class="section-header">Insights</p>', unsafe_allow_html=True)
    obs_col, rec_col = st.columns(2)
    with obs_col:
        st.markdown("**🔍 Key Observations**")
        for obs in result.get("key_observations", []):
            st.markdown(f"• {obs}")

    with rec_col:
        st.markdown("**💡 Recommendations**")
        for rec in result.get("recommendations", []):
            st.markdown(f"→ {rec}")

    # ── Export ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-header">Export</p>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        report_text = st.session_state.report_text or ""
        st.download_button(
            "📄 Download Text Report",
            data=report_text,
            file_name="recycling_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with dl2:
        json_str = json.dumps(result, indent=2)
        st.download_button(
            "📦 Download JSON",
            data=json_str,
            file_name="recycling_analysis.json",
            mime="application/json",
            use_container_width=True,
        )

    with dl3:
        # Summary table as CSV
        summary = {
            "Metric": ["Bottle Count", "Weight (kg)", "PET %", "Non-PET %",
                       "Contamination Risk", "Confidence", "Grade", "Value/kg", "Total Value ($)"],
            "Value": [
                result.get("bottle_count"),
                result.get("estimated_weight_kg"),
                result.get("pet_percentage"),
                result.get("non_pet_percentage"),
                result.get("contamination_risk"),
                f"{result.get('confidence', 0)*100:.0f}%",
                result.get("batch_grade"),
                result.get("estimated_value_per_kg"),
                result.get("estimated_total_value"),
            ]
        }
        csv_str = pd.DataFrame(summary).to_csv(index=False)
        st.download_button(
            "📊 Download CSV Summary",
            data=csv_str,
            file_name="recycling_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Raw JSON expander ──────────────────────────────────────────────────
    with st.expander("🔧 Raw JSON Response"):
        st.json(result)
