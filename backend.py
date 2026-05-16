"""
backend.py — Vision-Based Recycling Intelligence
Core analysis engine using Groq Vision API (llama-4-scout-17b via Groq).
"""

import os
import base64
import json
import re
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
import cv2
import numpy as np
from groq import Groq

# ─── Groq client (lazy init so import works without key set) ─────────────────
_client = None

def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Export it with: export GROQ_API_KEY='gsk_...'"
            )
        _client = Groq(api_key=api_key)
    return _client

# Best vision model on Groq — fast, multimodal, strong instruction following
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ─── Prompts ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a recycling facility AI analyst specialising in plastic bottle classification.
Analyse images and return ONLY valid JSON — no markdown, no backticks, no preamble, no explanation.

Output schema (all fields required):
{
  "bottle_count": <integer>,
  "pet_percentage": <0-100 float>,
  "non_pet_percentage": <0-100 float>,
  "color_distribution": {
    "clear": <0-100 float>,
    "green": <0-100 float>,
    "blue": <0-100 float>,
    "brown": <0-100 float>,
    "other": <0-100 float>
  },
  "quality_distribution": {
    "clean": <0-100 float>,
    "lightly_soiled": <0-100 float>,
    "heavily_soiled": <0-100 float>,
    "crushed_deformed": <0-100 float>
  },
  "contamination_risk": <"low"|"medium"|"high">,
  "confidence": <0.0-1.0 float>,
  "human_review_flag": <true|false>,
  "human_review_reason": <string or null>,
  "batch_grade": <"A"|"B"|"C"|"D">,
  "estimated_value_per_kg": <float in USD>,
  "estimated_weight_kg": <float>,
  "estimated_total_value": <float in USD>,
  "key_observations": [<string>, ...],
  "recommendations": [<string>, ...]
}

Grading rubric:
- Grade A: >90% PET, <5% contamination, mostly clear/clean -> $0.25-0.35/kg
- Grade B: 70-90% PET, 5-15% contamination -> $0.15-0.25/kg
- Grade C: 50-70% PET, 15-30% contamination -> $0.08-0.15/kg
- Grade D: <50% PET or >30% contamination -> $0.02-0.08/kg

Flag for human review if: confidence < 0.6, contamination_risk=high, or unusual materials detected.
Be conservative with counts when uncertain. Base weight estimate on count x avg bottle weight (0.03 kg).
Ensure color_distribution and quality_distribution each sum to 100."""

USER_PROMPT = """Analyse this recycling batch image. Identify all plastic bottles visible,
estimate their types (PET vs non-PET), colours, quality, and contamination level.
Return structured JSON only, nothing else."""


# ─── Image helpers ────────────────────────────────────────────────────────────

def load_image_bytes(file_obj) -> tuple[bytes, str]:
    """Accept a file-like object or path, return (raw_bytes, media_type)."""
    if isinstance(file_obj, (str, Path)):
        with open(file_obj, "rb") as f:
            data = f.read()
        ext = Path(str(file_obj)).suffix.lower()
    else:
        data = file_obj.read()
        ext = Path(file_obj.name).suffix.lower() if hasattr(file_obj, "name") else ".jpg"

    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    return data, media_map.get(ext, "image/jpeg")


def extract_video_frames(video_bytes: bytes, max_frames: int = 4) -> list[tuple[bytes, str]]:
    """Extract evenly-spaced frames from a video, return list of (bytes, media_type)."""
    tmp = "/tmp/_upload_video.mp4"
    with open(tmp, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(tmp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, min(max_frames, total), dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        frames.append((buf.getvalue(), "image/jpeg"))

    cap.release()
    return frames


def resize_image(image_bytes: bytes, max_dim: int = 1024) -> bytes:
    """
    Resize to max 1024px on longest edge and ensure JPEG output.
    Groq vision models perform best with reasonably-sized images.
    """
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def image_to_data_url(image_bytes: bytes) -> str:
    """Encode image bytes to a base64 JPEG data URL for Groq's vision API."""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ─── Core Groq call ───────────────────────────────────────────────────────────

def _call_groq(frames: list[tuple[bytes, str]]) -> dict:
    """
    Send image frame(s) to Groq Vision (llama-4-scout) and parse JSON response.
    Groq accepts image_url content blocks with base64 data URLs.
    Multiple frames are passed as multiple image blocks in one message.
    """
    content = []

    for img_bytes, _ in frames:
        img_bytes = resize_image(img_bytes)
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(img_bytes)},
        })

    content.append({"type": "text", "text": USER_PROMPT})

    response = get_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=1200,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE)

    return json.loads(raw.strip())


# ─── Multi-frame merger (for video) ───────────────────────────────────────────

def _merge_frame_results(results: list[dict]) -> dict:
    """Average numeric fields across multiple frame analyses."""
    if len(results) == 1:
        return results[0]

    def avg(key):
        vals = [r.get(key, 0) for r in results if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else 0

    def avg_dict(key):
        sample = results[0].get(key, {})
        return {
            k: round(sum(r.get(key, {}).get(k, 0) for r in results) / len(results), 1)
            for k in sample
        }

    risk_order = {"low": 0, "medium": 1, "high": 2}
    max_risk = max(results, key=lambda r: risk_order.get(r.get("contamination_risk", "low"), 0))
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    worst_grade = max(results, key=lambda r: grade_order.get(r.get("batch_grade", "D"), 3))

    all_obs = []
    for r in results:
        for obs in r.get("key_observations", []):
            if obs not in all_obs:
                all_obs.append(obs)

    return {
        "bottle_count": round(avg("bottle_count")),
        "pet_percentage": avg("pet_percentage"),
        "non_pet_percentage": avg("non_pet_percentage"),
        "color_distribution": avg_dict("color_distribution"),
        "quality_distribution": avg_dict("quality_distribution"),
        "contamination_risk": max_risk.get("contamination_risk", "medium"),
        "confidence": avg("confidence"),
        "human_review_flag": any(r.get("human_review_flag") for r in results),
        "human_review_reason": next(
            (r.get("human_review_reason") for r in results if r.get("human_review_flag")), None
        ),
        "batch_grade": worst_grade.get("batch_grade", "C"),
        "estimated_value_per_kg": avg("estimated_value_per_kg"),
        "estimated_weight_kg": avg("estimated_weight_kg"),
        "estimated_total_value": avg("estimated_total_value"),
        "key_observations": all_obs[:6],
        "recommendations": results[0].get("recommendations", []),
        "frame_count_analyzed": len(results),
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_image(file_obj) -> dict:
    """
    Analyse a single image file.
    file_obj: Streamlit UploadedFile, file-like, or path string.
    Returns structured analysis dict.
    """
    img_bytes, media_type = load_image_bytes(file_obj)
    result = _call_groq([(img_bytes, media_type)])
    result["input_type"] = "image"
    result["analysis_timestamp"] = datetime.utcnow().isoformat() + "Z"
    result["model_used"] = VISION_MODEL
    return result


def analyze_video(file_obj, max_frames: int = 4) -> dict:
    """
    Analyse a video by sampling frames.
    All sampled frames are passed together in one Groq call.
    """
    if isinstance(file_obj, (str, Path)):
        with open(file_obj, "rb") as f:
            video_bytes = f.read()
    else:
        video_bytes = file_obj.read()

    frames = extract_video_frames(video_bytes, max_frames=max_frames)
    if not frames:
        raise ValueError("Could not extract any frames from the video.")

    result = _call_groq(frames)
    result["input_type"] = "video"
    result["frames_sampled"] = len(frames)
    result["analysis_timestamp"] = datetime.utcnow().isoformat() + "Z"
    result["model_used"] = VISION_MODEL
    return result


def generate_report(analysis: dict) -> str:
    """Generate a plain-text operational report from analysis results."""
    ts = analysis.get("analysis_timestamp", "N/A")
    model = analysis.get("model_used", VISION_MODEL)
    lines = [
        "=" * 60,
        "  RECYCLING BATCH ANALYSIS REPORT",
        f"  Generated : {ts}",
        f"  Model     : {model}",
        "=" * 60,
        "",
        f"  Input type      : {analysis.get('input_type', 'N/A').upper()}",
        f"  Frames analysed : {analysis.get('frame_count_analyzed', 1)}",
        "",
        "-- BATCH SUMMARY -------------------------------------------",
        f"  Bottle count        : {analysis.get('bottle_count', 'N/A')}",
        f"  Estimated weight    : {analysis.get('estimated_weight_kg', 0):.2f} kg",
        f"  PET material        : {analysis.get('pet_percentage', 0):.1f}%",
        f"  Non-PET material    : {analysis.get('non_pet_percentage', 0):.1f}%",
        f"  Contamination risk  : {analysis.get('contamination_risk', 'N/A').upper()}",
        f"  Model confidence    : {analysis.get('confidence', 0)*100:.0f}%",
        "",
        "-- BATCH GRADE & VALUE -------------------------------------",
        f"  Grade               : {analysis.get('batch_grade', 'N/A')}",
        f"  Value per kg        : ${analysis.get('estimated_value_per_kg', 0):.3f}",
        f"  Estimated value     : ${analysis.get('estimated_total_value', 0):.2f}",
        "",
        "-- COLOUR DISTRIBUTION -------------------------------------",
    ]
    for color, pct in analysis.get("color_distribution", {}).items():
        lines.append(f"  {color:<20}: {pct:.1f}%")

    lines += ["", "-- QUALITY DISTRIBUTION ------------------------------------"]
    for quality, pct in analysis.get("quality_distribution", {}).items():
        lines.append(f"  {quality.replace('_', ' '):<20}: {pct:.1f}%")

    if analysis.get("human_review_flag"):
        lines += [
            "",
            "*** HUMAN REVIEW REQUIRED ***",
            f"    Reason: {analysis.get('human_review_reason', 'Uncertain classification')}",
        ]

    lines += ["", "-- KEY OBSERVATIONS ----------------------------------------"]
    for obs in analysis.get("key_observations", []):
        lines.append(f"  * {obs}")

    lines += ["", "-- RECOMMENDATIONS -----------------------------------------"]
    for rec in analysis.get("recommendations", []):
        lines.append(f"  -> {rec}")

    lines += ["", "=" * 60]
    return "\n".join(lines)
