# ♻️ RecycleVision AI — Prototype

Vision-Based Recycling Intelligence system for plastic bottle classification and batch valuation.

---

## Quick Start

```bash
# 1. Clone / extract the project
cd recycling_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
export GROQ_API_KEY="gsk_..."

# 4. Run the Streamlit dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser, upload a batch image or short video, and click **Analyse Batch**.

---

## Project Structure

```
recycling_ai/
├── app.py              # Streamlit dashboard (frontend)
├── backend.py          # Analysis engine (backend)
├── architecture.py     # Architecture diagram generator
├── requirements.txt
└── README.md
```

---

## Architecture & Design Decisions

### AI/ML Approach — What I Chose

**Primary model: Llama 4 Scout Vision via Groq API (`meta-llama/llama-4-scout-17b-16e-instruct`)**

| Approach | Chosen? | Reason |
|---|---|---|
| Groq Vision (llama-4-scout) | Yes | Ultra-fast inference (~500ms); free tier available; strong vision + instruction following; zero training data needed; produces structured JSON |
| Claude Vision / GPT-4o | Alternative | Excellent accuracy but higher latency & cost; easy swap-in if needed |
| YOLOv8 / Detectron2 (object detection) | No (prototype) | Needs labelled training data. Strong for count accuracy at scale, but cold-start problem kills it for day 1 |
| Classic CV (HSV colour thresholding) | No | Brittle under varying lighting |
| OCR (reading resin codes) | No | Only works if label is visible and clean — unreliable in real batches |
| Segmentation (SAM / Mask R-CNN) | Future | Great for per-bottle masks; combine with VLM classification |

**Trade-offs:**
- VLMs are slower per-image (~2–4 s) vs. YOLO (<50 ms). Acceptable for batch intake scanning, not real-time conveyor belts.
- Cost: ~$0.002–0.005 per image at current API pricing. For 1,000 batches/day, ~$2–5/day.
- Accuracy depends on image quality. Poor lighting, occlusion, or low resolution reduces reliability — hence the confidence score and human-review flag.

### What I Rejected (and Why)

1. **Fine-tuned YOLO** — Ideal long-term, but requires 5,000+ labelled images per class minimum. No data → no model. VLM bridges the gap.
2. **Rule-based colour detection** — Failed on partially occluded bottles, mixed lighting, coloured labels on clear bottles.
3. **LLM-only (text)** — Cannot reason about actual pixel content without vision.

---

## Five Outputs Delivered

| Output | How |
|---|---|
| Bottle count | Claude Vision counts visible bottles |
| PET vs non-PET composition | Claude estimates based on shape, colour, label cues |
| Colour & quality distribution | Structured JSON with percentages |
| Confidence score per prediction | Claude self-reports confidence (0–1) |
| Human-review flag | Auto-triggered if confidence < 0.6, contamination = high, or unusual materials |
| Batch grade & value | A–D grading rubric with $/kg estimate |

---

## Data Strategy

### Collection
- **Phase 1 (Bootstrap):** Use VLM predictions as weak labels. Export JSON predictions for each batch. Store image + prediction in S3.
- **Phase 2 (Active Learning):** Route low-confidence predictions (< 0.6) to human reviewers (CVAT or Label Studio). Prioritise ambiguous cases.
- **Phase 3 (Scale):** Use corrected labels to fine-tune a lightweight object detection model (YOLOv8n or RT-DETR).

### Labelling Schema
- Bounding box per bottle
- Class: PET / HDPE / PVC / other
- Colour: clear / green / blue / brown / other
- Condition: clean / lightly_soiled / heavily_soiled / crushed

### Improvement Loop
```
Image → VLM prediction → Confidence gate
    ├─ High confidence → auto-accept, log
    └─ Low confidence → human review → corrected label → retrain queue
```

Target: After 10,000 labelled images, replace VLM with fine-tuned YOLO + classifier. Keep VLM as fallback for edge cases.

---

## Evaluation Plan

Beyond accuracy:

| Metric | Why it matters |
|---|---|
| **Precision/Recall per material class** | Missing PET in batch = lost revenue; false PET = contamination fines |
| **Mean Absolute Error (bottle count)** | Operational: count drives weight and value estimates |
| **Calibration (confidence vs. actual accuracy)** | Confidence scores are only useful if calibrated; measure via reliability diagrams |
| **Human review rate** | Business SLA: if >20% of batches need human review, the system adds overhead rather than saving it |
| **Time-to-result** | Facility throughput — target < 5 s per batch |
| **$ Error (estimated vs. actual value)** | Directly tied to business outcome — track MAPE on batch value |
| **Contamination false negative rate** | Missing high-risk batches is costly (equipment damage, downstream rejections) |

---

## MLOps: Prototype → Production

| Stage | Action |
|---|---|
| **Prototype** (now) | Claude Vision API, Streamlit dashboard, manual uploads |
| **Pilot** | Deploy on a server near the facility; integrate camera feed → auto-upload |
| **Production** | Containerise (Docker), expose REST API, queue via Celery + Redis |
| **Model management** | MLflow for experiment tracking; model versioning in S3 |
| **Monitoring** | Track confidence distribution, human-review rate, value MAE weekly |
| **Retraining** | Trigger fine-tune job when labelled set > 2,000 new samples |
| **Fallback** | If fine-tuned model confidence < 0.5, escalate to VLM |

### REST API (future)
```
POST /analyse
  Content-Type: multipart/form-data
  Body: { file: <image|video> }

Response 200:
  { bottle_count, pet_percentage, batch_grade, estimated_total_value,
    confidence, human_review_flag, ... }
```

---

## Assumptions

1. Input images show the top-down or angled view of a batch of bottles (not individual bottles at high zoom).
2. Lighting is reasonable — not pitch-dark or heavily overexposed.
3. The facility has internet connectivity for API calls.
4. Batch size is < 500 bottles per image (beyond that, count accuracy degrades).
5. Video inputs are < 30 s; we sample up to 6 frames.

---

## Sample Output (JSON)

```json
{
  "bottle_count": 47,
  "pet_percentage": 82.0,
  "non_pet_percentage": 18.0,
  "color_distribution": {
    "clear": 55.0, "green": 15.0, "blue": 20.0, "brown": 5.0, "other": 5.0
  },
  "quality_distribution": {
    "clean": 60.0, "lightly_soiled": 25.0,
    "heavily_soiled": 10.0, "crushed_deformed": 5.0
  },
  "contamination_risk": "low",
  "confidence": 0.81,
  "human_review_flag": false,
  "human_review_reason": null,
  "batch_grade": "B",
  "estimated_value_per_kg": 0.22,
  "estimated_weight_kg": 1.41,
  "estimated_total_value": 0.31,
  "key_observations": [
    "Majority of bottles appear to be standard PET water/beverage bottles",
    "Some HDPE containers visible in upper-left cluster",
    "Minor soil/label contamination on ~25% of bottles"
  ],
  "recommendations": [
    "Sort out HDPE containers before baling to improve grade to A",
    "Rinse lightly soiled bottles to increase value per kg"
  ],
  "input_type": "image",
  "analysis_timestamp": "2026-05-16T08:22:11Z"
}
```

---

## Limitations

- **Count accuracy degrades** with dense piles (> 300 bottles, occlusion).
- **VLM cannot weigh bottles** — weight estimate is count × 0.03 kg (average). Real weight requires a scale.
- **Material classification** relies on visual cues (colour, shape, label text). Ambiguous bottles (black HDPE that looks like PET) will trigger human review.
- **No real-time video** — we sample frames, not a live stream. A conveyor belt system would need a different architecture.
- **Pricing is static** — $/kg values are hardcoded estimates. Production would pull from live commodity pricing APIs.
