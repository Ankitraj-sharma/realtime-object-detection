"""
VisionAI — Real-Time Object Detection App
==========================================
A production-grade Streamlit application for object detection using
YOLOv8 (Ultralytics). Supports image upload, video processing, and
live webcam snapshot detection with interactive analytics.

Author : VisionAI Team
Stack  : Streamlit · YOLOv8 · OpenCV · Plotly
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from detector import MODEL_OPTIONS, ObjectDetector
from utils import (
    class_distribution_chart,
    confidence_by_class_chart,
    confidence_distribution_chart,
    cv2_to_pil,
    detections_to_dataframe,
    image_to_bytes,
    pil_to_cv2,
    resize_for_display,
    summary_metrics,
)

# ═══════════════════════════════════════════════════════════════════════
#  Page configuration
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VisionAI · Real-Time Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
#  Custom CSS for a polished UI
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    /* ── Header banner ─────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(78, 205, 196, 0.25);
    }
    .main-header h1 {
        color: #4ECDC4;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
    }
    .main-header p {
        color: #b0b0b0;
        font-size: 1.05rem;
        margin: 0.4rem 0 0 0;
    }

    /* ── KPI metric cards ──────────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(78, 205, 196, 0.15);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(78, 205, 196, 0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4ECDC4;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.25rem;
    }

    /* ── Tab styling ───────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }

    /* ── Upload area ───────────────────────────────────────────────── */
    .upload-area {
        border: 2px dashed rgba(78, 205, 196, 0.3);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* ── Footer ────────────────────────────────────────────────────── */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #666;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255,255,255,0.05);
        margin-top: 3rem;
    }

    /* Hide Streamlit boilerplate */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  Session state initialisation
# ═══════════════════════════════════════════════════════════════════════
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []
if "total_images_processed" not in st.session_state:
    st.session_state.total_images_processed = 0

# ═══════════════════════════════════════════════════════════════════════
#  Header
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 VisionAI — Real-Time Object Detection</h1>
        <p>Detect, analyse and visualise objects in images, videos and live webcam
        feeds powered by state-of-the-art YOLOv8 deep learning models.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════
#  Sidebar — model & detection controls
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.markdown("---")

    # Model selection
    model_key = st.selectbox(
        "🧠 Model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Nano is fastest, Medium is most accurate.",
    )

    st.markdown("---")

    # Thresholds
    confidence = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Minimum confidence to keep a detection.",
    )

    iou_threshold = st.slider(
        "📐 IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Overlap threshold for Non-Maximum Suppression.",
    )

    st.markdown("---")

    # Class filtering — populated lazily after detector loads
    st.markdown("### 🏷️ Class Filter")
    filter_classes = st.checkbox("Filter specific classes", value=False)
    selected_classes = None

    if filter_classes:
        # Instantiate detector briefly to list classes
        _det = ObjectDetector(model_key)
        all_classes = _det.get_all_class_names()
        selected_classes = st.multiselect(
            "Select classes to detect",
            options=all_classes,
            default=all_classes[:5],
        )
        if not selected_classes:
            selected_classes = None  # detect all if nothing selected

    st.markdown("---")

    # Video processing options
    st.markdown("### 🎬 Video Settings")
    frame_skip = st.slider(
        "Process every N-th frame",
        min_value=1,
        max_value=10,
        value=3,
        help="Skip frames to speed up video processing.",
    )

    st.markdown("---")

    # Session stats
    st.markdown("### 📈 Session Stats")
    st.metric("Images Processed", st.session_state.total_images_processed)
    st.metric(
        "Total Detections",
        sum(len(h["detections"]) for h in st.session_state.detection_history),
    )

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.detection_history = []
        st.session_state.total_images_processed = 0
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════
#  Load detector (cached)
# ═══════════════════════════════════════════════════════════════════════
detector = ObjectDetector(model_key)

