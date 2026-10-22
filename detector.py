"""
Object Detection Engine powered by YOLOv8.

Provides a high-level interface for running object detection inference
on images using Ultralytics YOLOv8 models with configurable parameters.
"""

import numpy as np
import streamlit as st
from ultralytics import YOLO
from typing import Optional


# ── Available model variants ───────────────────────────────────────────
MODEL_OPTIONS: dict[str, str] = {
    "YOLOv8 Nano  (Fastest)": "yolov8n.pt",
    "YOLOv8 Small (Balanced)": "yolov8s.pt",
    "YOLOv8 Medium (Accurate)": "yolov8m.pt",
}


@st.cache_resource(show_spinner="Loading detection model …")
def load_model(model_path: str) -> YOLO:
    """Download (if needed) and cache a YOLOv8 model in memory."""
    return YOLO(model_path)

