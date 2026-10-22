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


class ObjectDetector:
    """
    Wraps a YOLOv8 model to provide a clean detection API.

    Parameters
    ----------
    model_key : str
        A human-readable key from ``MODEL_OPTIONS`` (e.g. "YOLOv8 Nano  (Fastest)").
    """

    def __init__(self, model_key: str = "YOLOv8 Nano  (Fastest)") -> None:
        model_path = MODEL_OPTIONS[model_key]
        self.model = load_model(model_path)
        self.class_names: dict[int, str] = self.model.names  # {0: 'person', …}

    # ── Core detection ─────────────────────────────────────────────────
    def detect(
        self,
        image: np.ndarray,
        confidence: float = 0.25,
