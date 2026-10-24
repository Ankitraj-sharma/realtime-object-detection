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
        iou_threshold: float = 0.45,
        selected_classes: Optional[list[str]] = None,
    ) -> dict:
        """
        Run detection on a single image.

        Parameters
        ----------
        image : np.ndarray
            BGR or RGB image array.
        confidence : float
            Minimum confidence threshold (0 – 1).
        iou_threshold : float
            IoU threshold for Non-Maximum Suppression.
        selected_classes : list[str] | None
            If provided, only detect these class names.

        Returns
        -------
        dict with keys:
            annotated_image – RGB image with drawn boxes/labels
            detections      – list of per-box dicts
            count           – total number of detections
            inference_ms    – model inference time in milliseconds
        """

        # Map selected class names → model indices
        class_indices = None
        if selected_classes:
            class_indices = [
                idx
                for idx, name in self.class_names.items()
                if name in selected_classes
            ]
            if not class_indices:
                class_indices = None

        results = self.model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            classes=class_indices,
            verbose=False,
        )

        result = results[0]

        # Parse individual detections
        detections: list[dict] = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(c, 1) for c in box.xyxy[0].cpu().numpy().tolist()],
                }
            )

        # Annotated image (RGB)
        annotated_image = result.plot()

        return {
            "annotated_image": annotated_image,
            "detections": detections,
            "count": len(detections),
            "inference_ms": round(result.speed.get("inference", 0), 1),
        }

    def predict_frame(
        self,
        frame: np.ndarray,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        selected_classes: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Optimized single-frame inference for continuous live WebRTC streaming.

        Returns
        -------
        tuple[np.ndarray, list[str]]:
            Annotated BGR frame and list of detected class names.
        """
        class_indices = None
        if selected_classes:
            class_indices = [
                idx
                for idx, name in self.class_names.items()
                if name in selected_classes
            ]
            if not class_indices:
                class_indices = None

        results = self.model.predict(
            source=frame,
            conf=confidence,
            iou=iou_threshold,
            classes=class_indices,
            verbose=False,
        )
        res = results[0]
        detected_names = [self.class_names[int(box.cls[0])] for box in res.boxes]
        return res.plot(), detected_names

    # ── Helpers ────────────────────────────────────────────────────────
    def get_all_class_names(self) -> list[str]:
        """Return sorted list of all COCO class names the model recognizes."""
        return sorted(self.class_names.values())
