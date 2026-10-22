"""
Utility helpers for visualization, analytics, and data export.
"""

import io
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


# ── Colour palette (20 distinct colours, cycled for >20 classes) ─────
_PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#82E0AA",
    "#F8C471", "#85C1E9", "#F1948A", "#73C6B6", "#FAD7A0",
    "#AED6F1", "#D2B4DE", "#A3E4D7", "#F9E79F", "#FADBD8",
]


# ═══════════════════════════════════════════════════════════════════════
#  Data-table helpers
# ═══════════════════════════════════════════════════════════════════════
def detections_to_dataframe(detections: list[dict]) -> pd.DataFrame:
    """Convert a list of detection dicts into a tidy DataFrame."""
    if not detections:
        return pd.DataFrame(columns=["#", "Class", "Confidence (%)", "Bounding Box"])

    rows = []
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["bbox"]
        rows.append(
            {
                "#": i,
                "Class": d["class_name"],
                "Confidence (%)": f"{d['confidence'] * 100:.1f}",
                "Bounding Box": f"({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})",
            }
        )
    return pd.DataFrame(rows)



# ═══════════════════════════════════════════════════════════════════════
#  Image / video helpers
# ═══════════════════════════════════════════════════════════════════════
def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV array (BGR)."""
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV array (BGR) to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_to_bytes(image: np.ndarray, fmt: str = "PNG") -> bytes:
    """Encode an image (RGB numpy array) to bytes for download."""
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def resize_for_display(image: np.ndarray, max_width: int = 1200) -> np.ndarray:
    """Resize an image if it exceeds max_width, preserving aspect ratio."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    new_size = (max_width, int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
