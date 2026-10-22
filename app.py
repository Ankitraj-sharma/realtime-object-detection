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


# ═══════════════════════════════════════════════════════════════════════
#  Helper: render detection results
# ═══════════════════════════════════════════════════════════════════════
def render_results(
    original: np.ndarray,
    result: dict,
    *,
    show_side_by_side: bool = True,
    key_prefix: str = "img",
) -> None:
    """Display annotated image, metrics, table and download button."""
    detections = result["detections"]
    metrics = summary_metrics(detections)

    # ── KPI row ────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics["total"]}</div>'
            f'<div class="metric-label">Objects Detected</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics["unique_classes"]}</div>'
            f'<div class="metric-label">Unique Classes</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{metrics["avg_confidence"]}%</div>'
            f'<div class="metric-label">Avg Confidence</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{result["inference_ms"]} ms</div>'
            f'<div class="metric-label">Inference Time</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Images ─────────────────────────────────────────────────────
    if show_side_by_side:
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("#### 📷 Original")
            st.image(original, channels="BGR", use_container_width=True)
        with col_det:
            st.markdown("#### 🎯 Detected")
            st.image(
                result["annotated_image"],
                channels="BGR",
                use_container_width=True,
            )
    else:
        st.image(
            result["annotated_image"],
            channels="BGR",
            use_container_width=True,
            caption="Detection Result",
        )

    # ── Download button ────────────────────────────────────────────
    annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
    st.download_button(
        label="⬇️ Download Annotated Image",
        data=image_to_bytes(annotated_rgb, "PNG"),
        file_name=f"visionai_detection_{int(time.time())}.png",
        mime="image/png",
        key=f"{key_prefix}_download",
        use_container_width=True,
    )

    # ── Detections table ───────────────────────────────────────────
    if detections:
        with st.expander("📋 Detection Details", expanded=True):
            st.dataframe(
                detections_to_dataframe(detections),
                use_container_width=True,
                hide_index=True,
            )

        # ── Inline charts ──────────────────────────────────────────
        ch1, ch2 = st.columns(2)
        with ch1:
            st.plotly_chart(
                class_distribution_chart(detections),
                use_container_width=True,
                key=f"{key_prefix}_class_dist",
            )
        with ch2:
            st.plotly_chart(
                confidence_distribution_chart(detections),
                use_container_width=True,
                key=f"{key_prefix}_conf_dist",
            )
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")


# ═══════════════════════════════════════════════════════════════════════
#  Tabs
# ═══════════════════════════════════════════════════════════════════════
tab_img, tab_vid, tab_cam, tab_analytics, tab_about = st.tabs(
    ["📷 Image Detection", "🎥 Video Detection", "📹 Webcam", "📊 Analytics", "ℹ️ About"]
)

# ── TAB 1 : Image Detection ──────────────────────────────────────────
with tab_img:
    st.markdown("### Upload an image to detect objects")

    uploaded_file = st.file_uploader(
        "Choose an image …",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
        help="Supported: JPG, PNG, BMP, WebP  ·  Max 50 MB",
    )

    if uploaded_file is not None:
        # Read image
        pil_img = Image.open(uploaded_file).convert("RGB")
        cv2_img = pil_to_cv2(pil_img)

        # Run detection
        with st.spinner("🔍 Running object detection …"):
            start = time.perf_counter()
            result = detector.detect(
                cv2_img,
                confidence=confidence,
                iou_threshold=iou_threshold,
                selected_classes=selected_classes,
            )
            wall_time = round((time.perf_counter() - start) * 1000, 1)

        # Record in session history
        st.session_state.total_images_processed += 1
        st.session_state.detection_history.append(
            {
                "source": "image",
                "detections": result["detections"],
                "inference_ms": result["inference_ms"],
                "timestamp": time.time(),
            }
        )

        render_results(cv2_img, result, key_prefix="img")

    else:
        st.markdown(
            """
            <div class="upload-area">
                <p style="font-size:3rem; margin:0;">📷</p>
                <p style="color:#888;">Drag & drop an image here or click <b>Browse files</b></p>
                <p style="color:#555; font-size:0.85rem;">
                    The model can detect <b>80 object classes</b> from the COCO dataset —
                    people, vehicles, animals, furniture, food and more.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── TAB 2 : Video Detection ──────────────────────────────────────────
with tab_vid:
    st.markdown("### Upload a video for frame-by-frame detection")

    uploaded_video = st.file_uploader(
        "Choose a video …",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
        help="Supported: MP4, AVI, MOV, MKV  ·  Max 50 MB",
    )

    if uploaded_video is not None:
        # Save uploaded video to a temp file (OpenCV needs a path)
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        st.info(
            f"📹 **{total_frames}** frames  ·  **{fps}** FPS  ·  processing every **{frame_skip}** frame(s)"
        )

        # UI containers
        progress_bar = st.progress(0, text="Processing video …")
        frame_display = st.empty()
        stats_display = st.empty()

        all_detections: list[dict] = []
        frame_idx = 0
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                result = detector.detect(
                    frame,
                    confidence=confidence,
                    iou_threshold=iou_threshold,
                    selected_classes=selected_classes,
                )
                all_detections.extend(result["detections"])
                processed += 1

                # Update display every few processed frames
                if processed % 3 == 0 or frame_idx + frame_skip >= total_frames:
                    frame_display.image(
                        result["annotated_image"],
                        channels="BGR",
                        use_container_width=True,
                        caption=f"Frame {frame_idx + 1}/{total_frames}",
                    )

                progress_pct = min((frame_idx + 1) / max(total_frames, 1), 1.0)
                progress_bar.progress(
                    progress_pct,
                    text=f"Frame {frame_idx + 1}/{total_frames}  ·  {len(all_detections)} detections so far",
                )

            frame_idx += 1

        cap.release()
        progress_bar.progress(1.0, text="✅ Video processing complete!")

        # Save to history
        st.session_state.detection_history.append(
            {
                "source": "video",
                "detections": all_detections,
                "frames_processed": processed,
                "timestamp": time.time(),
            }
        )

        # Show video summary
        st.markdown("---")
        st.markdown("### 📊 Video Detection Summary")
        vm = summary_metrics(all_detections)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Detections", vm["total"])
        c2.metric("Unique Classes", vm["unique_classes"])
        c3.metric("Avg Confidence", f'{vm["avg_confidence"]}%')
        c4.metric("Frames Processed", processed)

        if all_detections:
            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(
                    class_distribution_chart(all_detections),
                    use_container_width=True,
                    key="vid_class_dist",
                )
            with v2:
                st.plotly_chart(
                    confidence_distribution_chart(all_detections),
                    use_container_width=True,
                    key="vid_conf_dist",
                )
            with st.expander("📋 All Video Detections"):
                st.dataframe(
                    detections_to_dataframe(all_detections),
                    use_container_width=True,
                    hide_index=True,
                )

    else:
        st.markdown(
            """
            <div class="upload-area">
                <p style="font-size:3rem; margin:0;">🎥</p>
                <p style="color:#888;">Drag & drop a video here or click <b>Browse files</b></p>
                <p style="color:#555; font-size:0.85rem;">
                    Each frame is processed through the detection model.
                    Use the <b>frame skip</b> slider in the sidebar to speed up processing.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── TAB 3 : Webcam ───────────────────────────────────────────────────
with tab_cam:
    st.markdown("### Capture a snapshot from your webcam")
    st.caption(
        "Click the camera button to take a photo. Detection runs automatically on each capture."
    )

    camera_image = st.camera_input("📸 Take a photo", key="webcam_input")

    if camera_image is not None:
        pil_img = Image.open(camera_image).convert("RGB")
        cv2_img = pil_to_cv2(pil_img)

        with st.spinner("🔍 Detecting …"):
            result = detector.detect(
                cv2_img,
                confidence=confidence,
                iou_threshold=iou_threshold,
                selected_classes=selected_classes,
            )

        st.session_state.total_images_processed += 1
        st.session_state.detection_history.append(
            {
                "source": "webcam",
                "detections": result["detections"],
                "inference_ms": result["inference_ms"],
                "timestamp": time.time(),
            }
        )

        render_results(cv2_img, result, key_prefix="cam")

# ── TAB 4 : Analytics ────────────────────────────────────────────────
with tab_analytics:
    st.markdown("### 📊 Aggregated Detection Analytics")
    st.caption("Cumulative statistics from all detections in this session.")

    history = st.session_state.detection_history
    if not history:
        st.info(
            "No detections yet. Upload an image, video or use the webcam to get started!"
        )
    else:
        # Merge all detections
        all_dets = []
        for h in history:
            all_dets.extend(h["detections"])

        m = summary_metrics(all_dets)

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Objects", m["total"])
        k2.metric("Unique Classes", m["unique_classes"])
        k3.metric("Avg Confidence", f'{m["avg_confidence"]}%')
        k4.metric("Max Confidence", f'{m["max_confidence"]}%')
        k5.metric("Min Confidence", f'{m["min_confidence"]}%')

        st.markdown("---")

        # Charts
        a1, a2 = st.columns(2)
        with a1:
            st.plotly_chart(
                class_distribution_chart(all_dets),
                use_container_width=True,
                key="analytics_class",
            )
        with a2:
            st.plotly_chart(
                confidence_distribution_chart(all_dets),
                use_container_width=True,
                key="analytics_conf",
            )

        if len(set(d["class_name"] for d in all_dets)) > 1:
            st.plotly_chart(
                confidence_by_class_chart(all_dets),
                use_container_width=True,
                key="analytics_box",
            )

        # Full table
        with st.expander("📋 All Detections (raw)"):
            st.dataframe(
                detections_to_dataframe(all_dets),
                use_container_width=True,
                hide_index=True,
            )

# ── TAB 5 : About ────────────────────────────────────────────────────
with tab_about:
    st.markdown(
        """
        ### About VisionAI

        **VisionAI** is a real-time object detection web application built for 
        analysing images, videos and live webcam feeds. It leverages 
        **YOLOv8** — the latest generation of the YOLO (You Only Look Once) 
        family of models — which achieve an excellent balance between speed 
        and accuracy.

        ---

        #### 🏗️ Architecture

        ```
        ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
        │  Streamlit    │────▶│  Detector    │────▶│  YOLOv8 Model    │
        │  Frontend     │◀────│  Engine      │◀────│  (Ultralytics)   │
        └──────────────┘     └──────────────┘     └──────────────────┘
              │                                           │
              ▼                                           ▼
        ┌──────────────┐                          ┌──────────────────┐
        │  Plotly       │                          │  COCO Dataset    │
        │  Analytics    │                          │  (80 classes)    │
        └──────────────┘                          └──────────────────┘
        ```

        ---

        #### 🧠 Model Details

        | Model | Parameters | mAP@50 | Speed (CPU) |
        |-------|-----------|--------|-------------|
        | YOLOv8n (Nano) | 3.2M | 37.3 | ~80 ms |
        | YOLOv8s (Small) | 11.2M | 44.9 | ~130 ms |
        | YOLOv8m (Medium) | 25.9M | 50.2 | ~230 ms |

        ---

        #### 🎯 Detectable Object Classes (COCO)

        The model can detect **80 object classes** including:

        🚗 **Vehicles** — car, truck, bus, bicycle, motorcycle, airplane, boat, train  
        🧑 **People** — person  
        🐾 **Animals** — cat, dog, horse, bird, cow, sheep, elephant, bear, zebra, giraffe  
        🍕 **Food** — banana, apple, sandwich, orange, pizza, cake, donut, hot dog  
        🪑 **Furniture** — chair, couch, bed, dining table, toilet  
        📱 **Electronics** — laptop, TV, cell phone, keyboard, mouse, remote  
        ⚽ **Sports** — sports ball, tennis racket, baseball bat, skateboard, surfboard, ski  

        ---

        #### 🛠️ Tech Stack

        | Component | Technology |
        |-----------|-----------|
        | Frontend | Streamlit |
        | Detection Model | YOLOv8 (Ultralytics) |
        | Computer Vision | OpenCV |
        | Analytics | Plotly · Pandas |
        | Deployment | Streamlit Cloud |
        | Language | Python 3.10+ |

        ---

        #### 📚 References

        - [Ultralytics YOLOv8](https://docs.ultralytics.com/)
        - [COCO Dataset](https://cocodataset.org/)
        - [Streamlit Documentation](https://docs.streamlit.io/)

        ---

        <div style="text-align:center; color:#666; padding:1rem;">
            Built with ❤️ using Streamlit & YOLOv8
        </div>
        """,
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════
#  Footer
# ═══════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="footer">
        <b>VisionAI</b> · Real-Time Object Detection · 
        Powered by <a href="https://docs.ultralytics.com/" target="_blank">YOLOv8</a> & 
        <a href="https://streamlit.io/" target="_blank">Streamlit</a> ·
        © 2026
    </div>
    """,
    unsafe_allow_html=True,
)
