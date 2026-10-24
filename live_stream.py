"""
Live WebRTC Video Stream Processor for VisionAI.

Handles real-time webcam frame ingestion, YOLOv8 inference,
and streaming annotated frames back to the client browser via WebRTC.
"""

import threading
import time
from typing import Optional

import av
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase

from detector import ObjectDetector

# Public STUN servers for NAT traversal (essential for cloud deployment)
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }
)


class LiveVideoProcessor(VideoProcessorBase):
    """
    Processes incoming webcam frames through YOLOv8 in real time.
    Thread-safe attributes allow Streamlit UI to display live stats.
    """

    def __init__(self) -> None:
        self.detector: Optional[ObjectDetector] = None
        self.confidence: float = 0.30
        self.iou_threshold: float = 0.45
        self.selected_classes: Optional[list[str]] = None

        # Thread synchronization
        self._lock = threading.Lock()
        self.latest_count: int = 0
        self.latest_classes: list[str] = []
        self.fps: float = 0.0
        self._last_time: float = time.time()
        self._frame_count: int = 0

    def update_params(
        self,
        detector: ObjectDetector,
        confidence: float,
        iou_threshold: float,
        selected_classes: Optional[list[str]],
    ) -> None:
        """Dynamically update detection hyperparameters from sidebar."""
        self.detector = detector
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.selected_classes = selected_classes

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Callback invoked for every incoming video frame.
        Converts av.VideoFrame -> ndarray -> YOLOv8 predict -> av.VideoFrame.
        """
        img = frame.to_ndarray(format="bgr24")

        # Calculate live streaming FPS
        now = time.time()
        self._frame_count += 1
        elapsed = now - self._last_time
        if elapsed >= 1.0:
            self.fps = round(self._frame_count / elapsed, 1)
            self._frame_count = 0
            self._last_time = now

        if self.detector is not None:
            annotated_frame, detected_names = self.detector.predict_frame(
                img,
                confidence=self.confidence,
                iou_threshold=self.iou_threshold,
                selected_classes=self.selected_classes,
            )
            with self._lock:
                self.latest_count = len(detected_names)
                self.latest_classes = detected_names

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

        return frame

    def get_stats(self) -> dict:
        """Safely fetch the latest detection statistics."""
        with self._lock:
            return {
                "count": self.latest_count,
                "classes": list(self.latest_classes),
                "fps": self.fps,
            }
