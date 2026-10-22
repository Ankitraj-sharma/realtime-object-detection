# 🔍 VisionAI — Real-Time Object Detection App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://visionai-object-detection.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-brightgreen.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade** web application for real-time object detection powered by **YOLOv8** and **Streamlit**. Detect, analyse and visualise objects in images, videos and live webcam feeds — all from your browser.

![VisionAI Demo](https://img.shields.io/badge/Status-Live-brightgreen)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 **Image Detection** | Upload any image and instantly detect 80+ object classes |
| 🎥 **Video Processing** | Frame-by-frame detection on uploaded videos with progress tracking |
| 📹 **Webcam Capture** | Snap photos from your webcam for instant detection |
| 📊 **Interactive Analytics** | Plotly-powered charts — class distribution, confidence histograms, box plots |
| 🧠 **Multi-Model Support** | Switch between YOLOv8 Nano (fast), Small (balanced), and Medium (accurate) |
| 🎯 **Configurable Thresholds** | Fine-tune confidence and IoU thresholds via sidebar controls |
| 🏷️ **Class Filtering** | Selectively detect only the object classes you care about |
| ⬇️ **Export Results** | Download annotated images with bounding boxes and labels |
| 📋 **Detection Tables** | Sortable tables with class, confidence and bounding box coordinates |
| 🌙 **Dark Theme UI** | Professional dark interface with gradient styling |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        VisionAI Application                        │
├──────────────┬──────────────┬──────────────┬──────────────────────┤
│  Image Tab   │  Video Tab   │  Webcam Tab  │  Analytics Tab       │
├──────────────┴──────────────┴──────────────┴──────────────────────┤
│                     Streamlit Frontend (app.py)                     │
├───────────────────────────────────────────────────────────────────┤
│                   Object Detector (detector.py)                    │
│                   ┌─────────────────────────┐                      │
│                   │  YOLOv8 (Ultralytics)   │                      │
│                   │  • Nano  (3.2M params)  │                      │
│                   │  • Small (11.2M params) │                      │
│                   │  • Medium(25.9M params) │                      │
│                   └─────────────────────────┘                      │
├───────────────────────────────────────────────────────────────────┤
│               Utilities & Analytics (utils.py)                     │
│               Plotly Charts · Pandas · OpenCV                      │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher