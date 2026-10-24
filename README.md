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
| 🔴 **Live Webcam Stream** | **Zero upload needed!** Just click START — continuous real-time detection via WebRTC |
| 📷 **Image Detection** | Upload any image and instantly detect 80+ object classes |
| 🎥 **Video Processing** | Frame-by-frame detection on uploaded videos with progress tracking |
| 📸 **Camera Snapshot** | Snap single photos from webcam for detailed offline analysis |
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
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/realtime-object-detection.git
cd realtime-object-detection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🧠 Model Details

This application uses **YOLOv8** (You Only Look Once, version 8) by [Ultralytics](https://docs.ultralytics.com/) — a state-of-the-art, real-time object detection model.

| Model | Parameters | mAP@50 (COCO) | CPU Inference | Use Case |
|-------|-----------|----------------|---------------|----------|
| **YOLOv8n** (Nano) | 3.2M | 37.3% | ~80 ms | Real-time, resource-constrained |
| **YOLOv8s** (Small) | 11.2M | 44.9% | ~130 ms | Balanced speed/accuracy |
| **YOLOv8m** (Medium) | 25.9M | 50.2% | ~230 ms | Higher accuracy |

### Detectable Classes (COCO)

The model detects **80 object classes** across categories:

- 🧑 **People** — person
- 🚗 **Vehicles** — car, truck, bus, bicycle, motorcycle, airplane, boat, train
- 🐾 **Animals** — cat, dog, horse, bird, cow, sheep, elephant, bear, zebra, giraffe
- 🍕 **Food** — banana, apple, sandwich, orange, pizza, cake, donut
- 🪑 **Furniture** — chair, couch, bed, dining table
- 📱 **Electronics** — laptop, TV, cell phone, keyboard, mouse
- ⚽ **Sports** — sports ball, tennis racket, baseball bat, skateboard

---

## 📂 Project Structure

```
realtime-object-detection/
├── app.py                  # Main Streamlit application (UI + routing)
├── detector.py             # YOLOv8 detection engine (model loading + inference)
├── live_stream.py          # Real-time WebRTC continuous video streaming engine
├── utils.py                # Visualization helpers, charts, data conversion
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml         # Streamlit theme & server configuration
├── .gitignore
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## 🎛️ Configuration

All parameters are configurable via the **sidebar** at runtime:

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Model | Nano / Small / Medium | Nano | Detection model variant |
| Confidence | 0.0 – 1.0 | 0.30 | Minimum detection confidence |
| IoU Threshold | 0.0 – 1.0 | 0.45 | NMS overlap threshold |
| Frame Skip | 1 – 10 | 3 | Process every N-th frame (video) |
| Class Filter | Multi-select | All | Restrict to specific classes |

---

## 🌐 Deployment

### Streamlit Cloud (Recommended)

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Select the repository and set `app.py` as the main file
4. Click **Deploy** — Streamlit Cloud handles everything automatically

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Frontend** | [Streamlit](https://streamlit.io/) | ≥ 1.31 |
| **Detection** | [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/) | ≥ 8.1 |
| **Computer Vision** | [OpenCV](https://opencv.org/) | ≥ 4.8 |
| **Visualization** | [Plotly](https://plotly.com/) | ≥ 5.18 |
| **Data** | [Pandas](https://pandas.pydata.org/) | ≥ 2.0 |
| **Language** | Python | 3.10+ |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements

- [Ultralytics](https://docs.ultralytics.com/) for the YOLOv8 model family
- [COCO Dataset](https://cocodataset.org/) for the training data and class definitions
- [Streamlit](https://streamlit.io/) for the amazing web app framework

---

<div align="center">
  <b>Built with ❤️ using Streamlit & YOLOv8</b>
</div>
