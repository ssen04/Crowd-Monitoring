# Hybrid Vision Models for Crowd Disaster Prevention

**Integrating YOLOv5 and CSRNet with Real-Time Surge Detection and Dynamic Zoom**

This repository contains the complete implementation of a **hybrid computer vision system for real-time crowd disaster prevention**. The system dynamically combines **YOLOv5 object detection** and **CSRNet density estimation**, automatically adapting to changing crowd densities while providing predictive surge detection, dynamic zoom into high-risk regions, and a six-panel situational awareness dashboard.

This project was developed as part of **CMPT 742 – Visual Computing II** at **Simon Fraser University**.

---

## Motivation

Crowd disasters such as stampedes and crowd crushes remain one of the most preventable causes of mass casualties. Traditional CCTV systems are passive and rely entirely on delayed human response. This project addresses that gap by providing:

* Automated, real-time crowd analysis
* Predictive detection of dangerous crowd surges
* Adaptive monitoring across sparse to extremely dense crowds
* Visual decision support for rapid emergency response

---

## Key Features

* **Hybrid Vision Pipeline**

  * YOLOv5 for sparse crowds (accurate individual detection)
  * CSRNet for dense crowds (robust density estimation)

* **Automatic Model Switching**

  * Empirically derived switching threshold at **35 people per frame**

* **Temporal Surge Detection**

  * Detects rapid increases in crowd density (early warning signal)

* **Dynamic Zoom**

  * Automatically zooms into the most dangerous, fast-changing crowd regions

* **Six-Panel Real-Time Dashboard**

  * YOLO detections
  * CSRNet density heatmap
  * Temporal surge map
  * Dynamic zoom view
  * Original video feed
  * Live system statistics and alerts

* **Emergency Routing (In Development)**

  * Computes safest paths for responders using density + surge risk fields

---

## System Overview

The system operates in real time and adapts to crowd conditions using the following logic:

| Crowd Density | Active Model | Reason                        |
| ------------- | ------------ | ----------------------------- |
| < 35 people   | YOLOv5       | Accurate individual detection |
| ≥ 35 people   | CSRNet       | Robust under heavy occlusion  |

Both models run continuously, but only the optimal output is used at any time.

---

## Benchmark Highlights

Experiments were conducted on the **ShanghaiTech dataset**.

| Scenario          | YOLOv5 MAE | CSRNet MAE | Winner                |
| ----------------- | ---------- | ---------- | --------------------- |
| Very Sparse (<15) | **2.75**   | 6.50       | YOLOv5 (2.36× better) |
| Sparse (<50)      | 18.78      | **5.40**   | CSRNet                |
| Moderate (<100)   | 42.2       | **9.71**   | CSRNet (3.9× better)  |

This validates the necessity of a **hybrid adaptive system**.

---

## Architecture

```
Video Input
   │
   ├── YOLOv5 Detector ──┐
   │                     ├── Adaptive Switching ──► Output
   ├── CSRNet Density ───┘
   │
   ├── Temporal Surge Detection
   │
   ├── Dynamic Zoom Controller
   │
   └── 6-Panel Visualization Dashboard
```

---

## Repository Structure

```
.
├── src/
│   ├── yolo_detection.py        # YOLOv5 person detection
│   ├── csrnet_density.py        # CSRNet density estimation
│   ├── temporal_surge.py        # Frame-to-frame surge detection
│   ├── dynamic_zoom.py          # Automatic zoom into hotspots
│   ├── emergency_routing.py     # (In development)
│   └── main.py                  # System entry point
│
├── models/
│   ├── yolov5s.pt
│   └── csrnet_shanghaitech.pth
│
├── data/
│   └── sample_videos/
│
├── results/
│   └── visualizations/
│
├── requirements.txt
└── README.md
```

## Future Work

* Emergency routing optimization
* Stability scoring for surge prediction
* Fully autonomous safety agent
* Multi-camera fusion
* Edge-device deployment

---

## Project Resources

* **Final Report**: Included in this repository
* **Demo Videos & Visual Results**: [https://saswatisen.my.canva.site/crowd-disaster-prevention](https://saswatisen.my.canva.site/crowd-disaster-prevention)

---

## Acknowledgments

This project was completed as part of **CMPT 742 – Visual Computing II** at **Simon Fraser University**.

Special thanks to:

* **Prof. Ali Mahdavi-Amiri**
* **Amir Alimohammadi**
* **Sai Raj Kishore Perla**

for their guidance and feedback.

---

## License

This project is intended for **academic and research use**. Please cite appropriately if used in publications.

---

## Author

**Sukanya Sen**
Simon Fraser University
📧 [sukanya.sen@sfu.ca](mailto:sukanya.sen@sfu.ca)
