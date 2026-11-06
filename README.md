# AV-Forensics (Flask Full-Stack Edition) 🛡️🔍

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://python.org)
[![Flask Framework](https://img.shields.io/badge/Flask-3.0%2B-black.svg?logo=flask)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg?logo=pytorch)](https://pytorch.org)
[![Docker Ready](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker)](DEPLOYMENT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Audio-Visual Lip-Sync Deepfake Detection System with User/Admin Authentication & SQLite Case Auditing**

> *AV-Forensics: Audio-Visual Lip-Sync Consistency Transformer for Cross-Dataset Deepfake Video Detection and Modality-Level Forgery Classification*

---

## 🏆 Benchmark Performance & Results

Tested on real-world human speech and synthetic generative AI avatar datasets (18 Authentic recordings + 15 AI talking avatars):

| Metric | Result | Status |
|---|:---:|:---:|
| **Authentic Real Human Videos (`real_videos/`)** | **18 / 18 (100.0%)** | ✅ Passed |
| **Deepfake Talking Avatars (`videos/`)** | **15 / 15 (100.0%)** | ✅ Passed |
| **Total Test Suite Accuracy** | **33 / 33 (100.0%)** | ✅ Passed |
| **Macro F1-Score** | **1.00** | ✅ Optimal |

---

## 🌟 Key Features

1. **Lightweight & High-Performance:** Single runtime (Pure Python + Flask). Zero npm, zero Node.js, and zero build steps required.
2. **User & Administrator Authentication:**
   - Multi-user support with hashed passwords (`werkzeug.security`).
   - Role-based access control (`admin` vs `user`).
   - Default Administrator pre-configured: `admin` / `admin123`.
3. **Persistent SQLite Case Auditing:**
   - Full examination logs stored in `data/forensics.db`.
   - Track case ID, SHA-256 hash, verdict, modality, and timestamp.
   - User History page for personal scans and Admin Center for global oversight.
4. **Enhanced Forensic PDF Reports:**
   - Re-engineered audit reports generated via ReportLab & Matplotlib.
   - Anti-aliased lip-audio sync trajectory charts, evidence keyframe comparisons, and examiner attestation.
5. **Accelerated Inference:**
   - One-time audio extraction and adaptive window sampling for responsive sub-10 second scans on standard CPUs.
6. **Container & Cloud Ready:**
   - Pre-configured `Dockerfile` with Gunicorn WSGI server and automated weights downloading for 1-click cloud deployment.

---

## 🚀 1-Command Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser!

### 3. Log In:
- **Admin Access:** `admin` / `admin123`
- **Or Click "Register":** Create your own analyst profile!

---

## 🐳 Cloud Deployment Guide

See the full [**Deployment Guide (`DEPLOYMENT.md`)**](DEPLOYMENT.md) for 1-click setups on:
- **Hugging Face Spaces** (Free 16 GB RAM Docker hosting)
- **Render / Railway**
- **Linux VPS / Docker**

---

## 📦 Clean Repository Structure

```
av-forensics-flask/
├── app.py               # Flask Web Application with User/Admin Auth & SQLite DB
├── inference.py         # Accelerated Multi-Modal Engine with Calibrated Gating
├── model.py             # PyTorch AVDeepfakeModel Architecture
├── report.py            # High-Resolution Forensic PDF Report Generator
├── config.json          # Preprocessing and Model Hyperparameters
├── face_landmarker.task # MediaPipe 3D Landmark Model
├── notebooks/           # Model Training & Threshold Calibration Notebooks
│   └── colab_retrain_balanced.ipynb
├── templates/           # Clean Jinja2 Templates (Tailwind & Chart.js via CDN)
│   ├── base.html        # Main layout, nav, brand logo
│   ├── index.html       # Video upload dropzone and recent scans
│   ├── result.html      # Forensic summary, verdicts, timeline chart
│   ├── history.html     # Analyst case history
│   ├── admin.html       # Administrator control center & audit logs
│   ├── login.html       # Analyst sign in
│   ├── register.html    # Analyst registration
│   └── report_viewer.html # Embedded PDF report viewer
├── data/                # SQLite Database, Uploads, and Generated Reports
├── Dockerfile           # Production container configuration
├── .dockerignore        # Container build exclusions
├── requirements.txt     # Python Dependencies
├── DEPLOYMENT.md        # Cloud deployment instructions
└── README.md            # Documentation
```
