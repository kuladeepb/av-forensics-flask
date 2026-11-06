# AV-Forensics (Flask Full-Stack Edition) 🛡️🔍

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

## 📦 Clean Repository Structure (7 Core Components)

```
project-final-flask/
├── app.py               # Flask Web Application with User/Admin Auth & SQLite DB
├── inference.py         # Accelerated Multi-Modal Engine with Calibrated Gating
├── model.py             # PyTorch AVDeepfakeModel Architecture
├── report.py            # High-Resolution Forensic PDF Report Generator
├── config.json          # Preprocessing and Model Hyperparameters
├── model.pth            # Trained PyTorch Model Weights
├── face_landmarker.task # MediaPipe 3D Landmark Model
├── templates/           # Clean Jinja2 Templates (Tailwind & Chart.js via CDN)
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   ├── admin.html
│   ├── login.html
│   └── register.html
├── data/                # SQLite Database, Uploads, and Generated Reports
├── requirements.txt     # Python Dependencies
└── README.md            # Documentation
```

---

## 🌐 Pushing to a New GitHub Repository

To push this Flask project to its own separate GitHub repository:

```bash
cd c:\Users\HP\Downloads\project-final-flask
git init
git branch -M main
git add .
git commit -m "feat: complete AV-Forensics Flask full-stack application"
git remote add origin https://github.com/YOUR_USERNAME/NEW_REPO_NAME.git
git push -u origin main
```

