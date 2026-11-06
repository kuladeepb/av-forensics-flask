# Deployment Guide for AV-Forensics 🚀

This full-stack Flask application is pre-configured for containerized cloud deployment with automated SQLite database provisioning, production Gunicorn server, and headless OpenCV/MediaPipe support.

---

## ⚡ Deployment Options

### Option 1: Hugging Face Spaces (Recommended — Free 16GB RAM)
Hugging Face Spaces is the ideal platform for AI/ML deep learning applications because it provides **free 16GB RAM instances with CPU or GPU**:

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Set **Space SDK** to **Docker** (Blank).
3. Connect your GitHub repository (`kuladeepb/av-forensics-flask`).
4. In your Space's **Settings &rarr; Variables and secrets**:
   - Add `MODEL_WEIGHTS_URL`: Direct download URL to your `model.pth` (or upload `model.pth` directly into the Space repository using Git LFS).
5. The container will build automatically using the included `Dockerfile` and go live!

---

### Option 2: Render or Railway

1. Create a new **Web Service** on [Render](https://render.com) or [Railway](https://railway.app).
2. Connect your GitHub repository: `https://github.com/kuladeepb/av-forensics-flask`.
3. Set the build type to **Docker**.
4. **Environment Variables:**
   - `PORT`: `5000` (or leave default assigned by platform)
   - `MODEL_WEIGHTS_URL`: URL to download `model.pth`
   - `FLASK_SECRET_KEY`: A random secure string
5. **Persistent Storage (Recommended):**
   - Add a persistent disk mounted to `/app/data` (e.g. 5GB size) to preserve the SQLite audit database (`forensics.db`) and PDF reports across deploys.

---

### Option 3: Any Linux VPS (Ubuntu / Debian / AWS EC2 / DigitalOcean)

#### Using Docker:
```bash
# Clone the repository
git clone https://github.com/kuladeepb/av-forensics-flask.git
cd av-forensics-flask

# Copy your local model.pth into the project directory
# (or set MODEL_WEIGHTS_URL in your docker run command)

# Build and run container
docker build -t av-forensics .
docker run -d -p 5000:5000 -v $(pwd)/data:/app/data --name av-forensics-app av-forensics
```

#### Running Directly without Docker:
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1 libglib2.0-0

git clone https://github.com/kuladeepb/av-forensics-flask.git
cd av-forensics-flask

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start with Gunicorn production server
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 180 app:app
```

---

## 🔑 Default Administrator Credentials
On first launch, the database automatically initializes with:
- **Username:** `admin`
- **Password:** `admin123`
*(Remember to change or promote new accounts inside the Admin Control Center).*
