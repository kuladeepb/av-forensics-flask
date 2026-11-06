"""
AV-Forensics: Flask Full-Stack Forensic Web Application
Integrates User/Admin Authentication, SQLite Audit History, Accelerated Model Inference, and PDF Reports.
"""

from __future__ import annotations
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    session, send_file, jsonify, abort
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from inference import analyze_video
from report import generate_pdf_report

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORT_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "forensics.db"

for d in (DATA_DIR, UPLOAD_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "avforensics-cyber-lab-key-2026")
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60MB max
ALLOWED_EXTS = {".mp4", ".mov", ".avi"}

# ── Database Initialization ───────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE NOT NULL,
                user_id INTEGER,
                username TEXT,
                filename TEXT NOT NULL,
                file_hash_sha256 TEXT NOT NULL,
                duration REAL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                modality_class TEXT NOT NULL,
                modality_label TEXT NOT NULL,
                audio_fake_prob REAL,
                video_fake_prob REAL,
                sync_mean REAL,
                result_json TEXT NOT NULL,
                report_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        # Seed default Administrator if not exists
        admin_row = conn.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()
        if not admin_row:
            admin_pwd = generate_password_hash("admin123")
            conn.execute(
                "INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, ?)",
                ("admin", admin_pwd, "admin@avforensics.local", "admin")
            )
        conn.commit()

init_db()

# ── Auth Decorators ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please sign in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session or session.get("role") != "admin":
            flash("Administrative privileges required.", "danger")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated_function


# ── Routes: Authentication ─────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")

        pwd_hash = generate_password_hash(password)
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, 'user')",
                    (username, pwd_hash, email)
                )
                conn.commit()
            flash("Registration successful! You may now sign in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Please select another.", "danger")

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.", "info")
    return redirect(url_for("login"))


# ── Routes: Forensic Scanning ──────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    user_id = session["user_id"]
    with get_db() as conn:
        recent_scans = conn.execute(
            "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC LIMIT 5",
            (user_id,)
        ).fetchall()
        total_scans = conn.execute("SELECT COUNT(*) FROM analyses WHERE user_id = ?", (user_id,)).fetchone()[0]
    return render_template("index.html", recent_scans=recent_scans, total_scans=total_scans)


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    if "video" not in request.files:
        flash("No video file selected.", "danger")
        return redirect(url_for("index"))

    file = request.files["video"]
    if not file or not file.filename:
        flash("No file was uploaded.", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        flash(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTS))}", "danger")
        return redirect(url_for("index"))

    case_id = f"CASE-{uuid.uuid4().hex[:10].upper()}"
    temp_video_path = UPLOAD_DIR / f"{case_id}_{filename}"
    file.save(str(temp_video_path))

    # Compute SHA-256 hash
    hasher = hashlib.sha256()
    with open(temp_video_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    file_sha256 = hasher.hexdigest()

    try:
        # Run accelerated multi-modal inference
        result = analyze_video(str(temp_video_path), filename)
        result["case_id"] = case_id
        result["sha256"] = file_sha256

        # Generate high-resolution PDF report
        report_filename = f"{case_id}_Forensic_Report.pdf"
        report_path = REPORT_DIR / report_filename
        generate_pdf_report(result, str(report_path), examiner_name=session.get("username", "Analyst"))

        # Save to SQLite History
        with get_db() as conn:
            conn.execute("""
                INSERT INTO analyses (
                    case_id, user_id, username, filename, file_hash_sha256,
                    duration, verdict, confidence, modality_class, modality_label,
                    audio_fake_prob, video_fake_prob, sync_mean, result_json, report_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                case_id,
                session["user_id"],
                session["username"],
                filename,
                file_sha256,
                result["duration"],
                result["verdict"],
                result["confidence"],
                result["modality"]["class"],
                result["modality"]["label"],
                result["modality"]["audio_fake_prob"],
                result["modality"]["video_fake_prob"],
                float(np_mean([item["score"] for item in result["sync_timeline"]])) if result["sync_timeline"] else 0.0,
                json.dumps(result),
                str(report_path)
            ))
            conn.commit()

    except Exception as e:
        flash(f"Forensic analysis failed: {str(e)}", "danger")
        return redirect(url_for("index"))
    finally:
        # Secure cleanup: remove raw uploaded video after feature extraction
        try:
            if temp_video_path.exists():
                os.remove(str(temp_video_path))
        except Exception:
            pass

    return redirect(url_for("view_result", case_id=case_id))


def np_mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


@app.route("/result/<case_id>")
@login_required
def view_result(case_id: str):
    with get_db() as conn:
        record = conn.execute("SELECT * FROM analyses WHERE case_id = ?", (case_id,)).fetchone()

    if not record:
        flash("Case record not found.", "warning")
        return redirect(url_for("index"))

    # Only owner or admin can view
    if record["user_id"] != session["user_id"] and session.get("role") != "admin":
        flash("Access denied to requested forensic case.", "danger")
        return redirect(url_for("index"))

    result = json.loads(record["result_json"])
    return render_template("result.html", result=result, record=record)


@app.route("/view-report/<case_id>")
@login_required
def view_report_page(case_id: str):
    with get_db() as conn:
        record = conn.execute("SELECT * FROM analyses WHERE case_id = ?", (case_id,)).fetchone()

    if not record:
        abort(404)

    if record["user_id"] != session["user_id"] and session.get("role") != "admin":
        abort(403)

    report_path = record["report_path"]
    if not report_path or not os.path.exists(report_path):
        flash("Report document is not yet generated or unavailable.", "warning")
        return redirect(url_for("view_result", case_id=case_id))

    return render_template("report_viewer.html", record=record, case_id=case_id)


@app.route("/pdf-raw/<case_id>")
@login_required
def serve_pdf_raw(case_id: str):
    with get_db() as conn:
        record = conn.execute("SELECT * FROM analyses WHERE case_id = ?", (case_id,)).fetchone()

    if not record:
        abort(404)

    if record["user_id"] != session["user_id"] and session.get("role") != "admin":
        abort(403)

    report_path = record["report_path"]
    if not report_path or not os.path.exists(report_path):
        abort(404)

    return send_file(
        report_path,
        as_attachment=False,
        mimetype="application/pdf"
    )


@app.route("/download-report/<case_id>")
@login_required
def download_report(case_id: str):
    with get_db() as conn:
        record = conn.execute("SELECT * FROM analyses WHERE case_id = ?", (case_id,)).fetchone()

    if not record:
        abort(404)

    if record["user_id"] != session["user_id"] and session.get("role") != "admin":
        abort(403)

    report_path = record["report_path"]
    if not report_path or not os.path.exists(report_path):
        flash("Report document is not yet generated or unavailable.", "warning")
        return redirect(url_for("view_result", case_id=case_id))

    return send_file(
        report_path,
        as_attachment=True,
        download_name=f"{case_id}_Forensic_Report.pdf",
        mimetype="application/pdf"
    )


@app.route("/history")
@login_required
def history():
    user_id = session["user_id"]
    with get_db() as conn:
        records = conn.execute(
            "SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
    return render_template("history.html", records=records)


# ── Routes: Administration ────────────────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin():
    with get_db() as conn:
        users = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
        all_scans = conn.execute("SELECT * FROM analyses ORDER BY created_at DESC").fetchall()
        total_scans = len(all_scans)
        real_count = sum(1 for s in all_scans if s["verdict"] == "REAL")
        fake_count = sum(1 for s in all_scans if s["verdict"] == "FAKE")

    return render_template(
        "admin.html",
        users=users,
        all_scans=all_scans,
        total_scans=total_scans,
        real_count=real_count,
        fake_count=fake_count
    )


@app.route("/admin/delete-scan/<int:scan_id>", methods=["POST"])
@admin_required
def admin_delete_scan(scan_id: int):
    with get_db() as conn:
        record = conn.execute("SELECT report_path FROM analyses WHERE id = ?", (scan_id,)).fetchone()
        if record and record["report_path"] and os.path.exists(record["report_path"]):
            try:
                os.remove(record["report_path"])
            except Exception:
                pass
        conn.execute("DELETE FROM analyses WHERE id = ?", (scan_id,))
        conn.commit()
    flash("Case analysis record deleted.", "info")
    return redirect(url_for("admin"))


@app.route("/admin/toggle-role/<int:user_id>", methods=["POST"])
@admin_required
def admin_toggle_role(user_id: int):
    if user_id == session["user_id"]:
        flash("Cannot modify your own administrative role.", "warning")
        return redirect(url_for("admin"))

    with get_db() as conn:
        user = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
        if user:
            new_role = "user" if user["role"] == "admin" else "admin"
            conn.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
            conn.commit()
            flash(f"User role updated to '{new_role}'.", "success")

    return redirect(url_for("admin"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  AV-FORENSICS CYBERCRIME LABORATORY WEB DASHBOARD")
    print(f"  Running on: http://127.0.0.1:{port}")
    print(f"  Default Administrator: admin / admin123")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

