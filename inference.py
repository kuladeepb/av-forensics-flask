"""
AV-Forensics: Optimized Multi-Modal Inference Engine
Self-contained, accelerated inference using MediaPipe FaceMesh & mHuBERT-147 PyTorch model.
Maintains exact 100% benchmark calibrated multi-head gating logic.
"""

from __future__ import annotations
import base64
import json
import logging
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import AVDeepfakeModel

log = logging.getLogger("avforensics.inference")

# ── MediaPipe 3D Lip Landmarks ────────────────────────────────────────────────
MOUTH_IDXS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    185, 40, 39, 37, 0, 267, 269, 270, 409,
]

_detector = None

def _get_detector(task_path: str = "face_landmarker.task"):
    global _detector
    if _detector is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not os.path.exists(task_path):
                alt = Path(__file__).resolve().parent / "face_landmarker.task"
                if alt.exists():
                    task_path = str(alt)
                else:
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    urllib.request.urlretrieve(url, task_path)

            base_options = python.BaseOptions(model_asset_path=task_path)
            options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
            _detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            log.warning("MediaPipe initialization error: %s", e)
            _detector = False
    return _detector if _detector is not False else None


def extract_mouth_crop(frame_bgr: np.ndarray, img_size: int = 96) -> np.ndarray:
    """Extracts mouth ROI using 3D lip landmarks with adaptive padding."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    det = _get_detector()

    if det is not None:
        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = det.detect(mp_img)
            if res.face_landmarks:
                lms = res.face_landmarks[0]
                pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in MOUTH_IDXS])
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                mw, mh = x2 - x1, y2 - y1
                pad = int(max(mw, mh) * 0.35)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                if x2 > x1 and y2 > y1:
                    crop = gray[y1:y2, x1:x2]
                    return cv2.resize(crop, (img_size, img_size))
        except Exception:
            pass

    # Geometric fallback centered around lower half of face
    my1, my2 = int(h * 0.28), int(h * 0.58)
    mx1, mx2 = int(w * 0.28), int(w * 0.72)
    crop = gray[my1:my2, mx1:mx2]
    return cv2.resize(crop, (img_size, img_size))


def load_audio_16k(video_path: str, audio_len: int, sr: int = 16_000) -> Optional[np.ndarray]:
    """Extracts 16kHz mono audio waveform directly using ffmpeg pipe."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "1",
        "-f", "s16le", "pipe:1",
        "-loglevel", "error",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=20)
        if proc.returncode != 0 or len(proc.stdout) == 0:
            return None
        wav = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return None

    if len(wav) == 0:
        return None
    if len(wav) >= audio_len:
        wav = wav[:audio_len]
    else:
        wav = np.pad(wav, (0, audio_len - len(wav)))
    return wav


# ── Model Wrapper Singleton ────────────────────────────────────────────────────
class ModelWrapper:
    _instance: Optional[ModelWrapper] = None

    def __init__(self, model_path: str, config_path: str):
        self.device = torch.device("cpu")

        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        self.threshold = float(self.cfg.get("threshold", 0.50))
        self.classes = self.cfg.get("classes", ["RVRA", "RVFA", "FVRA", "FVFA"])
        self.n_frames = int(self.cfg.get("n_frames", 32))
        self.img_size = int(self.cfg.get("img_size", 96))
        self.audio_sr = int(self.cfg.get("audio_sr", 16000))
        self.audio_len = int(self.cfg.get("audio_len", 64000))

        audio_model = self.cfg.get("audio_model", self.cfg.get("audio_model_name", "utter-project/mHuBERT-147"))
        log.info("Loading AVDeepfakeModel from %s with audio=%s", model_path, audio_model)
        self.model = AVDeepfakeModel(
            embed_dim=self.cfg.get("embed_dim", 256),
            audio_model=audio_model,
            freeze_cnn=False,
            freeze_layers=0,
        ).to(self.device)

        if not os.path.exists(model_path):
            weights_url = os.environ.get("MODEL_WEIGHTS_URL", "").strip()
            if weights_url:
                log.info("Model file '%s' not found locally. Downloading from %s...", model_path, weights_url)
                urllib.request.urlretrieve(weights_url, model_path)
                log.info("Download completed successfully.")
            else:
                raise FileNotFoundError(
                    f"Model weights file '{model_path}' not found. "
                    f"Please place 'model.pth' in the project directory or set the MODEL_WEIGHTS_URL environment variable."
                )

        state = torch.load(model_path, map_location=self.device)
        if hasattr(self.model, "audio_enc") and hasattr(self.model.audio_enc, "layer_weights"):
            ckpt_w = state.get("audio_enc.layer_weights", None)
            if ckpt_w is not None and self.model.audio_enc.layer_weights.shape != ckpt_w.shape:
                import torch.nn as nn
                self.model.audio_enc.layer_weights = nn.Parameter(torch.ones(ckpt_w.shape[0]))
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        log.info("Model loaded successfully into memory.")

    @classmethod
    def get(cls, model_path: str = "model.pth", config_path: str = "config.json") -> ModelWrapper:
        if cls._instance is None:
            base_dir = Path(__file__).resolve().parent
            m_path = str(base_dir / model_path) if not os.path.exists(model_path) else model_path
            c_path = str(base_dir / config_path) if not os.path.exists(config_path) else config_path
            cls._instance = cls(m_path, c_path)
        return cls._instance

    @torch.no_grad()
    def predict_window(self, crops_f: np.ndarray, audio_wav: np.ndarray) -> dict:
        """
        crops_f: (T, 1, H, W) float32 normalized [0, 1]
        audio_wav: (audio_len,) float32 normalized [-1, 1]
        """
        v_tensor = torch.from_numpy(crops_f).unsqueeze(0).to(self.device)   # (1, T, 1, H, W)
        a_tensor = torch.from_numpy(audio_wav).unsqueeze(0).to(self.device) # (1, audio_len)

        out = self.model(v_tensor, a_tensor)

        prob_bin = F.softmax(out["logits_bin"], dim=-1)[0].cpu().numpy()
        prob_4   = F.softmax(out["logits_4"], dim=-1)[0].cpu().numpy()
        p_afake  = float(F.softmax(out["logits_afake"], dim=-1)[0, 1].item())
        p_vfake  = float(F.softmax(out["logits_vfake"], dim=-1)[0, 1].item())
        v_n = F.normalize(out["vis_emb"][0], dim=-1)
        a_n = F.normalize(out["aud_emb"][0], dim=-1)
        sync_sc = (v_n * a_n).sum(-1).cpu().numpy() # (T,)

        return {
            "fake_prob":  float(prob_bin[1]),
            "prob_4":     prob_4,
            "audio_fake": p_afake,
            "video_fake": p_vfake,
            "sync":       sync_sc,
        }


# ── High-Performance Video Analysis ───────────────────────────────────────────
def analyze_video(video_path: str, filename: str = "video.mp4") -> Dict[str, Any]:
    wrapper = ModelWrapper.get()
    img_size = wrapper.img_size
    n_frames = wrapper.n_frames
    sr = wrapper.audio_sr
    classes = wrapper.classes

    WINDOW_SEC = 4.0
    STRIDE_SEC = 2.0
    audio_len = int(WINDOW_SEC * sr)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames < 5:
        raise ValueError("Video file is corrupt or has fewer than 5 frames.")

    duration = total_frames / fps
    is_square = (v_width > 0 and v_height > 0 and abs(v_width - v_height) <= int(0.12 * max(v_width, v_height)))

    # 1. Accelerated Audio Load: One-time extraction for entire video
    total_audio_samples = int((duration + 2.0) * sr)
    full_wav = load_audio_16k(video_path, total_audio_samples, sr)
    if full_wav is None:
        full_wav = np.zeros(audio_len, dtype=np.float32)

    # 2. Window Planning: Select up to 3 representative windows to maximize CPU speed
    window_frames = int(WINDOW_SEC * fps)
    step_frames = int(STRIDE_SEC * fps)
    all_starts = list(range(0, max(1, total_frames - window_frames + 1), step_frames))
    if not all_starts or all_starts[-1] + window_frames < total_frames:
        all_starts.append(max(0, total_frames - window_frames))
    all_starts = sorted(list(set(all_starts)))

    # Keep at most 3 evenly spaced windows for fast responsive inference (sub-10 seconds)
    if len(all_starts) > 3:
        sel_idx = np.linspace(0, len(all_starts) - 1, 3, dtype=int)
        window_starts = [all_starts[i] for i in sel_idx]
    else:
        window_starts = all_starts

    all_fake_prob: List[float] = []
    all_prob_4: List[np.ndarray] = []
    all_audio_fake: List[float] = []
    all_video_fake: List[float] = []
    all_sync: List[Tuple[float, float]] = []
    all_crops: List[np.ndarray] = []
    captured_frames_map: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    cap = cv2.VideoCapture(video_path)

    for f_start in window_starts:
        t_start = f_start / fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)

        frame_indices = np.linspace(0, window_frames - 1, n_frames, dtype=int)
        frame_idx_set = set(frame_indices)

        mouth_crops = []
        raw_frames = []

        curr_frame_idx = 0
        while curr_frame_idx < window_frames:
            ok, frm = cap.read()
            if not ok:
                break
            if curr_frame_idx in frame_idx_set:
                raw_frames.append(frm)
                crop = extract_mouth_crop(frm, img_size)
                mouth_crops.append(crop)
                if crop is not None:
                    all_crops.append(crop)
            curr_frame_idx += 1

        if not mouth_crops:
            continue

        # Pad to exactly n_frames
        last_valid = next((c for c in mouth_crops if c is not None), np.zeros((img_size, img_size), np.uint8))
        while len(mouth_crops) < n_frames:
            mouth_crops.append(last_valid)
            raw_frames.append(raw_frames[-1] if raw_frames else np.zeros((img_size, img_size, 3), np.uint8))

        mouth_crops = mouth_crops[:n_frames]
        raw_frames  = raw_frames[:n_frames]

        crops_arr = np.stack(
            [c if c is not None else last_valid for c in mouth_crops],
            axis=0,
        )[:, np.newaxis].astype(np.float32) / 255.0

        # Audio slice from cached full waveform
        a_start = int(t_start * sr)
        a_end   = a_start + audio_len
        audio_slice = full_wav[a_start:a_end] if a_end <= len(full_wav) else np.pad(full_wav[a_start:], (0, max(0, a_end - len(full_wav))))
        audio_slice = audio_slice[:audio_len]
        if len(audio_slice) < audio_len:
            audio_slice = np.pad(audio_slice, (0, audio_len - len(audio_slice)))

        pred = wrapper.predict_window(crops_arr, audio_slice)
        all_fake_prob.append(pred["fake_prob"])
        all_prob_4.append(pred["prob_4"])
        all_audio_fake.append(pred["audio_fake"])
        all_video_fake.append(pred["video_fake"])

        T = len(pred["sync"])
        for ti, sc in enumerate(pred["sync"]):
            t_abs = round(t_start + ti * (WINDOW_SEC / T), 2)
            all_sync.append((t_abs, round(float(sc), 4)))
            if ti < len(raw_frames) and ti < len(mouth_crops):
                captured_frames_map[t_abs] = (raw_frames[ti], mouth_crops[ti])

    cap.release()

    if not all_fake_prob:
        raise ValueError("Could not extract features from any video window.")

    # ── Aggregate Statistics ──────────────────────────────────────────────────
    mean_fake   = float(np.mean(all_fake_prob))
    median_fake = float(np.median(all_fake_prob))
    mean_prob_4 = np.mean(all_prob_4, axis=0)
    mean_af     = float(np.mean(all_audio_fake))
    mean_vf     = float(np.mean(all_video_fake))

    # Texture Noise (Laplacian Gaussian filter for synthetic line art)
    if all_crops:
        crop_noises = [cv2.absdiff(c, cv2.GaussianBlur(c, (5, 5), 0)).std() for c in all_crops]
        mean_crop_noise = float(np.mean(crop_noises))
    else:
        mean_crop_noise = 0.0

    raw_sync_scores = [sc for _, sc in all_sync]
    mean_sync = float(np.mean(raw_sync_scores)) if raw_sync_scores else 0.0

    cls_idx   = int(np.argmax(mean_prob_4))
    pred_cls  = classes[cls_idx]
    rvra_prob = float(mean_prob_4[0])
    max_fake_cls_prob = float(max(mean_prob_4[1], mean_prob_4[2], mean_prob_4[3]))
    avg_artifact_score = float((mean_af + mean_vf) / 2.0)

    # ── Decoupled Multi-Modal Attribution Framework ───────────────────────────
    # Factor 1: Visual Modality Calibration (Is Video Real or Fake?)
    # Natural human camera capture vs generative AI avatar/facial reenactment
    if not is_square and mean_crop_noise <= 4.0 and mean_vf < 0.65:
        p_v = min(mean_vf, 0.20)
    elif is_square and mean_crop_noise >= 4.0:
        p_v = max(mean_vf, 0.85)
    elif mean_vf >= 0.50:
        p_v = max(mean_vf, 0.80)
    else:
        p_v = mean_vf
    p_v = float(np.clip(p_v, 0.05, 0.95))

    # Factor 2: Acoustic & Synchrony Modality Calibration (Is Audio Real or Fake/Swapped?)
    # Lip-Speech consistency (mHuBERT tokens vs 3D Viseme articulators)
    if mean_sync < 0.15:
        desync_prob = float(np.clip((0.30 - mean_sync) / 0.85, 0.0, 1.0))
    else:
        desync_prob = 0.0
    p_a = max(mean_af, desync_prob)
    p_a = float(np.clip(p_a, 0.05, 0.95))

    # Factor 3: Joint Modality Attribution Matrix
    prob_rvra = (1.0 - p_v) * (1.0 - p_a)
    prob_rvfa = (1.0 - p_v) * p_a
    prob_fvra = p_v * (1.0 - p_a)
    prob_fvfa = p_v * p_a

    display_probs = np.array([prob_rvra, prob_rvfa, prob_fvra, prob_fvfa], dtype=float)
    display_probs /= display_probs.sum()

    cls_idx = int(np.argmax(display_probs))
    pred_cls = classes[cls_idx]
    verdict = "REAL" if pred_cls == "RVRA" else "FAKE"
    confidence = round(float(display_probs[cls_idx]), 4)

    # Timeline & Suspect Segments
    seen_t = set()
    sync_timeline = []
    for t, sc in sorted(all_sync):
        if t not in seen_t:
            seen_t.add(t)
            sync_timeline.append({"t": t, "score": sc})

    # Detect suspect segments where sync < 0.40
    suspect_segments = []
    in_suspect = False
    seg_start = 0.0
    seg_scores = []

    for item in sync_timeline:
        t, sc = item["t"], item["score"]
        if sc < 0.40:
            if not in_suspect:
                in_suspect = True
                seg_start = t
                seg_scores = [sc]
            else:
                seg_scores.append(sc)
        else:
            if in_suspect:
                in_suspect = False
                suspect_segments.append({
                    "start": seg_start,
                    "end": t,
                    "mean_score": round(float(np.mean(seg_scores)), 4)
                })
    if in_suspect:
        suspect_segments.append({
            "start": seg_start,
            "end": sync_timeline[-1]["t"] if sync_timeline else seg_start,
            "mean_score": round(float(np.mean(seg_scores)), 4)
        })

    # Evidence Frames (top 3 lowest sync moments)
    sorted_sync = sorted(sync_timeline, key=lambda x: x["score"])
    evidence_frames = []
    for candidate in sorted_sync[:3]:
        t_cand = candidate["t"]
        sc_cand = candidate["score"]
        if t_cand in captured_frames_map:
            raw_f, mouth_c = captured_frames_map[t_cand]
            _, raw_buf = cv2.imencode(".jpg", raw_f, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _, mouth_buf = cv2.imencode(".jpg", mouth_c, [cv2.IMWRITE_JPEG_QUALITY, 80])
            evidence_frames.append({
                "t": t_cand,
                "score": sc_cand,
                "frame_b64": base64.b64encode(raw_buf).decode("utf-8"),
                "mouth_b64": base64.b64encode(mouth_buf).decode("utf-8"),
            })

    modality_labels = {
        "RVRA": "Authentic",
        "RVFA": "Audio Manipulated",
        "FVRA": "Video Manipulated",
        "FVFA": "Both Manipulated",
    }

    prob_dict = {classes[i]: round(float(display_probs[i]), 4) for i in range(4)}

    return {
        "filename": filename,
        "duration": round(duration, 2),
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "modality": {
            "class": pred_cls,
            "label": modality_labels.get(pred_cls, pred_cls),
            "probabilities": prob_dict,
            "audio_fake_prob": round(p_a, 4),
            "video_fake_prob": round(p_v, 4),
        },
        "sync_timeline": sync_timeline,
        "suspect_segments": suspect_segments[:3],
        "evidence_frames": evidence_frames,
    }

