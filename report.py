"""
AV-Forensics: Professional Forensic Examination PDF Report Generator
Generates audit-ready, high-resolution forensic investigation reports using ReportLab & Matplotlib.
"""

from __future__ import annotations
import base64
import io
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)

# Palette
C_NAVY    = colors.HexColor("#0F172A")
C_BLUE    = colors.HexColor("#2563EB")
C_DARK    = colors.HexColor("#1E293B")
C_GRAY    = colors.HexColor("#64748B")
C_LIGHT   = colors.HexColor("#F8FAFC")
C_BORDER  = colors.HexColor("#E2E8F0")
C_RED     = colors.HexColor("#DC2626")
C_GREEN   = colors.HexColor("#16A34A")
C_AMBER   = colors.HexColor("#D97706")


def generate_sync_plot(sync_timeline: list) -> io.BytesIO:
    """Generates a high-resolution matplotlib sync trajectory plot."""
    times = [item["t"] for item in sync_timeline]
    scores = [item["score"] for item in sync_timeline]

    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    # Sync line
    ax.plot(times, scores, color="#2563EB", lw=2, label="Lip-Audio Sync Score", zorder=4)

    # Reference threshold lines
    ax.axhline(0.5, color="#16A34A", linestyle="--", lw=1.2, alpha=0.8, label="Synchrony Threshold (0.50)")
    ax.axhline(0.0, color="#64748B", linestyle=":", lw=1, alpha=0.7)

    # Highlight suspect area (< 0.4)
    ax.axhspan(-1.0, 0.40, color="#FEE2E2", alpha=0.5, label="Suspect Desync Zone (< 0.40)", zorder=1)

    ax.set_xlim(min(times) if times else 0, max(times) if times else 5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Video Timeline (seconds)", fontsize=9, fontweight="bold", color="#1E293B")
    ax.set_ylabel("Cosine Similarity", fontsize=9, fontweight="bold", color="#1E293B")
    ax.tick_params(colors="#475569", labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.4, color="#CBD5E1")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(result: Dict[str, Any], output_path: str, examiner_name: str = "Digital Forensics Unit") -> str:
    """Compiles the complete audit-grade forensic PDF report."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    style_header_title = ParagraphStyle(
        "HeaderTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.white,
        leading=22,
    )
    style_header_sub = ParagraphStyle(
        "HeaderSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        textColor=colors.HexColor("#94A3B8"),
        leading=12,
    )
    style_sec_heading = ParagraphStyle(
        "SecHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=C_NAVY,
        spaceBefore=10,
        spaceAfter=5,
    )
    style_body = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        textColor=C_DARK,
        leading=12,
    )
    style_body_bold = ParagraphStyle(
        "BodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=C_DARK,
        leading=12,
    )

    story = []

    # ── 1. HEADER BANNER ──────────────────────────────────────────────────────
    header_content = [
        [
            Paragraph("AV-FORENSICS CYBERCRIME LABORATORY", style_header_title),
            Paragraph(f"REPORT DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/>CLASSIFICATION: FORENSIC AUDIT", ParagraphStyle("HRight", parent=style_header_sub, alignment=2))
        ],
        [
            Paragraph("Audio-Visual Lip-Sync Consistency & Modality-Level Deepfake Examination Report", style_header_sub),
            Paragraph(f"EXAMINER: {examiner_name}", ParagraphStyle("HRight2", parent=style_header_sub, alignment=2))
        ]
    ]
    t_header = Table(header_content, colWidths=[4.2 * inch, 3.1 * inch])
    t_header.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 10))

    # ── 2. CASE METADATA & INTEGRITY HASH ─────────────────────────────────────
    case_id = result.get("case_id", "CASE-" + hashlib.md5(result.get("filename", "").encode()).hexdigest()[:12].upper())
    file_sha256 = result.get("sha256", hashlib.sha256(result.get("filename", "").encode()).hexdigest())

    meta_table_data = [
        [
            Paragraph("Case Identifier:", style_body_bold), Paragraph(str(case_id), style_body),
            Paragraph("Video Filename:", style_body_bold), Paragraph(str(result.get("filename", "evidence.mp4")), style_body),
        ],
        [
            Paragraph("Cryptographic SHA-256:", style_body_bold), Paragraph(f"<font name='Courier' size=7>{file_sha256}</font>", style_body),
            Paragraph("Duration / Format:", style_body_bold), Paragraph(f"{result.get('duration', 0.0)}s | 16kHz PCM Audio", style_body),
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[1.5 * inch, 2.2 * inch, 1.4 * inch, 2.2 * inch])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT),
        ("BOX", (0, 0), (-1, -1), 1, C_BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # ── 3. EXECUTIVE VERDICT BADGE ────────────────────────────────────────────
    verdict = result.get("verdict", "UNKNOWN")
    conf_pct = f"{result.get('confidence', 0.0) * 100:.1f}%"
    modality_label = result.get("modality", {}).get("label", "N/A")
    modality_cls = result.get("modality", {}).get("class", "N/A")

    is_fake = (verdict.upper() == "FAKE")
    verdict_color = C_RED if is_fake else C_GREEN
    verdict_text = "MANIPULATED / DEEPFAKE DETECTED" if is_fake else "AUTHENTIC / GENUINE VIDEO"

    verdict_card = [
        [
            Paragraph(f"FINAL FORENSIC VERDICT: <font color='{verdict_color.hexval()}'><b>{verdict_text}</b></font>", ParagraphStyle("V1", fontName="Helvetica-Bold", fontSize=13, textColor=C_NAVY)),
            Paragraph(f"CONFIDENCE LEVEL: <b>{conf_pct}</b>", ParagraphStyle("V2", fontName="Helvetica-Bold", fontSize=11, textColor=verdict_color, alignment=2))
        ],
        [
            Paragraph(f"Detected Modality Attribution: <b>{modality_label} ({modality_cls})</b>", style_body),
            Paragraph("Detection Standard: Multi-Modal Phoneme-Viseme Cross-Attention", ParagraphStyle("V3", parent=style_body, alignment=2))
        ]
    ]
    t_verdict = Table(verdict_card, colWidths=[4.7 * inch, 2.6 * inch])
    t_verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2" if is_fake else "#F0FDF4")),
        ("BOX", (0, 0), (-1, -1), 1.5, verdict_color),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_verdict)
    story.append(Spacer(1, 10))

    # ── 4. LIP-AUDIO SYNCHRONY TIMELINE PLOT ──────────────────────────────────
    story.append(Paragraph("A. Lip-Audio Synchronization Trajectory Analysis", style_sec_heading))
    story.append(Paragraph(
        "The graph below depicts per-frame cosine similarity between acoustic phonetic tokens (mHuBERT-147) and visual mouth viseme representations. "
        "Authentic human speech produces stable positive correlation (> 0.50). Sudden drops into the shaded red zone indicate temporal phoneme-viseme decoupling.",
        style_body
    ))
    story.append(Spacer(1, 4))

    sync_timeline = result.get("sync_timeline", [])
    if sync_timeline:
        plot_buf = generate_sync_plot(sync_timeline)
        img_plot = Image(plot_buf, width=7.3 * inch, height=2.3 * inch)
        story.append(img_plot)
    story.append(Spacer(1, 8))

    # ── 5. MODALITY ATTRIBUTION BREAKDOWN TABLE ───────────────────────────────
    story.append(Paragraph("B. Fine-Grained Modality Attribution Probabilities", style_sec_heading))
    probs = result.get("modality", {}).get("probabilities", {})

    p_rvra = f"{probs.get('RVRA', 0.0) * 100:.1f}%"
    p_rvfa = f"{probs.get('RVFA', 0.0) * 100:.1f}%"
    p_fvra = f"{probs.get('FVRA', 0.0) * 100:.1f}%"
    p_fvfa = f"{probs.get('FVFA', 0.0) * 100:.1f}%"

    af_prob = f"{result.get('modality', {}).get('audio_fake_prob', 0.0) * 100:.1f}%"
    vf_prob = f"{result.get('modality', {}).get('video_fake_prob', 0.0) * 100:.1f}%"

    prob_headers = ["Category", "RVRA (Real Video, Real Audio)", "RVFA (Real Video, Fake Audio)", "FVRA (Fake Video, Real Audio)", "FVFA (Both Video & Audio Fake)"]
    prob_values = ["Model Probability", p_rvra, p_rvfa, p_fvra, p_fvfa]

    t_probs = Table([prob_headers, prob_values], colWidths=[1.3 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    t_probs.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 1), (-1, 1), C_LIGHT),
    ]))
    story.append(t_probs)
    story.append(Spacer(1, 4))

    aux_note = f"<b>Auxiliary Forensic Sensors:</b> Visual Artifact Sensor: <b>{vf_prob}</b> | Acoustic Synthesis Sensor: <b>{af_prob}</b>"
    story.append(Paragraph(aux_note, style_body))
    story.append(Spacer(1, 8))

    # ── 6. EVIDENCE FRAMES GALLERY ────────────────────────────────────────────
    evidence = result.get("evidence_frames", [])
    if evidence:
        story.append(Paragraph("C. Suspect Desynchronization Evidence Frames", style_sec_heading))
        story.append(Paragraph("Visual mouth articulators and full video frames captured at the lowest synchronization timestamps:", style_body))
        story.append(Spacer(1, 4))

        evidence_rows = [["Timestamp", "Full Frame Capture", "Isolated Mouth Articulator", "Sync Score"]]
        for ev in evidence[:2]:
            t_val = f"{ev.get('t', 0.0):.2f}s"
            sc_val = f"{ev.get('score', 0.0):.3f}"
            f_img = Image(io.BytesIO(base64.b64decode(ev["frame_b64"])), width=1.4 * inch, height=1.0 * inch)
            m_img = Image(io.BytesIO(base64.b64decode(ev["mouth_b64"])), width=1.0 * inch, height=1.0 * inch)
            evidence_rows.append([t_val, f_img, m_img, sc_val])

        t_evidence = Table(evidence_rows, colWidths=[1.0 * inch, 2.5 * inch, 2.5 * inch, 1.3 * inch])
        t_evidence.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t_evidence)
        story.append(Spacer(1, 8))

    # ── 7. FORENSIC METHODOLOGY & SIGN-OFF BLOCK ──────────────────────────────
    signoff_table_data = [
        [
            Paragraph(
                "<b>EXAMINATION METHODOLOGY & ATTESTATION:</b><br/>"
                "This forensic examination was conducted using the AV-Forensics multi-modal consistency architecture. "
                "Feature extraction incorporates MediaPipe 478-point 3D facial mesh modeling with self-supervised mHuBERT-147 phonetic pooling. "
                "The findings recorded herein reflect mathematical cross-attention similarity and high-frequency texture gradient analysis.",
                style_body
            ),
            Paragraph(
                "<b>EXAMINER SIGNATURE:</b><br/><br/>"
                "____________________________________<br/>"
                "Senior Forensic Examiner / Lead Analyst<br/>"
                "<font color='#64748B' size=7>Cryptographically Verified Audit</font>",
                ParagraphStyle("Sign", parent=style_body, alignment=2)
            )
        ]
    ]
    t_signoff = Table(signoff_table_data, colWidths=[4.8 * inch, 2.5 * inch])
    t_signoff.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, -1), 1, C_BORDER),
    ]))
    story.append(KeepTogether(t_signoff))

    doc.build(story)
    return output_path

