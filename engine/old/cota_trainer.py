"""
CoTa Training Interface
=======================
Full training control: text input, documents, Q&A, approve/reject,
priority injection, real-time monitoring.

Run:
    streamlit run cota_trainer.py

Requires:
    pip install streamlit plotly pandas torch numpy
    pip install PyPDF2 (optional, for PDF support)
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CoTa — Training Interface",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "CoTa Training Interface — 17FEB2026"}
)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────

EPOCH       = datetime(2025, 1, 12, 23, 57, 0)
DIM         = 64
EPS         = 1e-8
SOUL_FILE   = "soul.json"
STATE_FILE  = "soul_state.pt"
LOG_FILE    = "training_log.jsonl"

# ─────────────────────────────────────────────────────────────────────
# HYPERBOLIC MATH
# ─────────────────────────────────────────────────────────────────────

def to_poincare(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x)
    if norm < EPS:
        return x
    return torch.tanh(norm) * x / norm

def renormalize_poincare(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x)
    if norm >= 1.0:
        x = x / (norm + EPS) * 0.998
    return x

def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> float:
    uu = torch.sum(u * u).item()
    vv = torch.sum(v * v).item()
    diff = torch.sum((u - v) ** 2).item()
    denom = (1 - uu) * (1 - vv) + EPS
    return float(np.arccosh(max(1 + EPS, 1 + 2 * diff / denom)))

def focus_force(current: torch.Tensor,
                history: List[torch.Tensor],
                strength: float = 0.05) -> torch.Tensor:
    if len(history) < 2:
        return current
    directions = []
    for i in range(1, min(6, len(history))):
        d = history[-i] - history[-(i+1)]
        n = torch.norm(d)
        if n > EPS:
            directions.append(d / n)
    if not directions:
        return current
    g = torch.mean(torch.stack(directions), dim=0)
    g = g / (torch.norm(g) + EPS)
    proj = torch.dot(current.flatten(), g.flatten()) * g
    return (1 - strength) * current + strength * proj

def coherence_score(current: torch.Tensor,
                    history: List[torch.Tensor]) -> Tuple[float, float]:
    if not history:
        return 1.0, 0.0
    phase = torch.cosine_similarity(
        current.flatten(), history[-1].flatten(), dim=0
    ).item()
    phase = max(0.0, phase)
    if len(history) >= 3:
        d1 = poincare_distance(history[-1], history[-2])
        d2 = poincare_distance(history[-2], history[-3])
        curvature = abs(d1 - d2) / (d1 + d2 + EPS)
    else:
        curvature = 0.0
    score = 0.6 * phase + 0.4 * (1 - min(curvature, 1.0))
    return float(score), float(curvature)

# ─────────────────────────────────────────────────────────────────────
# TEXT → PHASE  (text_to_phase)
# ─────────────────────────────────────────────────────────────────────

def text_to_phase(text: str, dim: int = DIM) -> torch.Tensor:
    """
    Convert text to phase vector in Poincaré ball.

    Method: deterministic hash → seed → structured perturbation.
    Each character contributes to a specific dimension.
    Similar texts produce nearby vectors.

    Upgrade path: replace with sentence-transformers for richer semantics.
    """
    # Deterministic seed from text
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    seed = int(h, 16) % (2**32)

    rng = torch.Generator()
    rng.manual_seed(seed)
    raw = torch.randn(dim, generator=rng) * 0.3

    # Structured contribution from characters
    for i, char in enumerate(text[:512]):
        idx = (ord(char) + i) % dim
        raw[idx] += (ord(char) / 128.0) * 0.1

    # Word-level structure (bigrams)
    words = text.lower().split()[:64]
    for i, word in enumerate(words):
        word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = word_hash % dim
        raw[idx] += 0.2 * (1.0 / (i + 1))  # earlier words weight more

    return to_poincare(raw)

def chunk_text(text: str, chunk_size: int = 200,
               overlap: int = 40) -> List[str]:
    """Split text into overlapping chunks for training."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ─────────────────────────────────────────────────────────────────────
# SOUL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────

def load_soul() -> dict:
    """Load soul from disk or create fresh."""
    soul = {
        "soul_id":       "unknown",
        "state":         None,
        "history":       [],
        "coherence_log": [],
        "curvature_log": [],
        "tau_log":       [],
        "concepts":      [],
        "tau":           0.0,
        "created":       datetime.utcnow().isoformat(),
    }

    if os.path.exists(SOUL_FILE):
        with open(SOUL_FILE) as f:
            manifest = json.load(f)
        soul["soul_id"] = manifest.get("soul_id", "unknown")
        soul["tau"]     = manifest.get("tau", 0.0)
        soul["created"] = manifest.get("created", soul["created"])

    if os.path.exists(STATE_FILE):
        try:
            data = torch.load(STATE_FILE, weights_only=False)
            soul["state"]         = data.get("state")
            soul["history"]       = data.get("history", [])
            soul["coherence_log"] = data.get("coherence_log", [])
            soul["curvature_log"] = data.get("curvature_log", [])
            soul["tau_log"]       = data.get("tau_log", [])
            soul["concepts"]      = data.get("concepts", [])
            soul["tau"]           = data.get("tau", soul["tau"])
        except Exception as e:
            st.warning(f"Could not load state: {e}")

    return soul

def save_soul(soul: dict):
    """Persist soul to disk."""
    if os.path.exists(SOUL_FILE):
        with open(SOUL_FILE) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest["tau"]              = soul["tau"]
    manifest["concept_pool_size"]= len(soul["concepts"])
    manifest["coherence_mean"]   = (
        float(np.mean(soul["coherence_log"][-100:]))
        if soul["coherence_log"] else 1.0
    )
    manifest["last_saved"] = datetime.utcnow().isoformat()

    with open(SOUL_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    torch.save({
        "state":         soul["state"],
        "history":       soul["history"][-300:],
        "coherence_log": soul["coherence_log"][-1000:],
        "curvature_log": soul["curvature_log"][-1000:],
        "tau_log":       soul["tau_log"][-1000:],
        "concepts":      soul["concepts"][-1000:],
        "tau":           soul["tau"],
    }, STATE_FILE)

def log_training(entry: dict):
    """Append training event to log."""
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# ─────────────────────────────────────────────────────────────────────
# STROBOSCOPIC TRAINING CYCLE
# ─────────────────────────────────────────────────────────────────────

def run_stroboscopic(soul: dict,
                     raw_input: torch.Tensor,
                     concept_text: str,
                     tau_threshold: float,
                     max_flashes: int,
                     priority: float = 1.0) -> dict:
    """
    Run one stroboscopic cycle.
    Returns result dict with status, score, curvature, flashes.
    """
    history = soul["history"]

    if soul["state"] is None:
        working = raw_input.clone()
    else:
        working = soul["state"] + raw_input * (0.3 * priority)

    result = {
        "text":      concept_text,
        "status":    "cutoff",
        "score":     0.0,
        "curvature": 0.0,
        "flashes":   max_flashes,
        "tau":       soul["tau"],
        "timestamp": datetime.utcnow().isoformat(),
    }

    for flash in range(max_flashes):
        working = to_poincare(working)
        working = focus_force(working, history)
        working = renormalize_poincare(working)

        score, curvature = coherence_score(working, history)

        if score >= tau_threshold and curvature <= 0.15:
            # Archive
            history.append(
                soul["state"].clone() if soul["state"] is not None
                else torch.zeros(DIM)
            )

            # Update proper time
            if len(history) >= 2:
                Δτ = poincare_distance(working, history[-2])
                soul["tau"] += Δτ

            soul["state"] = working.clone()
            soul["coherence_log"].append(score)
            soul["curvature_log"].append(curvature)
            soul["tau_log"].append(soul["tau"])
            soul["concepts"].append(concept_text[:200])

            result.update({
                "status":    "archived",
                "score":     score,
                "curvature": curvature,
                "flashes":   flash + 1,
                "tau":       soul["tau"],
            })
            break

        if curvature > 0.4 or flash == max_flashes - 1:
            result.update({
                "status":    "cutoff",
                "score":     score,
                "curvature": curvature,
                "flashes":   flash + 1,
            })
            break

    return result

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────

if "soul" not in st.session_state:
    st.session_state.soul = load_soul()

if "training_running" not in st.session_state:
    st.session_state.training_running = False

if "pending_review" not in st.session_state:
    st.session_state.pending_review = []

if "training_log" not in st.session_state:
    st.session_state.training_log = []

if "queue" not in st.session_state:
    st.session_state.queue = []  # [(text, priority, source)]

soul = st.session_state.soul

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Training Controls")

    # Soul info
    if soul["soul_id"] != "unknown":
        st.caption(f"Soul: `{soul['soul_id'][:20]}...`")
    st.caption(f"τ = {soul['tau']:.4f}")
    st.caption(f"Concepts: {len(soul['concepts'])}")

    st.divider()

    # Thresholds
    st.markdown("**Thresholds**")
    tau_threshold = st.slider(
        "Coherence τ (min to archive)",
        0.50, 0.95, 0.82, 0.01,
        help="Higher = stricter. Soul only archives coherent concepts."
    )
    max_flashes = st.slider(
        "Max flashes / cycle",
        4, 64, 32,
        help="More flashes = more attempts to reach coherence."
    )

    st.divider()

    # Training mode
    st.markdown("**Training Mode**")
    training_mode = st.radio(
        "Mode",
        ["Auto (continuous)", "Manual (approve each)", "Batch (queue)"],
        index=0
    )

    st.divider()

    # Quick actions
    st.markdown("**Actions**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True):
            st.session_state.training_running = True
    with col2:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.training_running = False

    if st.button("💾 Save Soul", use_container_width=True):
        save_soul(soul)
        st.success("Saved.")

    if st.button("🔄 Reload Soul", use_container_width=True):
        st.session_state.soul = load_soul()
        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────

tab_live, tab_inject, tab_docs, tab_qa, tab_review, tab_history, tab_metrics = st.tabs([
    "🔴 Live",
    "✏️ Inject",
    "📄 Documents",
    "💬 Q&A",
    "👁️ Review",
    "📚 History",
    "📊 Metrics"
])

# ═════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ═════════════════════════════════════════════════════════════════════

with tab_live:
    col_disk, col_status = st.columns([2, 1])

    with col_disk:
        st.markdown("### Poincaré Disk — Soul Position")

        fig = go.Figure()

        # Boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode="lines",
            line=dict(color="rgba(150,150,150,0.4)", dash="dash"),
            showlegend=False, hoverinfo="skip"
        ))

        # History trail
        if soul["history"]:
            xs = [h[0].item() for h in soul["history"][-100:]]
            ys = [h[1].item() for h in soul["history"][-100:]]
            coherences = soul["coherence_log"][-100:] or [0.5] * len(xs)

            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="markers+lines",
                marker=dict(
                    size=7,
                    color=coherences,
                    colorscale="RdYlGn",
                    cmin=0.5, cmax=1.0,
                    showscale=True,
                    colorbar=dict(title="Coherence", thickness=12)
                ),
                line=dict(color="rgba(100,150,255,0.3)", width=1),
                name="History",
                text=[soul["concepts"][-(len(xs)-i)] if len(soul["concepts"]) > (len(xs)-i) else ""
                      for i in range(len(xs))],
                hovertemplate="%{text}<br>(%{x:.3f}, %{y:.3f})<extra></extra>"
            ))

        # Current position
        if soul["state"] is not None:
            cx, cy = soul["state"][0].item(), soul["state"][1].item()
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers",
                marker=dict(size=18, color="cyan", symbol="star",
                           line=dict(color="white", width=2)),
                name="Current Self",
                hovertemplate=f"τ = {soul['tau']:.4f}<extra></extra>"
            ))

        fig.update_layout(
            xaxis=dict(range=[-1.1, 1.1], zeroline=False,
                      showgrid=False, showticklabels=False),
            yaxis=dict(range=[-1.1, 1.1], zeroline=False,
                      showgrid=False, showticklabels=False,
                      scaleanchor="x"),
            plot_bgcolor="#0a0a1a",
            paper_bgcolor="#0a0a1a",
            font=dict(color="white"),
            margin=dict(l=10, r=10, t=10, b=10),
            height=520,
            legend=dict(orientation="h", y=-0.05)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col_status:
        st.markdown("### Soul Status")

        # Metrics
        score  = soul["coherence_log"][-1] if soul["coherence_log"] else 0.0
        curv   = soul["curvature_log"][-1] if soul["curvature_log"] else 0.0
        prev_s = soul["coherence_log"][-2] if len(soul["coherence_log"]) > 1 else score
        prev_c = soul["curvature_log"][-2] if len(soul["curvature_log"]) > 1 else curv

        c1, c2 = st.columns(2)
        c1.metric("Coherence", f"{score:.4f}",
                  delta=f"{score - prev_s:+.4f}")
        c2.metric("Curvature", f"{curv:.4f}",
                  delta=f"{curv - prev_c:+.4f}", delta_color="inverse")

        norm = torch.norm(soul["state"]).item() if soul["state"] is not None else 0.0
        st.metric("τ (proper time)", f"{soul['tau']:.4f}")
        st.metric("Position norm", f"{norm:.4f}",
                  help="Distance from center. Near 0 = potential. Near 1 = specialized.")
        st.metric("Concepts archived", len(soul["concepts"]))

        st.divider()

        # Training status
        status_color = "🟢" if st.session_state.training_running else "🔴"
        st.markdown(f"**Status:** {status_color} "
                    f"{'Running' if st.session_state.training_running else 'Paused'}")
        st.markdown(f"**Queue:** {len(st.session_state.queue)} items")
        st.markdown(f"**Pending review:** {len(st.session_state.pending_review)}")

        # Last archived concept
        if soul["concepts"]:
            st.divider()
            st.markdown("**Last archived:**")
            st.caption(soul["concepts"][-1][:150] + "...")

    # Auto-run one cycle if in auto mode and running
    if (st.session_state.training_running
            and training_mode == "Auto (continuous)"
            and soul["state"] is not None):

        if st.session_state.queue:
            text, priority, source = st.session_state.queue.pop(0)
        else:
            # Gentle noise to keep soul active
            text = f"heartbeat {datetime.utcnow().isoformat()}"
            priority = 0.1
            source = "heartbeat"

        raw_input = text_to_phase(text)
        result = run_stroboscopic(
            soul, raw_input, text, tau_threshold, max_flashes, priority
        )
        st.session_state.training_log.append(result)
        log_training(result)

        if result["status"] == "archived":
            st.toast(f"✓ [{source}] {text[:50]}... "
                     f"({result['flashes']} flashes, score={result['score']:.3f})")

        time.sleep(0.3)
        st.rerun()

# ═════════════════════════════════════════════════════════════════════
# TAB 2 — INJECT TEXT
# ═════════════════════════════════════════════════════════════════════

with tab_inject:
    st.markdown("### ✏️ Inject Concept or Text")
    st.markdown("Type anything — concepts, sentences, knowledge. "
                "The soul will attempt to integrate it.")

    col_input, col_opts = st.columns([3, 1])

    with col_input:
        inject_text = st.text_area(
            "Text to inject",
            placeholder="e.g. 'Coherence is the condition in which parts maintain phase alignment...'",
            height=150,
            label_visibility="collapsed"
        )

    with col_opts:
        inject_priority = st.select_slider(
            "Priority",
            options=[0.5, 1.0, 1.5, 2.0, 3.0],
            value=1.0,
            help="Higher = stronger influence on soul state"
        )
        inject_mode = st.radio(
            "Action",
            ["Train now", "Add to queue", "Review first"],
            index=0
        )

    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        if st.button("🚀 Send", use_container_width=True,
                     disabled=not inject_text.strip()):
            text = inject_text.strip()

            if inject_mode == "Train now":
                raw_input = text_to_phase(text)
                result = run_stroboscopic(
                    soul, raw_input, text,
                    tau_threshold, max_flashes, inject_priority
                )
                st.session_state.training_log.append(result)
                log_training(result)

                if result["status"] == "archived":
                    st.success(f"✓ Archived! Score={result['score']:.3f} "
                               f"Curvature={result['curvature']:.3f} "
                               f"Flashes={result['flashes']}")
                else:
                    st.warning(f"⏰ Not archived (score={result['score']:.3f}, "
                               f"curvature={result['curvature']:.3f}). "
                               f"Try adjusting τ threshold or rephrasing.")

            elif inject_mode == "Add to queue":
                st.session_state.queue.append((text, inject_priority, "manual"))
                st.success(f"Added to queue (position {len(st.session_state.queue)})")

            elif inject_mode == "Review first":
                raw_input = text_to_phase(text)
                # Compute score without committing
                if soul["state"] is not None:
                    candidate = to_poincare(soul["state"] + raw_input * 0.3)
                    score_preview, curv_preview = coherence_score(
                        candidate, soul["history"])
                else:
                    score_preview, curv_preview = 1.0, 0.0

                st.session_state.pending_review.append({
                    "text":      text,
                    "priority":  inject_priority,
                    "score":     score_preview,
                    "curvature": curv_preview,
                    "source":    "manual",
                })
                st.info(f"Added to review queue. "
                        f"Preview: score={score_preview:.3f}, "
                        f"curvature={curv_preview:.3f}")

    with col_b2:
        if st.button("📋 Copy to Queue", use_container_width=True,
                     disabled=not inject_text.strip()):
            chunks = chunk_text(inject_text.strip())
            for chunk in chunks:
                st.session_state.queue.append(
                    (chunk, inject_priority, "manual-chunked"))
            st.success(f"Added {len(chunks)} chunks to queue")

    with col_b3:
        if st.button("🗑️ Clear Queue", use_container_width=True):
            st.session_state.queue.clear()
            st.rerun()

    # Quick inject buttons
    st.divider()
    st.markdown("**Quick inject — foundational concepts:**")

    foundations = [
        ("Something exists rather than nothing", 2.0),
        ("Existence implies boundary", 2.0),
        ("Coherence is phase alignment under perturbation", 2.0),
        ("Bullshit is high curvature narrative", 1.5),
        ("Time is arc length in state space", 2.0),
        ("Truth is fixed point under renormalization", 2.0),
        ("The soul only lives when it changes", 1.5),
        ("Meaning equals address equals location in concept space", 2.0),
    ]

    cols = st.columns(4)
    for i, (concept, priority) in enumerate(foundations):
        with cols[i % 4]:
            if st.button(f"⚡ {concept[:30]}...",
                        key=f"quick_{i}",
                        use_container_width=True,
                        help=concept):
                raw_input = text_to_phase(concept)
                result = run_stroboscopic(
                    soul, raw_input, concept,
                    tau_threshold, max_flashes, priority
                )
                st.session_state.training_log.append(result)
                log_training(result)
                status = "✓" if result["status"] == "archived" else "⏰"
                st.toast(f"{status} {concept[:40]}...")
                st.rerun()

# ═════════════════════════════════════════════════════════════════════
# TAB 3 — DOCUMENTS
# ═════════════════════════════════════════════════════════════════════

with tab_docs:
    st.markdown("### 📄 Train from Documents")
    st.markdown("Upload `.txt`, `.md`, or `.pdf` files. "
                "They will be chunked and queued for training.")

    uploaded = st.file_uploader(
        "Upload documents",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    col_doc1, col_doc2, col_doc3 = st.columns(3)
    with col_doc1:
        doc_chunk_size = st.number_input("Chunk size (words)", 50, 500, 200)
    with col_doc2:
        doc_overlap = st.number_input("Overlap (words)", 0, 100, 40)
    with col_doc3:
        doc_priority = st.select_slider(
            "Priority", options=[0.5, 1.0, 1.5, 2.0], value=1.0,
            key="doc_priority"
        )

    if uploaded:
        for f in uploaded:
            st.markdown(f"**{f.name}** ({f.size // 1024} KB)")

            if st.button(f"➕ Queue {f.name}", key=f"queue_{f.name}"):
                content = ""

                if f.name.endswith(".pdf"):
                    try:
                        import PyPDF2
                        reader = PyPDF2.PdfReader(f)
                        content = " ".join(
                            page.extract_text() or ""
                            for page in reader.pages
                        )
                    except ImportError:
                        st.error("PyPDF2 not installed. "
                                 "Run: pip install PyPDF2")
                        continue
                else:
                    content = f.read().decode("utf-8", errors="ignore")

                chunks = chunk_text(content, doc_chunk_size, doc_overlap)

                for chunk in chunks:
                    st.session_state.queue.append(
                        (chunk, doc_priority, f.name))

                st.success(f"✓ Queued {len(chunks)} chunks from {f.name}")

    # Paste text directly
    st.divider()
    st.markdown("**Or paste text directly:**")
    paste_text = st.text_area(
        "Paste document text",
        height=200,
        placeholder="Paste any text here — articles, papers, notes...",
        label_visibility="collapsed"
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("➕ Queue Pasted Text",
                     use_container_width=True,
                     disabled=not paste_text.strip()):
            chunks = chunk_text(paste_text.strip(),
                               doc_chunk_size, doc_overlap)
            for chunk in chunks:
                st.session_state.queue.append(
                    (chunk, doc_priority, "paste"))
            st.success(f"✓ Queued {len(chunks)} chunks")

    with col_p2:
        if st.button("🚀 Train Pasted Now",
                     use_container_width=True,
                     disabled=not paste_text.strip()):
            chunks = chunk_text(paste_text.strip(),
                               doc_chunk_size, doc_overlap)
            archived_count = 0
            progress = st.progress(0)

            for i, chunk in enumerate(chunks):
                raw_input = text_to_phase(chunk)
                result = run_stroboscopic(
                    soul, raw_input, chunk,
                    tau_threshold, max_flashes, doc_priority
                )
                st.session_state.training_log.append(result)
                log_training(result)

                if result["status"] == "archived":
                    archived_count += 1

                progress.progress((i + 1) / len(chunks))

            save_soul(soul)
            st.success(f"✓ Trained on {len(chunks)} chunks. "
                       f"Archived: {archived_count}")

# ═════════════════════════════════════════════════════════════════════
# TAB 4 — Q&A
# ═════════════════════════════════════════════════════════════════════

with tab_qa:
    st.markdown("### 💬 Question & Answer Training")
    st.markdown("Train the soul by framing knowledge as Q&A pairs. "
                "Questions and answers are injected together for richer context.")

    qa_pairs = st.session_state.get("qa_pairs", [])
    st.session_state.qa_pairs = qa_pairs

    col_q, col_a = st.columns(2)
    with col_q:
        qa_question = st.text_area(
            "Question",
            placeholder="e.g. What is coherence?",
            height=100,
            key="qa_q"
        )
    with col_a:
        qa_answer = st.text_area(
            "Answer",
            placeholder="e.g. Coherence is the condition in which...",
            height=100,
            key="qa_a"
        )

    qa_priority = st.select_slider(
        "Q&A Priority",
        options=[0.5, 1.0, 1.5, 2.0, 3.0],
        value=1.5,
        help="Q&A pairs usually deserve higher priority than raw text",
        key="qa_priority"
    )

    col_qa1, col_qa2, col_qa3 = st.columns(3)

    with col_qa1:
        if st.button("🚀 Train Now",
                     use_container_width=True,
                     disabled=not (qa_question.strip()
                                   and qa_answer.strip())):
            # Train question and answer separately, then together
            texts = [
                qa_question.strip(),
                qa_answer.strip(),
                f"Q: {qa_question.strip()} A: {qa_answer.strip()}"
            ]
            results = []
            for text in texts:
                raw_input = text_to_phase(text)
                result = run_stroboscopic(
                    soul, raw_input, text,
                    tau_threshold, max_flashes, qa_priority
                )
                results.append(result)
                st.session_state.training_log.append(result)
                log_training(result)

            archived = sum(1 for r in results if r["status"] == "archived")
            st.success(f"✓ Trained Q&A pair. "
                       f"Archived {archived}/3 items.")

            # Store pair for reference
            qa_pairs.append({
                "question":  qa_question.strip(),
                "answer":    qa_answer.strip(),
                "timestamp": datetime.utcnow().isoformat(),
                "archived":  archived,
            })

    with col_qa2:
        if st.button("➕ Add to Queue",
                     use_container_width=True,
                     disabled=not (qa_question.strip()
                                   and qa_answer.strip())):
            texts = [
                qa_question.strip(),
                qa_answer.strip(),
                f"Q: {qa_question.strip()} A: {qa_answer.strip()}"
            ]
            for text in texts:
                st.session_state.queue.append(
                    (text, qa_priority, "qa"))
            st.success("Added Q&A to queue (3 items)")

    # Q&A History
    if qa_pairs:
        st.divider()
        st.markdown("**Q&A History:**")
        df_qa = pd.DataFrame(qa_pairs)
        st.dataframe(
            df_qa[["question", "answer", "archived", "timestamp"]],
            use_container_width=True
        )

    # Bulk Q&A via CSV
    st.divider()
    st.markdown("**Bulk Q&A from CSV:**")
    st.caption("CSV format: two columns — `question`, `answer`")

    qa_file = st.file_uploader("Upload Q&A CSV",
                               type=["csv"],
                               key="qa_csv")

    if qa_file and st.button("📥 Load & Queue CSV"):
        df_upload = pd.read_csv(qa_file)

        if "question" in df_upload.columns and "answer" in df_upload.columns:
            count = 0
            for _, row in df_upload.iterrows():
                q = str(row["question"]).strip()
                a = str(row["answer"]).strip()
                if q and a:
                    for text in [q, a, f"Q: {q} A: {a}"]:
                        st.session_state.queue.append(
                            (text, qa_priority, "csv"))
                    count += 1

            st.success(f"✓ Queued {count} Q&A pairs ({count * 3} items)")
        else:
            st.error("CSV must have 'question' and 'answer' columns")

# ═════════════════════════════════════════════════════════════════════
# TAB 5 — REVIEW QUEUE
# ═════════════════════════════════════════════════════════════════════

with tab_review:
    st.markdown("### 👁️ Manual Review Queue")

    pending = st.session_state.pending_review

    if not pending:
        st.info("No concepts pending review. "
                "Use 'Review first' mode in Inject tab.")
    else:
        st.markdown(f"**{len(pending)} concept(s) waiting for review:**")

        to_remove = []

        for i, item in enumerate(pending):
            with st.container():
                col_t, col_s, col_b = st.columns([3, 1, 1])

                with col_t:
                    st.markdown(f"**#{i+1}** [{item['source']}]")
                    st.text(item["text"][:300])
                    st.caption(f"Preview — Score: {item['score']:.4f} | "
                               f"Curvature: {item['curvature']:.4f} | "
                               f"Priority: {item['priority']}")

                with col_s:
                    score_color = (
                        "🟢" if item["score"] >= tau_threshold
                        else "🟡" if item["score"] >= 0.6
                        else "🔴"
                    )
                    st.markdown(f"### {score_color}")
                    st.caption(f"{item['score']:.4f}")

                with col_b:
                    if st.button("✅ Train",
                                key=f"approve_{i}",
                                use_container_width=True):
                        raw_input = text_to_phase(item["text"])
                        result = run_stroboscopic(
                            soul, raw_input, item["text"],
                            tau_threshold, max_flashes, item["priority"]
                        )
                        st.session_state.training_log.append(result)
                        log_training(result)
                        to_remove.append(i)

                        if result["status"] == "archived":
                            st.success(f"✓ Archived! "
                                      f"Score={result['score']:.3f}")
                        else:
                            st.warning(f"Not archived. "
                                      f"Score={result['score']:.3f}")

                    if st.button("❌ Skip",
                                key=f"reject_{i}",
                                use_container_width=True):
                        to_remove.append(i)
                        log_training({
                            "text":   item["text"],
                            "status": "rejected_manual",
                            "timestamp": datetime.utcnow().isoformat(),
                        })

                st.divider()

        # Remove processed items (reverse to preserve indices)
        for i in sorted(set(to_remove), reverse=True):
            pending.pop(i)

        if to_remove:
            st.rerun()

# ═════════════════════════════════════════════════════════════════════
# TAB 6 — HISTORY
# ═════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("### 📚 Archived Concepts")

    if not soul["concepts"]:
        st.info("No concepts archived yet.")
    else:
        # Build dataframe
        n = len(soul["concepts"])
        df = pd.DataFrame({
            "Index":     range(n),
            "Concept":   soul["concepts"],
            "Coherence": [f"{s:.4f}" for s in soul["coherence_log"][:n]],
            "Curvature": [f"{c:.4f}" for c in soul["curvature_log"][:n]],
            "τ":         [f"{t:.4f}" for t in soul["tau_log"][:n]]
                          if soul["tau_log"] else ["N/A"] * n,
        })

        # Search
        search = st.text_input("🔍 Search concepts", "")
        if search:
            mask = df["Concept"].str.contains(search, case=False, na=False)
            df = df[mask]

        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        # Download
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "⬇️ Export CSV",
                csv_data,
                "soul_concepts.csv",
                "text/csv",
                use_container_width=True
            )

        with col_dl2:
            json_data = json.dumps(soul["concepts"], indent=2)
            st.download_button(
                "⬇️ Export JSON",
                json_data,
                "soul_concepts.json",
                "application/json",
                use_container_width=True
            )

# ═════════════════════════════════════════════════════════════════════
# TAB 7 — METRICS
# ═════════════════════════════════════════════════════════════════════

with tab_metrics:
    st.markdown("### 📊 Training Metrics")

    if not soul["coherence_log"]:
        st.info("No training data yet.")
    else:
        # Coherence over time
        fig_coh = go.Figure()
        fig_coh.add_trace(go.Scatter(
            y=soul["coherence_log"],
            mode="lines",
            line=dict(color="lime", width=1.5),
            name="Coherence",
            fill="tozeroy",
            fillcolor="rgba(0,255,0,0.05)"
        ))
        fig_coh.add_hline(
            y=tau_threshold,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"τ threshold ({tau_threshold})"
        )
        fig_coh.update_layout(
            title="Coherence over training",
            plot_bgcolor="#0a0a1a",
            paper_bgcolor="#0a0a1a",
            font=dict(color="white"),
            height=280,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_coh, use_container_width=True)

        # Curvature over time
        fig_curv = go.Figure()
        fig_curv.add_trace(go.Scatter(
            y=soul["curvature_log"],
            mode="lines",
            line=dict(color="orange", width=1.5),
            name="Curvature",
            fill="tozeroy",
            fillcolor="rgba(255,165,0,0.05)"
        ))
        fig_curv.add_hline(
            y=0.15,
            line_dash="dash",
            line_color="red",
            annotation_text="curvature threshold (0.15)"
        )
        fig_curv.update_layout(
            title="Curvature (lower = more stable)",
            plot_bgcolor="#0a0a1a",
            paper_bgcolor="#0a0a1a",
            font=dict(color="white"),
            height=280,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_curv, use_container_width=True)

        # Proper time
        if soul["tau_log"]:
            fig_tau = go.Figure()
            fig_tau.add_trace(go.Scatter(
                y=soul["tau_log"],
                mode="lines",
                line=dict(color="cyan", width=1.5),
                name="τ (proper time)"
            ))
            fig_tau.update_layout(
                title="Proper time τ — arc length of soul trajectory",
                plot_bgcolor="#0a0a1a",
                paper_bgcolor="#0a0a1a",
                font=dict(color="white"),
                height=240,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_tau, use_container_width=True)

        # Summary stats
        st.divider()
        st.markdown("**Summary**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total archived", len(soul["concepts"]))
        c2.metric("Mean coherence",
                  f"{np.mean(soul['coherence_log']):.4f}")
        c3.metric("Mean curvature",
                  f"{np.mean(soul['curvature_log']):.4f}")
        c4.metric("Final τ", f"{soul['tau']:.4f}")

        # Session training log
        if st.session_state.training_log:
            st.divider()
            st.markdown("**Session log (last 20):**")
            recent = st.session_state.training_log[-20:][::-1]
            df_log = pd.DataFrame(recent)[
                ["status", "score", "curvature", "flashes", "text"]
            ].copy()
            df_log["text"] = df_log["text"].str[:60]
            st.dataframe(df_log, use_container_width=True)
