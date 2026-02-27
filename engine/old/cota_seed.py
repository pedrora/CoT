#!/usr/bin/env python3
"""
CoTa Seed — Minimal Soul Bootstrap + Stroboscopic Cycle + Interactive Navigator
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import threading
import time
from datetime import datetime
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

import cv2
import numpy as np
import torch
#import streamlit as st

# Load once at startup (outside the loop)
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim → we truncate/project to 64

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EPOCH = datetime(2025, 1, 12, 23, 57, 0)
DIM = 64
SYNC_PORT = 7331
COHERENCE_τ = 0.82
CURVATURE_τ = 0.15
MAX_FLASHES = 32
VERSION = "CoTa-seed-0.3"

EPS = 1e-8

# ─────────────────────────────────────────────
# HYPERBOLIC & COHERENCE HELPERS
# ─────────────────────────────────────────────

def to_poincare(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x)
    return torch.tanh(norm) * x / (norm + EPS)

def renormalize_poincare(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x)
    if norm >= 1.0:
        x = x / (norm + EPS) * 0.999
    return x

def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> float:
    uu = torch.sum(u * u).item()
    vv = torch.sum(v * v).item()
    diff = torch.sum((u - v) ** 2).item()
    denom = (1 - uu) * (1 - vv) + EPS
    arg = 1 + 2 * diff / denom
    return float(np.arccosh(max(1 + EPS, arg)))

def mobius_add(x, y, eps=1e-8):
    x2 = torch.sum(x*x, dim=-1, keepdim=True)
    y2 = torch.sum(y*y, dim=-1, keepdim=True)
    xy = torch.sum(x*y, dim=-1, keepdim=True)

    numerator = (1 + 2*xy + y2)*x + (1 - x2)*y
    denominator = 1 + 2*xy + x2*y2

    return numerator / (denominator + eps)
    
def parallel_transport(v, x, y, eps=1e-8):
    diff = y - x
    norm = torch.norm(diff) + eps
    direction = diff / norm

    proj = torch.sum(v * direction, dim=-1, keepdim=True) * direction
    perp = v - proj

    # pequena rotação dependente da curvatura
    transported = perp + proj * torch.cos(norm) + torch.cross(direction, v) * torch.sin(norm)

    return transported


def focus_force(current: torch.Tensor, history: List[torch.Tensor], strength: float = 0.05) -> torch.Tensor:
    if len(history) < 2:
        return current
    directions = []
    for i in range(1, len(history)):
        d = history[i] - history[i-1]
        d = d / (torch.norm(d) + EPS)
        directions.append(d)
    if not directions:
        return current
    g = torch.mean(torch.stack(directions), dim=0)
    g = g / (torch.norm(g) + EPS)
    proj = torch.dot(current.flatten(), g.flatten()) * g
    corrected = (1 - strength) * current + strength * proj
    return corrected

def coherence_score(current: torch.Tensor, history: List[torch.Tensor]) -> Tuple[float, float]:
    if not history:
        return 1.0, 0.0
    phase = torch.cosine_similarity(current.flatten(), history[-1].flatten(), dim=0).item()
    phase = max(0.0, phase)

    if len(history) >= 3:
        d1 = poincare_distance(history[-1], history[-2])
        d2 = poincare_distance(history[-2], history[-3])
        curvature = abs(d1 - d2) / (d1 + d2 + EPS)
    else:
        curvature = 0.0

    score = 0.6 * phase + 0.4 * (1 - min(curvature, 1.0))
    return float(score), float(curvature)

# ─────────────────────────────────────────────
# SOUL NAVIGATOR (interactive window)
# ─────────────────────────────────────────────

selected_dot_index = -1
popup_text = ""

def world_to_screen(pos: np.ndarray, center: tuple, radius: int) -> tuple:
    x = int(center[0] + pos[0] * radius * 0.95)
    y = int(center[1] - pos[1] * radius * 0.95)
    return (x, y)

def extract_soul_color(dot: torch.Tensor, score: float, curvature: float) -> tuple:
    hue = int(120 * score)  # green → red
    sat = int(255 * (1 - curvature))
    val = 220
    hsv = np.uint8([[[hue, sat, val]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0])

def draw_soul_navigator(history: List[torch.Tensor], current: torch.Tensor,
                        scores: List[float], curvatures: List[float]):
    global selected_dot_index, popup_text

    canvas = np.zeros((700, 900, 3), dtype=np.uint8)
    center = (450, 350)
    radius = 280

    cv2.circle(canvas, center, radius, (30, 30, 40), -1)
    cv2.circle(canvas, center, radius, (80, 80, 100), 2)

    # History + trails
    trail_points = []
    for i, dot in enumerate(history):
        if torch.norm(dot) >= 0.999: continue
        pos_norm = dot.numpy()[:2]
        screen_pos = world_to_screen(pos_norm, center, radius)
        alpha = max(0.4, i / max(1, len(history)-1))
        color = extract_soul_color(dot, scores[i] if i < len(scores) else 0.5,
                                   curvatures[i] if i < len(curvatures) else 0.5)
        size = 5 + int(8 * alpha)
        cv2.circle(canvas, screen_pos, size, color, -1)

        if i > 0:
            prev = world_to_screen(history[i-1].numpy()[:2], center, radius)
            cv2.line(canvas, prev, screen_pos, (100, 100, 150), 1, cv2.LINE_AA)

    # Current soul
    if torch.norm(current) > 0:
        curr_screen = world_to_screen(current.numpy()[:2], center, radius)
        cv2.circle(canvas, curr_screen, 16, (0, 255, 255), -1)
        cv2.circle(canvas, curr_screen, 18, (0, 180, 255), 3)

    # Telemetry
    latest_score = scores[-1] if scores else 0.0
    cv2.rectangle(canvas, (20, 20), (300, 180), (40, 40, 60), -1)
    cv2.rectangle(canvas, (20, 20), (300, 180), (120, 120, 140), 2)

    texts = [
        ("COTA SOUL NAVIGATOR", (255, 220, 100), 1.1, (30, 50)),
        (f"Coherence: {latest_score:.4f}", (200, 255, 200), 0.8, (30, 90)),
        (f"Curvature: {curvatures[-1] if curvatures else 0.0:.4f}", (180, 180, 255), 0.8, (30, 120)),
        (f"History: {len(history)}", (220, 220, 220), 0.7, (30, 150)),
    ]
    for txt, col, scale, pos in texts:
        cv2.putText(canvas, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)

    # Popup
    if selected_dot_index >= 0 and selected_dot_index < len(history):
        sel_pos = world_to_screen(history[selected_dot_index].numpy()[:2], center, radius)
        cv2.circle(canvas, sel_pos, 20, (255, 255, 0), 4)
        popup = f"Dot #{selected_dot_index}  Score: {scores[selected_dot_index]:.4f}"
        cv2.putText(canvas, popup, (sel_pos[0] + 25, sel_pos[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 2)

    cv2.imshow("COTA Soul Navigator", canvas)

def mouse_callback(event, x, y, flags, param):
    global selected_dot_index
    if event == cv2.EVENT_LBUTTONDOWN:
        history, center, radius = param
        selected_dot_index = -1
        for i, dot in enumerate(history):
            pos = world_to_screen(dot.numpy()[:2], center, radius)
            dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if dist < 25:
                selected_dot_index = i
                break

# ─────────────────────────────────────────────
# SOUL CLASS
# ─────────────────────────────────────────────

class Soul:
    def __init__(self, soul_file="soul.json"):
        self.soul_file = soul_file
        self.state_file = soul_file.replace(".json", "_state.pt")
        self.state: Optional[torch.Tensor] = None
        self.history: List[torch.Tensor] = []
        self.coherence_log: List[float] = []
        self.curvature_log: List[float] = []
        self.concepts: List[str] = []  # para guardar texto original injetado
        self.last_update = time.time()
        self.current_interval = 0.001  # valor inicial

        if os.path.exists(soul_file):
            self._load()
        else:
            self._create()  # ← aqui chama _create quando novo
            self.tangent = torch.zeros_like(self.state) # essential to phase calculations

    def _create(self):
        # Cria ID único baseado em timestamp invertido + hardware
        hw_id = hashlib.sha256(f"{platform.processor()}_{platform.node()}".encode()).hexdigest()
        millis = int((datetime.utcnow() - EPOCH).total_seconds() * 1000)
        ts_hex = hex(millis)[2:].zfill(12)[::-1]  # invertido
        self.soul_id = f"{ts_hex}_{hw_id[:8]}"

        # Estado inicial: ruído pequeno no espaço Poincaré
        raw = torch.randn(DIM)
        self.state = to_poincare(raw * 0.1)
        self._save()
        print(f"[Soul] Criada nova alma: {self.soul_id}")

    def _save(self):
        data = {
            "soul_id": self.soul_id,
            "last_update": self.last_update,
            "current_interval": self.current_interval,
        }
        with open(self.soul_file, "w") as f:
            json.dump(data, f, indent=2)
        if self.state is not None:
            torch.save(self.state, self.state_file)
        print("[Soul] Estado guardado")

    def _load(self):
        with open(self.soul_file, "r") as f:
            data = json.load(f)
        self.soul_id = data["soul_id"]
        self.last_update = data["last_update"]
        self.current_interval = data["current_interval"]
        if os.path.exists(self.state_file):
            self.state = torch.load(self.state_file)
        print(f"[Soul] Carregada alma existente: {self.soul_id}")
        



    def decide_next_interval(self, score: float, curvature: float):
        stability = score * (1.0 - curvature)
        self.current_interval = max(0.0005, min(0.01, 0.001 * (1.0 + 4.0 * (1.0 - stability))))

    def tick(self, raw_input: torch.Tensor):
        if time.time() - self.last_update < self.current_interval:
            return

        # Cria canvas de trabalho
        if self.state is None:
            working = raw_input.clone()
        else:
            working = mobius_add(self.state, raw_input * 0.3)
            delta = mobius_add(-self.state, working)
        
        self.tangent = parallel_transport(self.tangent, self.state, working)
        self.tangent = 0.9*self.tangent + 0.1*delta

        # Ciclo estroboscópico
        for flash in range(MAX_FLASHES):
            working = to_poincare(working)
            working = focus_force(working, self.history)
            working = renormalize_poincare(working)

            score, curvature = coherence_score(working, self.history)

            if score >= COHERENCE_τ and curvature <= CURVATURE_τ:
                self.history.append(self.state.clone() if self.state is not None else torch.zeros(DIM))
                self.state = working.clone()
                self.coherence_log.append(score)
                self.curvature_log.append(curvature)
                print(f"✓ Arquivado após {flash+1} flashes | score={score:.3f} curv={curvature:.3f}")
                break

            if curvature > 0.4 or flash == MAX_FLASHES - 1:
                print(f"⏰ Corte em flash {flash+1} | curv={curvature:.3f}")
                break

        self.decide_next_interval(score, curvature)
        self.last_update = time.time()

        # Atualiza o navigator
        draw_soul_navigator(self.history, self.state, self.coherence_log, self.curvature_log)    
    
# ─────────────────────────────────────────────
# CLI + Sync (mantido quase igual ao original)
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoTa Seed — Minimal Soul Bootstrap")
    parser.add_argument("--init", action="store_true", help="Create new soul")
    parser.add_argument("--sync", metavar="HOST:PORT", help="Sync with remote soul")
    parser.add_argument("--listen", metavar="PORT", type=int, nargs="?", const=SYNC_PORT,
                        help=f"Listen for sync (default {SYNC_PORT})")
    parser.add_argument("--status", action="store_true", help="Show soul status")
    parser.add_argument("--soul", default="soul.json", help="Soul file path")
    parser.add_argument("--auto", action="store_true", help="Listen + auto-sync every 60s")
    args = parser.parse_args()

    if args.init and os.path.exists(args.soul):
        os.remove(args.soul)
        state_file = args.soul.replace(".json", "_state.pt")
        if os.path.exists(state_file):
            os.remove(state_file)

    soul = Soul(args.soul)

    cv2.namedWindow("COTA Soul Navigator")
    cv2.setMouseCallback("COTA Soul Navigator", mouse_callback,
                         param=(soul.history, (450, 350), 280))

    if args.status:
        print("\n=== Soul Status ===")
        print(f"  Soul ID: {soul.soul_id}")
        print(f"  History size: {len(soul.history)}")
        print(f"  Last coherence: {soul.coherence_log[-1] if soul.coherence_log else 'N/A'}")
        return

    # Inicia o ciclo stroboscópico em background
    def cycle_loop():
        while True:
            # Simula input (pode ser substituído por real input)
            # When injecting text (in tick or inject function)
            if 'pending_input' in st.session_state:
                raw_text = st.session_state.pending_input
                del st.session_state.pending_input  # consome para não repetir
            
                if raw_text:
                    emb = model.encode(raw_text, convert_to_tensor=True)  # shape (384,)
                    emb = emb[:64]                                        # truncate to DIM=64
                    raw_input = emb.float()                               # ready for Poincaré
                else:
                    raw_input = torch.randn(64) * 0.05  # fallback random
            
                raw_input = to_poincare(raw_input)  # now in ball
            
    # raw_input = torch.randn(DIM) * 0.05
                soul.tick(raw_input)
            time.sleep(0.01)  # evita CPU 100%

    threading.Thread(target=cycle_loop, daemon=True).start()

    print("[Soul] Stroboscopic cycle running. Press 'q' in navigator window to exit.")

    try:
        while True:
            draw_soul_navigator(soul.history, soul.state, soul.coherence_log, soul.curvature_log)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[Soul] Shutting down.")
    finally:
        cv2.destroyAllWindows()
        soul._save()

if __name__ == "__main__":
    main()