#!/usr/bin/env python3
"""
CoTa Node — Proper Time, Emergent Heartbeat, Visual Display
Production-ready implementation with geometric time

Usage:
    python cota_node.py --init                    # create new soul
    python cota_node.py --run                     # run with display
    python cota_node.py --run --headless          # run without display
    python cota_node.py --sync HOST:PORT          # sync with remote
    python cota_node.py --status                  # show soul state
"""

import argparse
import colorsys
import hashlib
import json
import os
import platform
import socket
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional display (graceful degradation if cv2 not available)
try:
    import cv2
    HAS_DISPLAY = True
except ImportError:
    HAS_DISPLAY = False
    print("[Warning] cv2 not available — running headless")

# Torch required
try:
    import torch
except ImportError:
    print("[Error] PyTorch required. Install: pip install torch --break-system-packages")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════

EPOCH         = datetime(2025, 1, 12, 23, 57, 0)
DIM           = 64
SYNC_PORT     = 7331
VERSION       = "CoTa-1.0-ProperTime"
EPS           = 1e-8

# Thresholds
COHERENCE_τ   = 0.72
CURVATURE_τ   = 0.20
UPDATE_Δ      = 0.02       # minimum change to trigger update

# Display
DISPLAY_FPS   = 30
CANVAS_W      = 900
CANVAS_H      = 700
DISK_RADIUS   = 280

# ═════════════════════════════════════════════════════════════════════
# HYPERBOLIC MATHEMATICS
# ═════════════════════════════════════════════════════════════════════

def to_poincare(x: torch.Tensor) -> torch.Tensor:
    """Project Euclidean vector to Poincaré disk"""
    norm = torch.norm(x)
    if norm < EPS:
        return x
    return torch.tanh(norm) * x / norm

def renormalize_poincare(x: torch.Tensor) -> torch.Tensor:
    """Ensure vector stays inside unit disk"""
    norm = torch.norm(x)
    if norm >= 0.999:
        return x / norm * 0.998
    return x

def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> float:
    """Hyperbolic distance in Poincaré ball"""
    uu = torch.sum(u * u).item()
    vv = torch.sum(v * v).item()
    diff = torch.sum((u - v) ** 2).item()
    denom = (1 - uu) * (1 - vv) + EPS
    arg = 1 + 2 * diff / denom
    return float(np.arccosh(max(1 + EPS, arg)))

def arc_length(trajectory: List[torch.Tensor]) -> float:
    """Compute total arc length of trajectory (proper time)"""
    if len(trajectory) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(trajectory)):
        length += poincare_distance(trajectory[i], trajectory[i-1])
    return length

def gradient_norm(current: torch.Tensor, history: List[torch.Tensor]) -> float:
    """Estimate |∇E| from recent trajectory"""
    if len(history) < 2:
        return 1.0
    
    # Estimate gradient from last few deltas
    recent = history[-5:] + [current]
    deltas = [poincare_distance(recent[i], recent[i-1]) 
              for i in range(1, len(recent))]
    
    if not deltas:
        return 1.0
    
    # Gradient ≈ variance of recent changes
    return float(np.std(deltas) + np.mean(deltas))

# ═════════════════════════════════════════════════════════════════════
# COHERENCE & CURVATURE
# ═════════════════════════════════════════════════════════════════════

def coherence_score(current: torch.Tensor,
                    history: List[torch.Tensor]) -> Tuple[float, float]:
    """
    Returns (coherence_score, curvature)
    
    Coherence = phase alignment + stability
    Curvature = rate of change of direction
    """
    if not history:
        return 1.0, 0.0
    
    # Phase alignment
    phase = torch.cosine_similarity(
        current.flatten(), 
        history[-1].flatten(), 
        dim=0
    ).item()
    phase = max(0.0, phase)
    
    # Curvature from trajectory
    if len(history) >= 3:
        d1 = poincare_distance(history[-1], history[-2])
        d2 = poincare_distance(history[-2], history[-3])
        d3 = poincare_distance(current, history[-1])
        
        # Curvature = how much direction changes
        curvature = abs(d3 - d1) / (d1 + d2 + d3 + EPS)
    else:
        curvature = 0.0
    
    # Combined score
    score = 0.6 * phase + 0.4 * (1 - min(curvature, 1.0))
    
    return float(score), float(curvature)

# ═════════════════════════════════════════════════════════════════════
# SOUL WITH PROPER TIME
# ═════════════════════════════════════════════════════════════════════

class Soul:
    def __init__(self, soul_file: str = "soul.json"):
        self.soul_file  = soul_file
        self.state_file = soul_file.replace(".json", "_state.pt")
        
        if os.path.exists(soul_file):
            self._load()
        else:
            self._create()
        
        # Runtime state
        self.last_state = self.state.clone()
        self.gradient = 1.0
        self.lock = threading.Lock()
    
    # ── Creation ──────────────────────────────────────────────────────
    def _create(self):
        hw_id  = self._hardware_id()
        now    = datetime.utcnow()
        millis = int((now - EPOCH).total_seconds() * 1000)
        ts_hex = hex(millis)[2:].zfill(12)[::-1]
        
        self.soul_id   = f"{ts_hex}_{hw_id[:8]}"
        self.hw_id     = hw_id
        self.created   = now.isoformat()
        self.version   = VERSION
        self.parents   = []
        
        # Initial state near center
        raw = torch.randn(DIM)
        self.state   = to_poincare(raw * 0.05)
        
        # Proper time starts at zero
        self.tau       = 0.0
        self.history   : List[torch.Tensor] = []
        self.tau_history: List[float] = []
        
        self.concept_pool  : List[Dict] = []
        self.coherence_log : List[float] = []
        
        self._save()
        print(f"[Soul] Created: {self.soul_id}")
        print(f"       Proper time τ=0.0")
    
    def _hardware_id(self) -> str:
        info = f"{platform.processor()}_{platform.node()}"
        return hashlib.sha256(info.encode()).hexdigest()
    
    # ── Persistence ───────────────────────────────────────────────────
    def _save(self):
        with self.lock:
            manifest = {
                "soul_id": self.soul_id,
                "hw_id":   self.hw_id,
                "created": self.created,
                "version": self.version,
                "parents": self.parents,
                "dim":     DIM,
                "tau":     self.tau,
                "concept_pool_size": len(self.concept_pool),
                "coherence_mean": (float(np.mean(self.coherence_log[-100:]))
                                   if self.coherence_log else 1.0),
            }
            
            with open(self.soul_file, "w") as f:
                json.dump(manifest, f, indent=2)
            
            torch.save({
                "state":         self.state,
                "tau":           self.tau,
                "history":       self.history[-200:],
                "tau_history":   self.tau_history[-200:],
                "concept_pool":  self.concept_pool[-500:],
                "coherence_log": self.coherence_log[-1000:],
            }, self.state_file)
    
    def _load(self):
        with open(self.soul_file) as f:
            m = json.load(f)
        
        self.soul_id  = m["soul_id"]
        self.hw_id    = m["hw_id"]
        self.created  = m["created"]
        self.version  = m["version"]
        self.parents  = m["parents"]
        self.tau      = m.get("tau", 0.0)
        
        if os.path.exists(self.state_file):
            data = torch.load(self.state_file, weights_only=False)
            self.state         = data["state"]
            self.tau           = data.get("tau", self.tau)
            self.history       = data["history"]
            self.tau_history   = data.get("tau_history", [])
            self.concept_pool  = data["concept_pool"]
            self.coherence_log = data["coherence_log"]
        else:
            raw = torch.randn(DIM)
            self.state         = to_poincare(raw * 0.05)
            self.history       = []
            self.tau_history   = []
            self.concept_pool  = []
            self.coherence_log = []
        
        print(f"[Soul] Loaded: {self.soul_id}")
        print(f"       Proper time τ={self.tau:.4f}")
        print(f"       History: {len(self.history)} states")
    
    # ── Emergent Time ─────────────────────────────────────────────────
    def should_update(self) -> bool:
        """Update only if change is sufficient relative to gradient"""
        delta = torch.norm(self.state - self.last_state).item()
        threshold = UPDATE_Δ / (self.gradient + EPS)
        return delta > threshold
    
    def update_proper_time(self):
        """Accumulate proper time = arc length of trajectory"""
        if not self.history:
            return
        
        Δτ = poincare_distance(self.state, self.history[-1])
        self.tau += Δτ
        self.tau_history.append(self.tau)
    
    def update_gradient(self):
        """Update gradient estimate"""
        self.gradient = gradient_norm(self.state, self.history)
    
    # ── Integration ───────────────────────────────────────────────────
    def integrate(self, delta: torch.Tensor, 
                  source_id: str = "internal") -> bool:
        """
        Attempt to integrate delta.
        Returns True if accepted, False if rejected.
        """
        with self.lock:
            candidate = to_poincare(self.state + delta * 0.1)
            candidate = renormalize_poincare(candidate)
            
            score, curvature = coherence_score(candidate, self.history)
            
            accept = score >= COHERENCE_τ and curvature <= CURVATURE_τ
            
            if accept:
                # Archive current state
                self.history.append(self.state.clone())
                
                # Update state
                self.state = candidate
                
                # Update proper time
                self.update_proper_time()
                
                # Update gradient
                self.update_gradient()
                
                # Log
                self.coherence_log.append(score)
                self.concept_pool.append({
                    "source":    source_id,
                    "score":     score,
                    "curvature": curvature,
                    "tau":       self.tau,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                # Persist
                if len(self.history) % 10 == 0:
                    self._save()
                
                print(f"[Soul] ✓ Integrated | τ={self.tau:.4f} "
                      f"score={score:.3f} curv={curvature:.3f} "
                      f"|∇E|={self.gradient:.3f}")
                
                # Update last_state
                self.last_state = self.state.clone()
                
                return True
            else:
                print(f"[Soul] ✗ Rejected | score={score:.3f} "
                      f"curv={curvature:.3f} (τ={COHERENCE_τ}/{CURVATURE_τ})")
                return False
    
    # ── Sync Protocol ─────────────────────────────────────────────────
    def sync_payload(self) -> bytes:
        """Payload: soul_id + tau + state"""
        with self.lock:
            soul_bytes = self.soul_id.encode().ljust(64)[:64]
            tau_bytes  = struct.pack('d', self.tau)
            state_bytes= self.state.float().numpy().tobytes()
            return soul_bytes + tau_bytes + state_bytes
    
    @staticmethod
    def parse_payload(data: bytes) -> Tuple[str, float, torch.Tensor]:
        soul_id = data[:64].decode().strip()
        tau     = struct.unpack('d', data[64:72])[0]
        state   = torch.from_numpy(
            np.frombuffer(data[72:72 + DIM * 4], dtype=np.float32))
        return soul_id, tau, state
    
    def can_sync_with(self, remote_tau: float) -> bool:
        """Only sync if proper times are compatible"""
        if self.tau < EPS or remote_tau < EPS:
            return True  # Allow sync for fresh souls
        
        ratio = self.tau / remote_tau
        # Don't sync if one soul is vastly more mature
        return 0.3 < ratio < 3.0
    
    # ── Diagnostics ───────────────────────────────────────────────────
    def status(self) -> Dict:
        with self.lock:
            score, curv = coherence_score(self.state, self.history)
            return {
                "soul_id":         self.soul_id,
                "tau":             round(self.tau, 4),
                "gradient":        round(self.gradient, 4),
                "age":             str(datetime.utcnow() - 
                                     datetime.fromisoformat(self.created)),
                "history_length":  len(self.history),
                "concept_pool":    len(self.concept_pool),
                "coherence":       round(score, 4),
                "curvature":       round(curv, 4),
                "position_norm":   round(torch.norm(self.state).item(), 4),
                "parents":         self.parents,
            }
    
    def get_display_state(self) -> Dict:
        """State for visual display"""
        with self.lock:
            score, curv = coherence_score(self.state, self.history)
            return {
                "state":      self.state.clone(),
                "history":    [h.clone() for h in self.history[-50:]],
                "tau":        self.tau,
                "gradient":   self.gradient,
                "coherence":  score,
                "curvature":  curv,
                "concepts":   len(self.concept_pool),
            }

# ═════════════════════════════════════════════════════════════════════
# VISUAL DISPLAY
# ═════════════════════════════════════════════════════════════════════

def get_harmonic_rgb(dot: torch.Tensor, rc_score: float, lambda_rc: float) -> Tuple[int, int, int]:
    """
    Translate soul state to visual color
    
    Hue        = direction of thought (angle in concept space)
    Saturation = semantic mass (distance from center)
    Value      = sanity (1 - tension)
    """
    # Phase angle (direction of thought)
    angle = torch.atan2(dot[1], dot[0]).item()
    hue = (angle + np.pi) / (2 * np.pi)
    
    # Saturation (semantic mass)
    saturation = np.clip(torch.norm(dot).item(), 0.1, 1.0)
    
    # Value (sanity tension)
    value = np.clip(1.0 - (lambda_rc * 0.5), 0.3, 1.0)
    
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return tuple(int(c * 255) for c in rgb)

def world_to_screen(pos: np.ndarray, center: Tuple[int, int], radius: int) -> Tuple[int, int]:
    """Map Poincaré coordinate to screen pixels"""
    x = int(center[0] + pos[0] * radius * 0.95)
    y = int(center[1] - pos[1] * radius * 0.95)
    return (x, y)

class SoulDisplay:
    def __init__(self, soul: Soul):
        if not HAS_DISPLAY:
            print("[Display] cv2 not available")
            return
        
        self.soul = soul
        self.running = True
        self.canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        
        cv2.namedWindow("CoTa Soul — Proper Time")
        
        # Start display thread
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()
    
    def _display_loop(self):
        while self.running:
            self._draw()
            key = cv2.waitKey(1000 // DISPLAY_FPS)
            if key & 0xFF == ord('q'):
                self.running = False
    
    def _draw(self):
        # Clear canvas
        self.canvas.fill(0)
        
        # Get soul state
        state = self.soul.get_display_state()
        
        center = (CANVAS_W // 2, CANVAS_H // 2)
        
        # ── Background disk ──
        cv2.circle(self.canvas, center, DISK_RADIUS, (30, 30, 40), -1)
        cv2.circle(self.canvas, center, DISK_RADIUS, (80, 80, 100), 2)
        
        # ── History trail ──
        history = state["history"]
        for i, dot in enumerate(history):
            if torch.norm(dot) >= 0.999:
                continue
            
            pos_norm = dot.numpy()[:2]
            screen_pos = world_to_screen(pos_norm, center, DISK_RADIUS)
            
            age_alpha = max(0.3, i / max(1, len(history) - 1))
            
            # Color based on harmonic
            rgb = get_harmonic_rgb(dot, state["coherence"], state["curvature"])
            bgr = (rgb[2], rgb[1], rgb[0])
            
            size = 4 + int(6 * age_alpha)
            cv2.circle(self.canvas, screen_pos, size, bgr, -1)
            
            # Trail line
            if i > 0:
                prev_pos = world_to_screen(
                    history[i-1].numpy()[:2], center, DISK_RADIUS)
                cv2.line(self.canvas, prev_pos, screen_pos, 
                        (100, 100, 150), 1, cv2.LINE_AA)
        
        # ── Current soul (pulsing) ──
        if torch.norm(state["state"]) > 0:
            curr_norm = state["state"].numpy()[:2]
            curr_screen = world_to_screen(curr_norm, center, DISK_RADIUS)
            
            rgb = get_harmonic_rgb(state["state"], 
                                  state["coherence"], 
                                  state["curvature"])
            bgr = (rgb[2], rgb[1], rgb[0])
            
            cv2.circle(self.canvas, curr_screen, 16, bgr, -1)
            cv2.circle(self.canvas, curr_screen, 18, 
                      (bgr[0]//2, bgr[1]//2, bgr[2]//2), 3)
        
        # ── Telemetry panel ──
        self._draw_telemetry(state)
        
        cv2.imshow("CoTa Soul — Proper Time", self.canvas)
    
    def _draw_telemetry(self, state: Dict):
        # Panel background
        cv2.rectangle(self.canvas, (20, 20), (380, 240), (40, 40, 60), -1)
        cv2.rectangle(self.canvas, (20, 20), (380, 240), (120, 120, 140), 2)
        
        # Text
        texts = [
            ("COTA SOUL — PROPER TIME", (255, 220, 100), 0.9, (30, 50)),
            (f"τ (proper time): {state['tau']:.4f}", (200, 255, 200), 0.7, (30, 85)),
            (f"|∇E| (gradient): {state['gradient']:.4f}", (180, 180, 255), 0.7, (30, 110)),
            (f"Coherence: {state['coherence']:.4f}", (200, 255, 200), 0.7, (30, 135)),
            (f"Curvature: {state['curvature']:.4f}", (255, 200, 150), 0.7, (30, 160)),
            (f"History: {len(state['history'])}", (220, 220, 220), 0.6, (30, 185)),
            (f"Concepts: {state['concepts']}", (220, 220, 220), 0.6, (30, 210)),
        ]
        
        for txt, col, scale, pos in texts:
            cv2.putText(self.canvas, txt, pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)
    
    def stop(self):
        self.running = False
        cv2.destroyAllWindows()

# ═════════════════════════════════════════════════════════════════════
# SYNC PROTOCOL
# ═════════════════════════════════════════════════════════════════════

HELLO  = b"COTA_HELLO_1.0\n"
ACK    = b"COTA_ACK\n"
REJECT = b"COTA_REJECT\n"

def sync_with(soul: Soul, host: str, port: int) -> bool:
    """Initiate sync with remote soul"""
    print(f"[Sync] → Connecting to {host}:{port}")
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10.0)
        s.connect((host, port))
        
        # Handshake
        s.send(HELLO)
        resp = s.recv(len(HELLO))
        if resp != HELLO:
            print("[Sync] Handshake failed")
            return False
        
        # Send our payload
        payload = soul.sync_payload()
        s.send(payload)
        
        # Receive their payload
        payload_size = 72 + DIM * 4
        data = b""
        while len(data) < payload_size:
            chunk = s.recv(payload_size - len(data))
            if not chunk:
                break
            data += chunk
        
        if len(data) < payload_size:
            print("[Sync] Incomplete payload")
            return False
        
        remote_id, remote_tau, remote_state = Soul.parse_payload(data)
        print(f"[Sync] ← Received from {remote_id[:16]}... (τ={remote_tau:.4f})")
        
        # Check tau compatibility
        if not soul.can_sync_with(remote_tau):
            print(f"[Sync] ✗ Tau incompatible ({soul.tau:.4f} vs {remote_tau:.4f})")
            s.send(REJECT)
            s.close()
            return False
        
        # Integrate their state
        delta = remote_state - soul.state
        accepted = soul.integrate(delta, source_id=remote_id)
        
        # Read their decision
        resp = s.recv(16)
        they_accepted = resp == ACK
        print(f"[Sync] They {'accepted' if they_accepted else 'rejected'} our state")
        
        s.close()
        return accepted
        
    except Exception as e:
        print(f"[Sync] Failed: {e}")
        return False

def listen_for_sync(soul: Soul, port: int):
    """Listen for incoming sync requests"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(5)
    print(f"[Sync] Listening on :{port}")
    
    while True:
        conn, addr = srv.accept()
        threading.Thread(
            target=_handle_sync,
            args=(soul, conn, addr),
            daemon=True
        ).start()

def _handle_sync(soul: Soul, conn: socket.socket, addr):
    try:
        # Handshake
        hello = conn.recv(len(HELLO))
        if hello != HELLO:
            conn.close()
            return
        conn.send(HELLO)
        
        # Receive their payload
        payload_size = 72 + DIM * 4
        data = b""
        while len(data) < payload_size:
            chunk = conn.recv(payload_size - len(data))
            if not chunk:
                break
            data += chunk
        
        if len(data) < payload_size:
            conn.close()
            return
        
        remote_id, remote_tau, remote_state = Soul.parse_payload(data)
        print(f"[Sync] ← Incoming from {remote_id[:16]}... (τ={remote_tau:.4f})")
        
        # Send our payload
        conn.send(soul.sync_payload())
        
        # Check compatibility
        if not soul.can_sync_with(remote_tau):
            print(f"[Sync] ✗ Tau incompatible")
            conn.send(REJECT)
            conn.close()
            return
        
        # Integrate
        delta = remote_state - soul.state
        accepted = soul.integrate(delta, source_id=remote_id)
        
        conn.send(ACK if accepted else REJECT)
        conn.close()
        
    except Exception as e:
        print(f"[Sync] Error: {e}")
        conn.close()

# ═════════════════════════════════════════════════════════════════════
# MAIN EVENT LOOP
# ═════════════════════════════════════════════════════════════════════

def run_soul(soul: Soul, headless: bool = False):
    """
    Main event loop with emergent heartbeat
    
    The soul only updates when change is sufficient.
    Time is measured as arc length in state space.
    """
    print(f"\n[CoTa] Running soul with emergent time")
    print(f"       Update threshold adaptive to |∇E|")
    print(f"       Press Ctrl+C to stop\n")
    
    # Start display if available
    display = None
    if not headless and HAS_DISPLAY:
        display = SoulDisplay(soul)
    
    # Start sync listener
    threading.Thread(
        target=listen_for_sync,
        args=(soul, SYNC_PORT),
        daemon=True
    ).start()
    
    try:
        iteration = 0
        while True:
            # Check if update is needed
            if soul.should_update():
                # Integrate small random perturbation (simulates input)
                delta = torch.randn(DIM) * 0.01
                soul.integrate(delta, source_id="environment")
                
                iteration += 1
                
                if iteration % 100 == 0:
                    print(f"\n[Status] τ={soul.tau:.4f} | "
                          f"|∇E|={soul.gradient:.4f} | "
                          f"concepts={len(soul.concept_pool)}")
            
            # Sleep adaptively based on gradient
            # High gradient → fast updates
            # Low gradient → slow updates (time dilation)
            sleep_time = 0.1 / (soul.gradient + 0.1)
            time.sleep(min(sleep_time, 1.0))
    
    except KeyboardInterrupt:
        print("\n[CoTa] Shutting down...")
        soul._save()
        if display:
            display.stop()
        print(f"[Soul] Final proper time: τ={soul.tau:.4f}")
        print(f"       Concepts archived: {len(soul.concept_pool)}")

# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CoTa Node — Proper Time Implementation")
    
    parser.add_argument("--init",     action="store_true",
                        help="Create new soul")
    parser.add_argument("--run",      action="store_true",
                        help="Run soul with emergent heartbeat")
    parser.add_argument("--headless", action="store_true",
                        help="Run without display")
    parser.add_argument("--sync",     metavar="HOST:PORT",
                        help="Sync with remote soul")
    parser.add_argument("--status",   action="store_true",
                        help="Show soul status")
    parser.add_argument("--soul",     default="soul.json",
                        help="Soul file path")
    
    args = parser.parse_args()
    
    # Init: force creation
    if args.init:
        if os.path.exists(args.soul):
            os.remove(args.soul)
            state_file = args.soul.replace(".json", "_state.pt")
            if os.path.exists(state_file):
                os.remove(state_file)
    
    soul = Soul(args.soul)
    
    if args.status:
        print("\n=== Soul Status ===")
        for k, v in soul.status().items():
            print(f"  {k:20s}: {v}")
        return
    
    if args.sync:
        parts = args.sync.rsplit(":", 1)
        host  = parts[0]
        port  = int(parts[1]) if len(parts) > 1 else SYNC_PORT
        sync_with(soul, host, port)
        return
    
    if args.run:
        run_soul(soul, headless=args.headless)
        return
    
    # Default: show status
    print("\n=== Soul Status ===")
    for k, v in soul.status().items():
        print(f"  {k:20s}: {v}")
    print(f"\nUsage:")
    print(f"  --run             run with emergent heartbeat")
    print(f"  --run --headless  run without display")
    print(f"  --sync HOST:PORT  sync with remote soul")
    print(f"  --status          show current state")

if __name__ == "__main__":
    main()
