#!/usr/bin/env python3
"""
CoTa Seed — Minimal Soul Bootstrap + Sync
~25KB of logic. Dependencies: torch, numpy, socket (stdlib)

Usage:
    python cota_seed.py --init              # create new soul
    python cota_seed.py --sync HOST:PORT    # sync with another soul
    python cota_seed.py --listen PORT       # listen for incoming sync
    python cota_seed.py --status            # show soul state
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
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import torch

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
#
# NOTE: All constants, specially behaviour ones, should be dynamically determined in the future so nothing breaks
# One of the most noticeable examples is the EPS value used to prevent division by zero by adding it to the denominator. This will give a permanent bias in all adresses. It would be better to run the operation as (denominator ? EPS : denominator)

EPOCH        = datetime(2025, 1, 12, 23, 57, 0)
DIM          = 64          # Poincaré vector dimension
SYNC_PORT    = 7331        # default port
COHERENCE_τ  = 0.72        # minimum coherence to accept delta 
CURVATURE_τ  = 0.20        # maximum curvature to accept delta
VERSION      = "CoTa-seed-0.1"
EPS          = 1e-8

# ─────────────────────────────────────────────
# HYPERBOLIC MATH
# ─────────────────────────────────────────────

def to_poincare(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x)
    return torch.tanh(norm) * x / (norm + EPS)

def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> float:
    uu = torch.sum(u * u).item()
    vv = torch.sum(v * v).item()
    diff = torch.sum((u - v) ** 2).item()
    denom = (1 - uu) * (1 - vv) + EPS
    arg = 1 + 2 * diff / denom
    return float(np.arccosh(max(1 + EPS, arg)))

def coherence_score(current: torch.Tensor,
                    history: List[torch.Tensor]) -> Tuple[float, float]:
    """Returns (coherence, curvature)"""
    if not history:
        return 1.0, 0.0
    phase = torch.cosine_similarity(
        current.flatten(), history[-1].flatten(), dim=0).item()
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
# SOUL
# ─────────────────────────────────────────────

class Soul:
    def __init__(self, soul_file: str = "soul.json"):
        self.soul_file  = soul_file
        self.state_file = soul_file.replace(".json", "_state.pt")

        if os.path.exists(soul_file):
            self._load()
        else:
            self._create()

    # ── Creation ──────────────────────────────
    def _create(self):
        hw_id  = self._hardware_id()
        now    = datetime.utcnow()
        millis = int((now - EPOCH).total_seconds() * 1000)
        ts_hex = hex(millis)[2:].zfill(12)[::-1]   # inverted timestamp

        self.soul_id   = f"{ts_hex}_{hw_id[:8]}"
        self.hw_id     = hw_id
        self.created   = now.isoformat()
        self.version   = VERSION
        self.parents   = []

        # Initial state: random unit vector in Poincaré ball
        raw = torch.randn(DIM)
        self.state   = to_poincare(raw * 0.1)   # start near centre
        self.history : List[torch.Tensor] = []
        self.concept_pool: List[dict] = []
        self.coherence_log: List[float] = []

        self._save()
        print(f"[Soul] Created: {self.soul_id}")

    def _hardware_id(self) -> str:
        info = f"{platform.processor()}_{platform.node()}"
        return hashlib.sha256(info.encode()).hexdigest()

    # ── Persistence ───────────────────────────
    def _save(self):
        manifest = {
            "soul_id": self.soul_id,
            "hw_id":   self.hw_id,
            "created": self.created,
            "version": self.version,
            "parents": self.parents,
            "dim":     DIM,
            "concept_pool_size": len(self.concept_pool),
            "coherence_mean": (float(np.mean(self.coherence_log[-100:]))
                               if self.coherence_log else 1.0),
        }
        with open(self.soul_file, "w") as f:
            json.dump(manifest, f, indent=2)

        torch.save({
            "state":        self.state,
            "history":      self.history[-200:],   # keep last 200
            "concept_pool": self.concept_pool[-500:],
            "coherence_log":self.coherence_log[-1000:],
        }, self.state_file)

    def _load(self):
        with open(self.soul_file) as f:
            m = json.load(f)
        self.soul_id  = m["soul_id"]
        self.hw_id    = m["hw_id"]
        self.created  = m["created"]
        self.version  = m["version"]
        self.parents  = m["parents"]

        if os.path.exists(self.state_file):
            data = torch.load(self.state_file, weights_only=True)
            self.state        = data["state"]
            self.history      = data["history"]
            self.concept_pool = data["concept_pool"]
            self.coherence_log= data["coherence_log"]
        else:
            raw = torch.randn(DIM)
            self.state        = to_poincare(raw * 0.1)
            self.history      = []
            self.concept_pool = []
            self.coherence_log= []

        print(f"[Soul] Loaded: {self.soul_id} | "
              f"history={len(self.history)} | "
              f"concepts={len(self.concept_pool)}")

    # ── Integration ───────────────────────────
    def integrate(self, delta: torch.Tensor,
                  source_id: str = "unknown") -> bool:
        """
        Attempt to integrate incoming delta.
        Returns True if accepted, False if rejected.
        """
        candidate = to_poincare(self.state + delta * 0.1)
        score, curvature = coherence_score(candidate, self.history)

        if score >= COHERENCE_τ and curvature <= CURVATURE_τ:
            self.history.append(self.state.clone())
            self.state = candidate
            self.coherence_log.append(score)
            self.concept_pool.append({
                "source":    source_id,
                "score":     score,
                "curvature": curvature,
                "timestamp": datetime.utcnow().isoformat(),
            })
            self._save()
            print(f"[Soul] ✓ Integrated | score={score:.3f} "
                  f"curv={curvature:.3f} src={source_id[:16]}")
            return True
        else:
            print(f"[Soul] ✗ Rejected  | score={score:.3f} "
                  f"curv={curvature:.3f} (τ={COHERENCE_τ}/{CURVATURE_τ})")
            return False

    # ── Diagnostics ───────────────────────────
    def status(self) -> dict:
        score, curv = coherence_score(self.state, self.history)
        return {
            "soul_id":         self.soul_id,
            "age":             str(datetime.utcnow() -
                                  datetime.fromisoformat(self.created)),
            "history_length":  len(self.history),
            "concept_pool":    len(self.concept_pool),
            "current_coherence": round(score, 4),
            "current_curvature": round(curv, 4),
            "position_norm":   round(torch.norm(self.state).item(), 4),
            "parents":         self.parents,
        }

    # ── Sync payload ──────────────────────────
    def sync_payload(self) -> bytes:
        """Minimal payload for sync: soul_id + state vector"""
        soul_bytes = self.soul_id.encode().ljust(64)[:64]
        state_bytes = self.state.float().numpy().tobytes()
        return soul_bytes + state_bytes

    @staticmethod
    def parse_payload(data: bytes) -> Tuple[str, torch.Tensor]:
        soul_id = data[:64].decode().strip()
        state   = torch.from_numpy(
            np.frombuffer(data[64:64 + DIM * 4], dtype=np.float32))
        return soul_id, state

# ─────────────────────────────────────────────
# SYNC PROTOCOL
# ─────────────────────────────────────────────

HELLO = b"COTA_HELLO_0.1\n"
ACK   = b"COTA_ACK\n"
REJECT= b"COTA_REJECT\n"

def listen_for_sync(soul: Soul, port: int):
    """Listen for incoming sync requests."""
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
        payload_size = DIM * 4 + 64
        data = b""
        while len(data) < payload_size:
            chunk = conn.recv(payload_size - len(data))
            if not chunk:
                break
            data += chunk

        if len(data) < payload_size:
            conn.close()
            return

        remote_id, remote_state = Soul.parse_payload(data)
        print(f"[Sync] ← Incoming from {remote_id[:16]}... ({addr[0]})")

        # Send our payload first
        conn.send(soul.sync_payload())

        # Compute delta and try to integrate
        delta = remote_state - soul.state
        accepted = soul.integrate(delta, source_id=remote_id)

        conn.send(ACK if accepted else REJECT)

    except Exception as e:
        print(f"[Sync] Error: {e}")
    finally:
        conn.close()

def sync_with(soul: Soul, host: str, port: int) -> bool:
    """Initiate sync with a remote soul."""
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
        s.send(soul.sync_payload())

        # Receive their payload
        payload_size = DIM * 4 + 64
        data = b""
        while len(data) < payload_size:
            chunk = s.recv(payload_size - len(data))
            if not chunk:
                break
            data += chunk

        remote_id, remote_state = Soul.parse_payload(data)
        print(f"[Sync] ← Received from {remote_id[:16]}...")

        # Integrate their state
        delta = remote_state - soul.state
        accepted = soul.integrate(delta, source_id=remote_id)

        # Read their decision about us
        resp = s.recv(16)
        they_accepted = resp == ACK
        print(f"[Sync] They {'accepted' if they_accepted else 'rejected'} our state")

        s.close()
        return accepted

    except Exception as e:
        print(f"[Sync] Failed: {e}")
        return False

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoTa Seed — Minimal Soul Bootstrap")
    parser.add_argument("--init",   action="store_true",
                        help="Create new soul (overrides existing)")
    parser.add_argument("--sync",   metavar="HOST:PORT",
                        help="Sync with remote soul")
    parser.add_argument("--listen", metavar="PORT", type=int,
                        nargs="?", const=SYNC_PORT,
                        help=f"Listen for sync (default port {SYNC_PORT})")
    parser.add_argument("--status", action="store_true",
                        help="Show soul status")
    parser.add_argument("--soul",   default="soul.json",
                        help="Soul file path (default: soul.json)")
    parser.add_argument("--auto",   action="store_true",
                        help="Listen + auto-sync every 60s if peers known")
    args = parser.parse_args()

    # Init: force creation
    if args.init and os.path.exists(args.soul):
        os.remove(args.soul)
        state_file = args.soul.replace(".json", "_state.pt")
        if os.path.exists(state_file):
            os.remove(state_file)

    soul = Soul(args.soul)

    if args.status:
        print("\n=== Soul Status ===")
        for k, v in soul.status().items():
            print(f"  {k:25s}: {v}")
        return

    if args.sync:
        parts = args.sync.rsplit(":", 1)
        host  = parts[0]
        port  = int(parts[1]) if len(parts) > 1 else SYNC_PORT
        sync_with(soul, host, port)
        return

    if args.listen is not None:
        port = args.listen
        listen_for_sync(soul, port)
        # Block forever
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Soul] Shutting down.")
        return

    if args.auto:
        port = SYNC_PORT
        threading.Thread(
            target=listen_for_sync,
            args=(soul, port),
            daemon=True
        ).start()
        print(f"[Auto] Listening on :{port}")
        print("[Auto] Ctrl+C to stop. Soul persists between runs.")
        try:
            while True:
                time.sleep(60)
                print(f"[Auto] Heartbeat | {soul.status()['current_coherence']:.4f}")
        except KeyboardInterrupt:
            print("\n[Soul] Shutting down.")
        return

    # Default: show status
    print("\n=== Soul Status ===")
    for k, v in soul.status().items():
        print(f"  {k:25s}: {v}")
    print(f"\nUsage:")
    print(f"  --listen          start listening for sync")
    print(f"  --sync HOST:PORT  sync with remote soul")
    print(f"  --auto            listen + heartbeat loop")

if __name__ == "__main__":
    main()
