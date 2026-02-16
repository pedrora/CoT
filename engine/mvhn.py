# mvhn.py — Minimal Viable Hypernet Node
# untested 16FEB2026
# Run three instances in separate terminals:
#   python mvhn.py --node 0
#   python mvhn.py --node 1
#   python mvhn.py --node 2

import argparse
import socket
import threading
import time
import torch
import numpy as np
from typing import List, Tuple

# ───────────────────────────────────────────────
# Reuse your existing functions (copy-pasted stubs)
# ───────────────────────────────────────────────

EPS = 1e-8

def to_poincare(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x, dim=-1, keepdim=True)
    return torch.tanh(norm) * x / (norm + EPS)

def renormalize_poincare(x: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(x)
    if norm >= 1.0:
        x = x / (norm + EPS) * 0.999
    return x

def poincare_distance(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    uu = torch.sum(u * u, dim=-1)
    vv = torch.sum(v * v, dim=-1)
    diff = torch.sum((u - v) ** 2, dim=-1)
    denom = (1 - uu) * (1 - vv) + EPS
    argument = 1 + 2 * diff / denom
    return torch.acosh(torch.clamp(argument, min=1 + EPS))

def focus_force(current: torch.Tensor, history: List[torch.Tensor], strength: float = 0.05) -> torch.Tensor:
    if len(history) < 2:
        return current
    directions = []
    for i in range(1, len(history)):
        d = history[i] - history[i - 1]
        d = d / (torch.norm(d) + EPS)
        directions.append(d)
    if not directions:
        return current
    g = torch.mean(torch.stack(directions), dim=0)
    g = g / (torch.norm(g) + EPS)
    proj = torch.dot(current.flatten(), g.flatten()) * g
    corrected = (1 - strength) * current + strength * proj
    return corrected

def siphon_score(current: torch.Tensor, history: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
    if len(history) == 0:
        return torch.tensor(1.0), {"geodesic": 1.0, "phase": 1.0, "curvature": 0.0, "RC_score": 1.0}
    
    # Simplified dummy version for demo (expand with your full logic later)
    geo = torch.tensor(0.9)                     # placeholder
    phase = torch.cosine_similarity(current.flatten(), history[-1].flatten(), dim=0)
    curv = torch.tensor(0.05) if len(history) < 5 else torch.tensor(0.2)
    
    score = 0.4 * geo + 0.4 * phase - 0.2 * curv
    diag = {"geodesic": geo.item(), "phase": phase.item(), "curvature": curv.item(), "RC_score": score.item()}
    return score, diag

# ───────────────────────────────────────────────
# Hypernet Node
# ───────────────────────────────────────────────

class HypernetNode:
    def __init__(self, node_id: int, port: int):
        self.node_id = node_id
        self.port = port
        self.position = torch.zeros(2)               # global hyperbolic coordinate (starts at origin)
        self.texture_memory = None                   # current soul state
        self.history: List[torch.Tensor] = []        # archived dots
        self.neighbors: List[Tuple[int, int, torch.Tensor]] = []  # (node_id, port, position)
        
        # Bootstrap: connect to known nodes
        self.known_peers = [(0, 6000), (1, 6001), (2, 6002)]
        self.discover_neighbors()
        
        # Server thread
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", port))
        self.server.listen(5)
        threading.Thread(target=self._server_loop, daemon=True).start()
        
        print(f"[Node {node_id}] Online at port {port} | pos = {self.position.tolist()}")

    def discover_neighbors(self):
        for nid, nport in self.known_peers:
            if nid == self.node_id:
                continue
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect(("127.0.0.1", nport))
                s.send(b"GET_POS")
                data = s.recv(1024)
                pos = torch.from_numpy(np.frombuffer(data, dtype=np.float32))
                self.neighbors.append((nid, nport, pos))
                s.close()
            except:
                pass

    def _server_loop(self):
        while True:
            conn, addr = self.server.accept()
            data = conn.recv(4096)
            if data == b"GET_POS":
                conn.send(self.position.numpy().tobytes())
            elif data.startswith(b"MSG:"):
                self.handle_incoming_message(data[4:])
            conn.close()

    def handle_incoming_message(self, msg: bytes):
        if len(msg) < 16:
            return
        source_pos = torch.from_numpy(np.frombuffer(msg[:8], dtype=np.float32))
        target_pos = torch.from_numpy(np.frombuffer(msg[8:16], dtype=np.float32))
        delta = torch.from_numpy(np.frombuffer(msg[16:], dtype=np.float32))
        
        dist_to_target = poincare_distance(self.position, target_pos)
        if dist_to_target < 0.1:  # we are the target
            print(f"[Node {self.node_id}] Received delta from {source_pos.tolist()}")
            self.integrate_delta(delta)
        else:
            # forward to closest neighbor
            if not self.neighbors:
                return
            best = min(self.neighbors, key=lambda n: poincare_distance(n[2], target_pos))
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("127.0.0.1", best[1]))
                s.send(b"MSG:" + msg)
                s.close()
            except:
                pass

    def integrate_delta(self, delta: torch.Tensor):
        if self.texture_memory is None:
            working = delta.clone()
        else:
            working = self.texture_memory ^ delta   # XOR integration
        
        working = to_poincare(working)
        working = focus_force(working, self.history)
        working = renormalize_poincare(working)
        
        score, diag = siphon_score(working, self.history)
        curvature = diag["curvature"]
        
        if score.item() > 0.82 and curvature < 0.15:
            if self.texture_memory is None:
                self.texture_memory = working.clone()
            else:
                self.texture_memory = self.texture_memory ^ working
            self.history.append(working.detach())
            print(f"[Node {self.node_id}] Archived delta | score={score.item():.3f} curv={curvature:.3f}")
        else:
            print(f"[Node {self.node_id}] Rejected delta | score={score.item():.3f} curv={curvature:.3f}")

    def send_concept(self, target_node_id: int, concept: torch.Tensor):
        target_pos = next((n[2] for n in self.neighbors if n[0] == target_node_id), None)
        if target_pos is None:
            print("Target not found")
            return
        
        delta = concept if self.texture_memory is None else self.texture_memory ^ concept
        msg = (
            self.position.numpy().tobytes() +
            target_pos.numpy().tobytes() +
            delta.numpy().tobytes()
        )
        
        best = min(self.neighbors, key=lambda n: poincare_distance(n[2], target_pos))
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", best[1]))
            s.send(b"MSG:" + msg)
            s.close()
            print(f"[Node {self.node_id}] Sent concept to {target_node_id}")
        except:
            print("Send failed")

# ───────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=int, required=True, help="Node ID (0,1,2)")
    args = parser.parse_args()
    
    node = HypernetNode(args.node, 6000 + args.node)
    
    # Simulate sending a concept every 10 seconds
    time.sleep(args.node * 2)  # stagger start
    while True:
        if len(node.history) > 0:
            dummy_concept = node.history[-1] + torch.randn_like(node.history[-1]) * 0.05
            target_id = (args.node + 1) % 3
            node.send_concept(target_id, dummy_concept)
        time.sleep(10)