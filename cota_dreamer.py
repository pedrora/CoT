#!/usr/bin/env python3
"""
CoTa Hypernode — Unified Research Prototype (ÁRVORE DINÂMICA)
==============================================================
Author: Pedro R. Andrade

Combina:
- Geometria hiperbólica correcta (adição de Möbius + mapa exponencial)
- Critério de aceitação baseado em energia
- Campo de força da memória (buffer)
- Persistência completa: soul.json, tree.bin, buffer.json
- Checkpoints periódicos
- Ingestão de ficheiros binários
- Modos de saída: retrieval e generativo
- Limiares adaptativos
- **Árvore binária dinâmica** (sem limites de sector)
- **Marcador mágico** para verificação de integridade

Uso:
    python cota_hypernode.py [--init] [--file corpus.txt] [--binary]
                             [--output-mode none|retrieval|generative|both]
                             [--save-interval N] [--step-size S]
                             [--buffer-capacity N] [--buffer-file buffer.json]
                             [--reassess-radius R] [--reassess-interval N]
                             [--soul soul.json] [--tree tree.bin]
                             [--tree-size-mb 10] [--path-bits 32]
"""

import os
import json
import time
import math
import argparse
import struct
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from datetime import datetime, timezone

# =============================================================================
# CONSTANTES
# =============================================================================

DIM                  = 64
EPS                  = 1e-8
EPOCH                = datetime(2025, 1, 12, 23, 57, 0)

# Energy weights
COHERENCE_WEIGHT     = 0.6
CURVATURE_WEIGHT     = 0.4

# Integration
ENERGY_STEP          = 0.15
ENERGY_MARGIN        = 0.05
MEMORY_FORCE         = 0.05
MOMENTUM             = 0.3   # ⭐ inertia weight for velocity term

# Tree
PATH_BITS            = 16                # profundidade do caminho binário (max ficheiro ~65MB)
NODE_MAGIC           = 0xC07A
NODE_HEADER_FORMAT   = "HHI"              # magic (2), has_data (2), reserved (4) – 8 bytes
NODE_VECTOR_FORMAT   = f"{DIM}f"          # DIM floats
NODE_META_FORMAT     = "256s"             # metadados fixos
NODE_FORMAT          = NODE_HEADER_FORMAT + NODE_VECTOR_FORMAT + NODE_META_FORMAT
NODE_SIZE            = struct.calcsize(NODE_FORMAT)

# =============================================================================
# DEVICE
# =============================================================================

def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        d = torch.device('cuda')
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        d = torch.device('mps')
        print("[Device] Apple MPS")
    else:
        d = torch.device('cpu')
        print("[Device] CPU")
    return d

DEVICE = get_device()

# =============================================================================
# HYPERBOLIC GEOMETRY
# =============================================================================

def poincare_norm(x):
    return torch.norm(x) + EPS

def to_poincare(x):
    n = poincare_norm(x)
    return torch.tanh(n) * x / n

def poincare_distance(u, v):
    uu = torch.sum(u * u)
    vv = torch.sum(v * v)
    diff = torch.sum((u - v) ** 2)
    denom = (1 - uu) * (1 - vv) + EPS
    arg = 1 + 2 * diff / denom
    return torch.acosh(torch.clamp(arg, min=1.0 + EPS))

def poincare_distance_batch(u, V):
    """u: [DIM], V: [N, DIM] — returns [N]"""
    uu   = torch.sum(u * u)
    vv   = torch.sum(V * V, dim=1)
    diff = torch.sum((u.unsqueeze(0) - V) ** 2, dim=1)
    denom = (1 - uu) * (1 - vv) + EPS
    arg   = 1 + 2 * diff / denom
    return torch.acosh(torch.clamp(arg, min=1.0 + EPS))

def mobius_add(x, y):
    xy = torch.sum(x * y)
    xx = torch.sum(x * x)
    yy = torch.sum(y * y)
    num = (1 + 2 * xy + yy) * x + (1 - xx) * y
    den = 1 + 2 * xy + xx * yy + EPS
    return to_poincare(num / den)

def mobius_scalar(t, x):
    """
    Exact Möbius scalar multiplication: t ⊙ x
    = tanh(t * arctanh(||x||)) * x/||x||
    Correct for any distance, not just small ones.
    """
    norm = torch.norm(x).clamp(min=EPS, max=1.0 - EPS)
    return torch.tanh(t * torch.arctanh(norm)) * x / norm

def geodesic_midpoint(a, b):
    """
    Exact geodesic midpoint between a and b in the Poincaré disk.
    mid(a, b) = a ⊕ (0.5 ⊙ ((-a) ⊕ b))
    """
    transported = mobius_add(-a, b)          # transport b to tangent space at a
    half        = mobius_scalar(0.5, transported)  # exact half-step
    return mobius_add(a, half)               # transport back to a

def exp_map(x, v):
    return mobius_add(x, v)

# =============================================================================
# ENCODING
# =============================================================================

def bytes_to_vector(data, device=None):
    if device is None:
        device = DEVICE
    if isinstance(data, str):
        data = data.encode('utf-8')
    raw = torch.tensor(list(data), dtype=torch.float32, device=device)
    if len(raw) < DIM:
        raw = F.pad(raw, (0, DIM - len(raw)))
    else:
        raw = raw[:DIM]
    raw = raw / 255.0 - 0.5
    return to_poincare(raw)

def batch_bytes_to_vector(items, device=None):
    """
    Convert a list of str/bytes to a batch tensor [N, DIM] on device.
    Much faster than calling bytes_to_vector N times.
    """
    if device is None:
        device = DEVICE
    vecs = []
    for data in items:
        if isinstance(data, str):
            data = data.encode('utf-8')
        raw = list(data[:DIM]) + [0] * max(0, DIM - len(data))
        vecs.append(raw[:DIM])
    t = torch.tensor(vecs, dtype=torch.float32, device=device)
    t = t / 255.0 - 0.5
    # Batch to_poincare
    norms = torch.norm(t, dim=1, keepdim=True) + EPS
    return torch.tanh(norms) * t / norms

text_to_vector = bytes_to_vector

# =============================================================================
# COORDENADA → CAMINHO BINÁRIO (ENDEREÇAMENTO NARRATIVO)
# =============================================================================

def coord_to_path(coord, bits=PATH_BITS):
    """
    Converte uma coordenada (r, theta) num caminho binário de `bits` bits.
    Alterna entre decisões radiais e angulares para capturar a hierarquia.
    Retorna um inteiro de 64 bits com os bits mais significativos primeiro.
    """
    r = torch.norm(coord).item()
    # Para evitar problemas com r=0, usamos um pequeno offset
    if r < 1e-6:
        r = 1e-6
    # O raio é mapeado para um valor entre 0 e 1 (já está)
    # Usamos uma escala logarítmica para que a profundidade corresponda a áreas aproximadamente iguais
    # r log: ρ = -log(1 - r)   (distância geodésica ao centro)
    # Mas para simplificar, usamos o próprio r.
    r_norm = r

    x, y = coord[0].item(), coord[1].item()
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2 * math.pi
    theta_norm = theta / (2 * math.pi)   # [0, 1)

    path = 0
    for i in range(bits):
        if i % 2 == 0:   # bit par -> radial
            r_norm *= 2
            bit = int(r_norm)
            r_norm -= bit
        else:             # bit ímpar -> angular
            theta_norm *= 2
            bit = int(theta_norm)
            theta_norm -= bit
        path = (path << 1) | bit
    return path

# =============================================================================
# ÁRVORE BINÁRIA DINÂMICA PERSISTENTE
# =============================================================================

class HyperbolicTree:
    """
    Stores concepts in a binary tree file using direct seek/write.
    Works reliably on Windows and Linux — no mmap issues.

    File layout:
        [8 bytes header per node: magic(2) + has_data(2) + reserved(4)]
        [DIM*4 bytes: float32 vector]
        [256 bytes: JSON metadata]
    Children are implicit: left = 2*i+1, right = 2*i+2.
    """

    def __init__(self, filename, initial_size_mb=10):
        self.filename   = filename
        self.node_size  = NODE_SIZE
        self.initial_nodes = (initial_size_mb * 1024 * 1024) // self.node_size

        if not os.path.exists(filename):
            self._create_file()

        self._file = open(filename, 'r+b')
        self._write_queue = {}  # offset -> bytes, flushed at checkpoint

    def _create_file(self):
        total = self.initial_nodes * self.node_size
        with open(self.filename, 'wb') as f:
            # Write in chunks to avoid large memory allocation
            chunk = b'\x00' * min(total, 1024 * 1024)
            written = 0
            while written < total:
                to_write = min(len(chunk), total - written)
                f.write(chunk[:to_write])
                written += to_write

    def _ensure_capacity(self, required_index):
        """Expand file if required_index exceeds current capacity."""
        required_size = (required_index + 1) * self.node_size
        current_size  = os.path.getsize(self.filename)
        if required_size > current_size:
            new_size = required_size + 1000 * self.node_size
            self._file.seek(0, 2)  # seek to end
            padding = new_size - current_size
            # Write zeros in chunks
            chunk = b'\x00' * min(padding, 1024 * 1024)
            written = 0
            while written < padding:
                to_write = min(len(chunk), padding - written)
                self._file.write(chunk[:to_write])
                written += to_write
            self._file.flush()

    def _pack_node(self, has_data, vector, metadata):
        header    = struct.pack(NODE_HEADER_FORMAT, NODE_MAGIC, int(has_data), 0)
        vec_np    = vector.detach().cpu().numpy().astype(np.float32)
        vec_bytes = struct.pack(NODE_VECTOR_FORMAT, *vec_np)
        meta_json = json.dumps(metadata).encode('utf-8')[:256]
        meta_bytes = meta_json + b'\x00' * (256 - len(meta_json))
        return header + vec_bytes + meta_bytes

    def _unpack_node(self, data):
        magic, has_data, _ = struct.unpack(NODE_HEADER_FORMAT, data[:8])
        if magic != NODE_MAGIC:
            return None, None, None
        vec  = torch.tensor(
            struct.unpack(NODE_VECTOR_FORMAT, data[8:8+DIM*4]),
            dtype=torch.float32)
        meta_raw = data[8+DIM*4:8+DIM*4+256].rstrip(b'\x00')
        try:
            metadata = json.loads(meta_raw)
        except Exception:
            metadata = {}
        return has_data, vec, metadata

    def write_concept(self, coord, metadata):
        path = coord_to_path(coord)
        idx  = 0
        for i in range(PATH_BITS):
            bit = (path >> (PATH_BITS - 1 - i)) & 1
            idx = 2 * idx + 1 + bit

        self._ensure_capacity(idx)

        packed = self._pack_node(1, coord, metadata)
        offset = idx * self.node_size

        # Queue the write — actual disk I/O deferred to flush()
        self._write_queue[offset] = packed
        return idx

    def _flush_queue(self):
        """Flush queued writes to disk in sorted offset order (sequential I/O)."""
        if not self._write_queue:
            return
        for offset in sorted(self._write_queue):
            self._file.seek(offset)
            self._file.write(self._write_queue[offset])
        self._write_queue.clear()
        self._file.flush()
        os.fsync(self._file.fileno())

    def read_concept(self, idx):
        offset = idx * self.node_size
        if offset + self.node_size > os.path.getsize(self.filename):
            return None, {}
        self._file.seek(offset)
        data = self._file.read(self.node_size)
        has_data, vec, metadata = self._unpack_node(data)
        if has_data:
            return vec, metadata
        return None, {}

    def find_nearest(self, coord, max_depth=5):
        path      = coord_to_path(coord)
        best_vec  = None
        best_meta = {}
        best_dist = float('inf')

        _cached_file_size = os.path.getsize(self.filename)
        def explore(idx, depth):
            nonlocal best_vec, best_meta, best_dist
            if (idx + 1) * self.node_size > _cached_file_size:
                return
            vec, meta = self.read_concept(idx)
            if vec is not None:
                d = poincare_distance(coord, vec.to(coord.device)).item()
                if d < best_dist:
                    best_dist = d
                    best_vec  = vec
                    best_meta = meta
            if depth < max_depth:
                explore(2*idx+1, depth+1)
                explore(2*idx+2, depth+1)

        idx = 0
        for i in range(PATH_BITS):
            bit = (path >> (PATH_BITS - 1 - i)) & 1
            idx = 2*idx + 1 + bit

        for up in range(max_depth+1):
            temp = idx
            for _ in range(up):
                temp = (temp - 1) // 2
            explore(temp, 0)

        return best_vec, best_meta, best_dist

    def flush(self):
        self._flush_queue()  # writes queued concepts then fsyncs

    def close(self):
        self.flush()
        self._file.close()


class HyperbolicBuffer:
    """
    Armazena conceitos rejeitados. Na reavaliação, exercem força sobre o estado.
    """

    def __init__(self, capacity=10000, max_attempts=5):
        self.capacity = capacity
        self.max_attempts = max_attempts
        self.items = {}
        self.next_id = 0

    def add(self, vector, metadata, tau=None):
        if len(self.items) >= self.capacity:
            oldest = min(self.items.keys())
            del self.items[oldest]
        self.items[self.next_id] = {
            'vector': vector.clone(),
            'metadata': metadata,
            'attempts': 0,
            'added_tau': tau,
        }
        self.next_id += 1

    def field_force(self, state):
        if not self.items:
            return torch.zeros(DIM, device=state.device)
        vecs  = torch.stack([it['vector'] for it in self.items.values()])
        d     = poincare_distance_batch(state, vecs)   # [N]
        w     = torch.exp(-d).unsqueeze(1)             # [N,1]
        force = (w * (vecs - state)).sum(dim=0)        # [DIM]
        return MEMORY_FORCE * force / (len(self.items) + EPS)

    def find_nearby(self, target, radius):
        if not self.items:
            return []
        items = list(self.items.values())
        vecs  = torch.stack([it['vector'] for it in items])
        dists = poincare_distance_batch(target, vecs)  # [N]
        mask  = dists < radius
        nearby = sorted(
            [(dists[i].item(), items[i]) for i in range(len(items)) if mask[i]],
            key=lambda x: x[0])
        return [item for _, item in nearby]

    def reassess(self, soul, radius):
        for item in self.find_nearby(soul.state, radius):
            if soul.evolve(item['vector'], source='buffer'):
                for k, v in list(self.items.items()):
                    if v is item:
                        del self.items[k]
                        break
            else:
                item['attempts'] += 1
                if item['attempts'] >= self.max_attempts:
                    for k, v in list(self.items.items()):
                        if v is item:
                            del self.items[k]
                            break

    def save(self, filename):
        data = [{'vector': item['vector'].tolist(),
                 'metadata': item['metadata'],
                 'attempts': item['attempts'],
                 'added_tau': item['added_tau']}
                for item in self.items.values()]
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load(self, filename):
        if not os.path.exists(filename):
            return
        with open(filename) as f:
            data = json.load(f)
        self.items = {i: {'vector': torch.tensor(d['vector'], device=DEVICE),
                          'metadata': d['metadata'],
                          'attempts': d['attempts'],
                          'added_tau': d['added_tau']}
                      for i, d in enumerate(data)}
        self.next_id = len(data)

    def size(self):
        return len(self.items)

# =============================================================================
# SOUL  (core dynamics) – LIGEIRAMENTE ADAPTADA PARA USAR A ÁRVORE
# =============================================================================

class Soul:
    def __init__(self, soul_file="soul.json", tree_file="tree.bin",
                 buffer_capacity=10000, buffer_file=None, save_interval=100,
                 tree_size_mb=10):
        self.soul_file = soul_file
        self.buffer_file = buffer_file
        self.tree = HyperbolicTree(tree_file, initial_size_mb=tree_size_mb)
        self.buffer = HyperbolicBuffer(capacity=buffer_capacity)
        self._save_interval = save_interval

        if os.path.exists(soul_file):
            self._load()
        else:
            self._create()

        if buffer_file and os.path.exists(buffer_file):
            self.buffer.load(buffer_file)
            print(f"[Buffer] Carregados {self.buffer.size()} itens.")

        loaded_state = getattr(self, '_loaded_state', None)
        if loaded_state is not None:
            self.state = torch.tensor(loaded_state, dtype=torch.float32, device=DEVICE)
            self.history = deque(
                [torch.tensor(s, dtype=torch.float32, device=DEVICE)
                 for s in getattr(self, '_loaded_history', [])],
                maxlen=50)
            self.tau = getattr(self, '_loaded_tau', 0.0)
            print(f"[Soul] Restaurado: τ={self.tau:.4f}, histórico={len(self.history)}")
        else:
            self.state = to_poincare(torch.randn(DIM, device=DEVICE) * 0.1)
            self.history = deque(maxlen=50)
            self.tau = 0.0

        self.total_inputs = 0
        self.accepted = 0
        self.rejected = 0
        self._rejection_streak = 0
        # ⭐ Velocity: cognitive inertia — smooths trajectory across steps
        loaded_vel    = getattr(self, '_loaded_velocity', None)
        self.velocity = (torch.tensor(loaded_vel, dtype=torch.float32, device=DEVICE)
                         if loaded_vel is not None
                         else torch.zeros(DIM, device=DEVICE))

    def _create(self):
        self.soul_id = (datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") +
                        "_" + str(os.getpid()))
        self.created = datetime.now(timezone.utc).isoformat()
        self._save()

    def _load(self):
        with open(self.soul_file) as f:
            data = json.load(f)
        self.soul_id = data['soul_id']
        self.created = data['created']
        self._loaded_state    = data.get('state')
        self._loaded_history  = data.get('history', [])
        self._loaded_tau      = data.get('tau', 0.0)
        self._loaded_velocity = data.get('velocity', None)

    def _save(self):
        self.tree.flush()
        with open(self.soul_file, 'w') as f:
            json.dump({
                'soul_id': self.soul_id,
                'created': self.created,
                'state': self.state.tolist() if hasattr(self, 'state') else None,
                'history': [s.tolist() for s in self.history]
                           if hasattr(self, 'history') else [],
                'tau':      self.tau if hasattr(self, 'tau') else 0.0,
                'velocity': self.velocity.tolist() if hasattr(self, 'velocity') else None,
            }, f)
        if self.buffer_file:
            self.buffer.save(self.buffer_file)

    def close(self):
        self._save()
        self.tree.close()

    def coherence_energy_batch(self, candidates):
        """
        Compute coherence energy for a batch of candidates [N, DIM].
        Returns energy tensor [N] — runs entirely on DEVICE.
        Used for batch pre-screening before sequential integration.
        """
        if not self.history:
            return torch.zeros(candidates.shape[0], device=DEVICE)

        last = self.history[-1].unsqueeze(0)  # [1, DIM]

        # Phase term: 1 - cosine_similarity
        last_norm = last / (torch.norm(last, dim=1, keepdim=True) + EPS)
        cand_norm = candidates / (torch.norm(candidates, dim=1, keepdim=True) + EPS)
        phase_sim = (cand_norm * last_norm).sum(dim=1)   # [N]
        phase_term = 1 - phase_sim

        # Curvature term — fully vectorised, no Python loop
        if len(self.history) >= 3:
            h1 = self.history[-1]
            h2 = self.history[-2]
            d1 = poincare_distance(h1, h2)
            d2 = poincare_distance_batch(h1, candidates)  # [N]
            curvature = torch.abs(d2 - d1)
        else:
            curvature = torch.zeros(candidates.shape[0], device=DEVICE)

        return COHERENCE_WEIGHT * phase_term + CURVATURE_WEIGHT * curvature

    def coherence_energy(self, candidate):
        if not self.history:
            return torch.tensor(0.0)
        last = self.history[-1]
        phase = F.cosine_similarity(candidate, last, dim=0)
        phase_term = 1 - phase
        if len(self.history) >= 3:
            d1 = poincare_distance(self.history[-1], self.history[-2])
            d2 = poincare_distance(candidate, self.history[-1])
            curvature = torch.abs(d2 - d1)
        else:
            curvature = torch.tensor(0.0)
        return COHERENCE_WEIGHT * phase_term + CURVATURE_WEIGHT * curvature

    def evolve(self, input_vec, source="user"):
        self.total_inputs += 1

        # ⭐ direction = input pull + buffer field + momentum
        direction  = input_vec - self.state
        direction += self.buffer.field_force(self.state)
        direction += MOMENTUM * self.velocity

        candidate  = exp_map(self.state, ENERGY_STEP * direction)
        energy_new = self.coherence_energy(candidate)
        energy_old = self.coherence_energy(self.state)
        margin     = ENERGY_MARGIN + self._rejection_streak * 0.01
        delta_e    = energy_new - energy_old

        if delta_e <= margin:
            self._rejection_streak = 0
            self.history.append(self.state.clone())
            dist = poincare_distance(candidate, self.state)
            self.tau     += dist.item()
            self.velocity = candidate - self.state   # ⭐ update velocity
            self.state    = candidate
            self.tree.write_concept(self.state, {
                'source':    source,
                'energy':    energy_new.item(),
                'tau':       self.tau,
                'timestamp': time.time(),
            })
            self.accepted += 1
            if self.accepted % self._save_interval == 0:
                self._save()
                print(f"[Save] Checkpoint: {self.accepted} aceites, τ={self.tau:.4f}")
            streak_info = f" [margem={margin:.3f}]" if margin > ENERGY_MARGIN else ""
            print(f"[Aceite] τ={self.tau:.4f} ΔE={delta_e.item():.4f}{streak_info}")
            return True
        else:
            self._rejection_streak += 1
            if source != 'buffer':
                self.buffer.add(candidate, {'source': source}, tau=self.tau)
            self.rejected += 1
            print(f"[Buffer] ΔE={delta_e.item():.4f} streak={self._rejection_streak}")
            return False

    def evolve_batch(self, input_vecs, sources=None, batch_size=64):
        """
        Process a list of input vectors efficiently.

        Pipeline:
          1. Split into mini-batches
          2. For each mini-batch, compute candidate positions on GPU
          3. Compute energy for all candidates in parallel on GPU
          4. Integrate sequentially only those below energy threshold
             (sequential because each acceptance changes state)

        Returns list of bools (accepted/rejected per input).
        """
        if sources is None:
            sources = ['batch'] * len(input_vecs)

        results = []
        for start in range(0, len(input_vecs), batch_size):
            batch_vecs = input_vecs[start:start+batch_size]
            batch_srcs = sources[start:start+batch_size]

            # 🚨 FIX B: buffer force computed ONCE per mini-batch
            buffer_force = self.buffer.field_force(self.state)

            candidates = []
            for vec in batch_vecs:
                direction = vec - self.state + buffer_force + MOMENTUM * self.velocity
                candidate = exp_map(self.state, ENERGY_STEP * direction)
                candidates.append(candidate)

            cand_tensor  = torch.stack(candidates)
            energies_new = self.coherence_energy_batch(cand_tensor)

            # 🚨 FIX A: recompute energy_old after each acceptance
            for i, (vec, src, candidate) in enumerate(
                    zip(batch_vecs, batch_srcs, candidates)):
                self.total_inputs += 1
                energy_old = self.coherence_energy(self.state)
                margin     = ENERGY_MARGIN + self._rejection_streak * 0.01
                delta_e    = energies_new[i] - energy_old

                if delta_e <= margin:
                    self._rejection_streak = 0
                    self.history.append(self.state.clone())
                    dist = poincare_distance(candidate, self.state)
                    self.tau     += dist.item()
                    self.velocity = candidate - self.state   # ⭐ velocity
                    self.state    = candidate
                    self.tree.write_concept(self.state, {
                        'source':    src,
                        'energy':    energies_new[i].item(),
                        'tau':       self.tau,
                        'timestamp': time.time(),
                    })
                    self.accepted += 1
                    if self.accepted % self._save_interval == 0:
                        self._save()
                        print(f"[Save] Checkpoint: {self.accepted} aceites, τ={self.tau:.4f}")
                    streak_info = f" [margem={margin:.3f}]" if margin > ENERGY_MARGIN else ""
                    print(f"[Aceite] τ={self.tau:.4f} ΔE={delta_e.item():.4f}{streak_info}")
                    results.append(True)
                else:
                    self._rejection_streak += 1
                    if src != 'buffer':
                        self.buffer.add(candidate, {'source': src}, tau=self.tau)
                    self.rejected += 1
                    print(f"[Buffer] ΔE={delta_e.item():.4f} streak={self._rejection_streak}")
                    results.append(False)

        return results

    def reassess_buffer(self, radius):
        before = self.buffer.size()
        self.buffer.reassess(self, radius)
        after = self.buffer.size()
        if before != after:
            print(f"[Buffer] Reavaliado: {before} -> {after} itens")


    # =========================================================================
    # DREAM CYCLE
    # Three phases per RG theory:
    #   1. Relaxation   — state drifts under memory field, no external input
    #   2. Reconciliation — buffer reassessed
    #   3. RG step      — nearby attractors fused into geodesic midpoints
    # =========================================================================

    def _dream_step(self, force_scale=1.5):
        """
        Single relaxation step: state moves under memory field only.
        More permissive energy margin — dreaming explores more freely.
        """
        force     = self.buffer.field_force(self.state)
        direction = force * force_scale + MOMENTUM * self.velocity
        if torch.norm(direction) < EPS:
            return False

        candidate  = exp_map(self.state, ENERGY_STEP * direction)
        energy_new = self.coherence_energy(candidate)
        energy_old = self.coherence_energy(self.state)

        if (energy_new - energy_old) <= ENERGY_MARGIN * 2.0:
            self.history.append(self.state.clone())
            dist          = poincare_distance(candidate, self.state)
            self.tau     += dist.item()
            self.velocity = candidate - self.state
            self.state    = candidate
            return True
        return False

    def abstract_memory(self, merge_radius=0.15, samples=200):
        """
        RG step: fuse nearby stored concepts into exact geodesic midpoints.

        mid(a, b) = a ⊕ (0.5 ⊙ ((-a) ⊕ b))
        where ⊙ is exact Möbius scalar multiplication.

        Pair selection: find vec near state, then find vec2 near vec
        and measure the direct distance between vec and vec2 —
        not the distance from the perturbed point (previous bug).
        """
        merged = 0
        for _ in range(samples):
            # Find concept nearest to current state
            vec, meta, dist = self.tree.find_nearest(self.state)
            if vec is None:
                break
            vec = vec.to(self.state.device)

            # Find concept nearest to vec (with small perturbation to
            # avoid returning vec itself from the same tree path)
            noise     = torch.randn_like(vec) * 0.01
            perturbed = to_poincare(vec + noise)
            vec2, meta2, _ = self.tree.find_nearest(perturbed)
            if vec2 is None:
                continue
            vec2 = vec2.to(self.state.device)

            # FIX: measure direct distance between vec and vec2,
            # not the distance from the perturbed point
            actual_dist = poincare_distance(vec, vec2).item()

            if actual_dist < 1e-4:
                continue  # same node, skip

            if actual_dist < merge_radius:
                # Exact geodesic midpoint via Möbius scalar multiplication
                centroid = geodesic_midpoint(vec, vec2)

                self.tree.write_concept(centroid, {
                    'source':    'abstraction',
                    'tau':       self.tau,
                    'merged':    True,
                    'dist':      float(actual_dist),
                    'timestamp': time.time(),
                })
                merged += 1

                # Pull state toward new abstraction
                # Strength proportional to proximity — closer = stronger pull
                pull_strength = 0.05 * (1.0 - actual_dist / merge_radius)
                pull          = mobius_add(-self.state, centroid)
                self.state    = to_poincare(
                    self.state + pull_strength * pull)

        if merged > 0:
            print(f"[Dream/RG] abstrações formadas: {merged}")
        return merged

    def dream(self, cycles=500, merge_radius=0.15, rg_every=40, verbose=True):
        """
        Full dream cycle.

        cycles       : total relaxation steps
        merge_radius : attractor fusion threshold
        rg_every     : run abstract_memory every N cycles

        Phases per cycle:
          every step    → _dream_step() (relaxation)
          every 10 steps → buffer.reassess() (reconciliation)
          every rg_every → abstract_memory() (RG / abstraction formation)
          every 200 steps → checkpoint save
        """
        print(f"\n[Dream] Iniciando {cycles} ciclos "
              f"| τ={self.tau:.4f} | buffer={self.buffer.size()} "
              f"| merge_radius={merge_radius}")

        moved = 0
        fused = 0

        for k in range(1, cycles + 1):

            # Phase 1 — relaxation
            if self._dream_step():
                moved += 1

            # Phase 2 — buffer reconciliation
            if k % 10 == 0:
                before = self.buffer.size()
                self.buffer.reassess(self, radius=merge_radius * 2)
                after  = self.buffer.size()
                if before != after and verbose:
                    print(f"  [Dream/{k}] buffer: {before}→{after}")

            # Phase 3 — RG abstraction step
            if k % rg_every == 0:
                n      = self.abstract_memory(merge_radius=merge_radius,
                                              samples=min(100, cycles // 5))
                fused += n
                if n > 0:
                    # Flush immediately — abstractions are rare and valuable
                    self.tree.flush()

            # Checkpoint
            if k % 200 == 0:
                self._save()
                print(f"  [Dream/{k}/{cycles}] τ={self.tau:.4f} "
                      f"moved={moved} fused={fused}")

        self._save()
        print(f"[Dream] Concluído | moved={moved} | fused={fused} | τ={self.tau:.4f}")
        return {'moved': moved, 'fused': fused, 'tau': self.tau}

    def get_stats(self):
        return {
            'tau': self.tau,
            'history_len': len(self.history),
            'state_norm': torch.norm(self.state).item(),
            'total_inputs': self.total_inputs,
            'accepted': self.accepted,
            'rejected': self.rejected,
            'buffer_size': self.buffer.size(),
        }

# =============================================================================
# OUTPUT
# =============================================================================

def nearest_concept(soul):
    """
    Encontra o conceito armazenado mais próximo do estado actual,
    usando a busca heurística da árvore.
    """
    vec, meta, dist = soul.tree.find_nearest(soul.state)
    if vec is not None:
        return meta, dist
    return None, float('inf')

def state_to_text(state):
    raw = (state + 0.5) * 255.0
    raw = raw.clamp(32, 126).to(torch.int32)
    return raw.detach().cpu().numpy().tobytes().decode('ascii', errors='replace')

def produce_output(soul, mode):
    if mode == 'none':
        return
    if mode in ('retrieval', 'both'):
        meta, dist = nearest_concept(soul)
        if meta:
            src = meta.get('source', '')
            print(f'[Retrieval] dist={dist:.4f} -> "{src}"')
        else:
            print("[Retrieval] Árvore vazia.")
    if mode in ('generative', 'both'):
        print(f'[Generative] -> "{state_to_text(soul.state)}"')

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CoTa Hypernode (Árvore Dinâmica)")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--file", type=str)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--soul", default="soul.json")
    parser.add_argument("--tree", default="tree.bin", help="Ficheiro da árvore")
    parser.add_argument("--buffer-file", default="buffer.json")
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--reassess-radius", type=float, default=0.5)
    parser.add_argument("--reassess-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--output-mode", default="none",
                        choices=["none", "retrieval", "generative", "both"])
    parser.add_argument("--tree-size-mb", type=int, default=10,
                        help="Tamanho inicial do ficheiro da árvore em MB")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for GPU processing (default: 64)")
    parser.add_argument("--path-bits", type=int, default=16,
                        help="Número de bits do caminho (profundidade da árvore)")
    # Dream mode
    parser.add_argument("--start-dreaming", action="store_true",
                        help="Entra em modo de sonho após processar inputs")
    parser.add_argument("--dream-cycles", type=int, default=500,
                        help="Ciclos de sonho (default: 500)")
    parser.add_argument("--dream-merge-radius", type=float, default=0.15,
                        help="Raio de fusão de atractores (default: 0.15)")
    parser.add_argument("--dream-rg-every", type=int, default=40,
                        help="Passos RG a cada N ciclos (default: 40)")
    args = parser.parse_args()

    # Ajusta a constante PATH_BITS global (simplificação)
    global PATH_BITS
    PATH_BITS = args.path_bits

    if args.init:
        for f in [args.soul, args.tree, args.buffer_file]:
            if os.path.exists(f):
                os.remove(f)

    soul = Soul(args.soul, args.tree,
                buffer_capacity=args.buffer_capacity,
                buffer_file=args.buffer_file,
                save_interval=args.save_interval,
                tree_size_mb=args.tree_size_mb)

    print(f"[CoTa Hypernode] soul={args.soul} tree={args.tree}")
    print(f"tree_size_mb={args.tree_size_mb}, path_bits={args.path_bits}")

    input_count = 0

    def process(vec, source):
        nonlocal input_count
        input_count += 1
        if soul.evolve(vec, source=source):
            produce_output(soul, args.output_mode)
        if input_count % args.reassess_interval == 0:
            soul.reassess_buffer(args.reassess_radius)

    BATCH = args.batch_size

    if args.file:
        if args.binary:
            with open(args.file, 'rb') as f:
                data = f.read()
            total = max(1, len(data) // DIM)
            chunks = []
            sources = []
            for i in range(0, len(data), DIM):
                chunk = data[i:i+DIM]
                if len(chunk) < DIM:
                    chunk += b'\x00' * (DIM - len(chunk))
                chunks.append(chunk)
                sources.append(f"{args.file}@{i}")

            # Process in batches
            for b_start in range(0, len(chunks), BATCH):
                b_chunks  = chunks[b_start:b_start+BATCH]
                b_sources = sources[b_start:b_start+BATCH]
                vecs = batch_bytes_to_vector(b_chunks)
                results = soul.evolve_batch(vecs, b_sources, batch_size=BATCH)
                accepted_count = sum(results)
                print(f"\r[Block {b_start+len(b_chunks)}/{total}] "
                      f"aceites={accepted_count}/{len(results)}",
                      end='', flush=True)
                if args.output_mode != 'none':
                    for j, ok in enumerate(results):
                        if ok:
                            produce_output(soul, args.output_mode)
                input_count += len(b_chunks)
                if input_count % args.reassess_interval < BATCH:
                    soul.reassess_buffer(args.reassess_radius)
            print()
        else:
            with open(args.file, encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f if l.strip()]

            for b_start in range(0, len(lines), BATCH):
                b_lines = lines[b_start:b_start+BATCH]
                vecs    = batch_bytes_to_vector(b_lines)
                results = soul.evolve_batch(vecs, [l[:50] for l in b_lines],
                                            batch_size=BATCH)
                for line, accepted in zip(b_lines, results):
                    status = "[Accepted]" if accepted else "[Buffered]"
                    print(f"{status} >> {line[:80]}")
                    if args.output_mode != 'none' and accepted:
                        produce_output(soul, args.output_mode)
                input_count += len(b_lines)
                if input_count % args.reassess_interval < BATCH:
                    soul.reassess_buffer(args.reassess_radius)
    else:
        # Interactive mode: single input at a time (no batching needed)
        print("Modo interactivo. Ctrl+D para sair.\n")
        try:
            while True:
                line = input("> ")
                if line:
                    process(bytes_to_vector(line), line[:50])
        except EOFError:
            pass

    if args.start_dreaming:
        soul.dream(
            cycles=args.dream_cycles,
            merge_radius=args.dream_merge_radius,
            rg_every=args.dream_rg_every,
        )

    stats = soul.get_stats()
    print("\n" + "="*50)
    print(f"τ final:        {stats['tau']:.4f}")
    print(f"Total inputs:   {stats['total_inputs']}")
    print(f"Aceites:        {stats['accepted']} ({stats['accepted']/max(1,stats['total_inputs'])*100:.1f}%)")
    print(f"Rejeitados:     {stats['rejected']}")
    print(f"Buffer size:    {stats['buffer_size']}")
    print(f"Norma do estado:{stats['state_norm']:.4f}")
    print("="*50)

    soul.close()

if __name__ == "__main__":
    main()