#!/usr/bin/env python3
"""
CoTa Hypernode — Unified Research Prototype (ÁRVORE COM PONTEIROS - FINAL)
==========================================================================
Author: Pedro R. Andrade - pedrorandrade.substack.com
github.com/pedrora/CoT 

Development round #100 or so.

Combina:
- Geometria hiperbólica correcta (adição de Möbius + mapa exponencial)
- Critério de aceitação baseado em energia
- Campo de força da memória (buffer)
- Persistência completa: soul.json, tree.bin, buffer.json
- Checkpoints periódicos
- Ingestão de ficheiros binários
- Modos de saída: retrieval e generativo
- Limiares adaptativos
- **Árvore binária com nós ligados por offsets** (crescimento linear)
- **Marcador mágico** para verificação de integridade
- **Cabeçalho com raiz e próximo offset** para gestão correcta do ficheiro
- e muito mais...
Uso:
    python cota_hypernode.py [--init] [-h --help]
    '-- init' RESETS ALL FILES - don't use it unless you are intending on 
    having to redo learning from scratch
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

def fisher_norm(x):
    """
    Normalised Fisher Information weight at point x in Poincaré disk.
    fisher_norm = 1 - (1 - ||x||²)²  ∈ [0, 1)
    = 0 at origin (flat, low penalty)
    → 1 at boundary (maximally curved, high penalty)
    Implements γ I[ρ] influence on λ R_COH in the ToAE equation.
    """
    r2 = torch.sum(x * x).clamp(max=1.0 - EPS)
    return 1.0 - (1.0 - r2) ** 2

def fisher_norm_batch(xs):
    """
    Vectorised fisher_norm for [N, DIM] tensor.
    Returns [N] tensor.
    """
    r2 = torch.sum(xs * xs, dim=1).clamp(max=1.0 - EPS)  # [N]
    return 1.0 - (1.0 - r2) ** 2

def log_map(base, vec):
    """
    Logarithmic map: geodesic direction from base to vec in tangent space.
    log_map(base, vec) = (-base) ⊕ vec
    """
    return mobius_add(-base, vec)

def exp_map(x, v):
    return mobius_add(x, v)

# =============================================================================
# ENCODING
# =============================================================================

_POLY_XS: torch.Tensor = torch.tensor(
    [(i + 1) / (DIM + 1) for i in range(DIM)], dtype=torch.float32)


def bytes_to_vector(data, device=None):
    """
    Vandermonde polynomial embedding.
    p(x) = sum_j (b_j/255) * x^j evaluated at DIM fixed points.
    Injective, order-preserving, deterministic.
    Longer sequences sit deeper in the Poincare disk.
    """
    if device is None: device = DEVICE
    if isinstance(data, str): data = data.encode('utf-8')
    raw = list(data); n = len(raw)
    if n == 0: return torch.zeros(DIM, device=device)
    xs = _POLY_XS.to(device)
    b  = torch.tensor([c/255.0 for c in raw], dtype=torch.float32, device=device)
    j  = torch.arange(n, dtype=torch.float32, device=device)
    V  = xs.unsqueeze(1) ** j.unsqueeze(0)   # (DIM, n)
    v  = V @ b                                # (DIM,)
    norm = v.norm()
    if norm < 1e-8: return torch.zeros(DIM, device=device)
    radius = math.tanh(math.log1p(norm.item()) * 0.3)
    return (v / norm) * radius

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

    def read_concept(self, idx, _file_size=None):
        offset = idx * self.node_size
        fs = _file_size if _file_size is not None else os.path.getsize(self.filename)
        if offset + self.node_size > fs:
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
            vec, meta = self.read_concept(idx, _file_size=_cached_file_size)
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

    def find_k_nearest(self, coord, k=10, max_depth=8):
        """
        Beam search for k nearest nodes.

        Maintains a priority queue (min-heap by distance) of
        (dist, idx, depth) candidates. At each step expands the
        closest unexpanded node. Guarantees the k nearest nodes
        within the explored subtree.

        Search starts from the leaf that coord maps to, then
        walks up PATH_BITS levels, expanding each ancestor
        subtree — same strategy as find_nearest but tracking k.
        """
        import heapq

        file_size = os.path.getsize(self.filename)
        results   = []  # max-heap of (-dist, vec, meta): keep k smallest
        visited   = set()

        # Priority queue: (lower_bound_dist, node_idx, depth)
        # Lower bound = distance from coord to the node IF it has data,
        # else 0 (we must visit to find out)
        frontier = []

        MAX_VISITED = k * 200

        def push(idx, depth):
            if idx in visited or len(visited) > MAX_VISITED:
                return
            if (idx + 1) * self.node_size > file_size:
                return
            visited.add(idx)
            vec, meta = self.read_concept(idx, _file_size=file_size)
            if vec is not None:
                d = poincare_distance(
                    coord, vec.to(coord.device)).item()
                if len(results) < k:
                    heapq.heappush(results, (-d, idx, vec, meta))
                elif d < -results[0][0]:
                    heapq.heapreplace(results, (-d, idx, vec, meta))
                heapq.heappush(frontier, (d, 2*idx+1, depth+1))
                heapq.heappush(frontier, (d, 2*idx+2, depth+1))
            elif depth < max_depth:
                heapq.heappush(frontier, (float("inf"), 2*idx+1, depth+1))
                heapq.heappush(frontier, (float("inf"), 2*idx+2, depth+1))

        # Seed: path from root to coord leaf, then walk up
        path = coord_to_path(coord)
        leaf_idx = 0
        for i in range(PATH_BITS):
            bit      = (path >> (PATH_BITS - 1 - i)) & 1
            leaf_idx = 2 * leaf_idx + 1 + bit

        for up in range(max_depth + 1):
            ancestor = leaf_idx
            for _ in range(up):
                ancestor = (ancestor - 1) // 2
            push(ancestor, up)

        # Expand frontier until k results confirmed or exhausted
        while frontier:
            lb, idx, depth = heapq.heappop(frontier)
            # Pruning: if lower bound >= worst result, stop
            if len(results) == k and lb >= -results[0][0]:
                break
            if depth > max_depth:
                continue
            push(idx, depth)

        # Return sorted by ascending distance
        out = sorted([(-d, vec, meta)
                      for d, idx, vec, meta in results])
        return out  # list of (dist, vec, meta)

    def flush(self):
        self._flush_queue()  # writes queued concepts then fsyncs

    def close(self):
        self.flush()
        self._file.close()



def _normalise_bytes(raw):
    """Coerce raw to bytes: pass through bytes, decode hex str, else empty."""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        try:
            return bytes.fromhex(raw)
        except ValueError:
            return raw.encode('utf-8', errors='replace')
    return b''

class RejectedLog:
    """
    Persistent store of rejected inputs — HyperbolicTree binary format.

    Replaces rejected.jsonl:
      - writes are queued and flushed in bulk (no per-entry disk I/O)
      - geometric deduplication via find_nearest (radius 0.15)
      - reconciliation score = pressure / (1 + hyperbolic_dist_to_soul)
      - rejected.bin is readable by cota_diagnostics.py for cluster analysis
    """

    PRESSURE_GAIN  = 0.2
    PRESSURE_FLOOR = 0.25
    MAX_PRESSURE   = 8.0
    DEDUP_RADIUS   = 0.15

    _SKIP = {"buffer", "self_reflection", "dream",
             b"buffer", b"assimilation", b"self_reflection",
             b"dream_reconcile", b"dream"}

    def __init__(self, path="rejected.bin", tree_size_mb=2):
        self.path = path
        self.tree = HyperbolicTree(path, initial_size_mb=tree_size_mb)
        self._meta = {}
        self._load_meta()

    def _load_meta(self):
        """Scan tree file and rebuild in-memory pressure/metadata index."""
        self._meta = {}
        if not os.path.exists(self.path):
            return
        file_size = os.path.getsize(self.path)
        node_size = self.tree.node_size
        self.tree._file.seek(0)
        for idx in range(file_size // node_size):
            offset = idx * node_size
            self.tree._file.seek(offset)
            data = self.tree._file.read(node_size)
            if len(data) < node_size:
                break
            has_data, vec, meta = self.tree._unpack_node(data)
            if has_data and meta and 'pressure' in meta:
                self._meta[idx] = meta

    def record(self, source, vector, delta_e, tau):
        """
        Record a rejected input. Write is queued — flushed at checkpoint.
        Geometrically deduplicates: if a similar concept is already stored,
        its pressure is increased instead of creating a new entry.
        """
        if source in self._SKIP:
            return

        if isinstance(source, (bytes, bytearray)):
            src_text = source.decode('utf-8', errors='replace')
        elif isinstance(source, str):
            src_text = source
        else:
            src_text = str(source)
        src_text = src_text.strip()[:200]

        vec_t = (vector if isinstance(vector, torch.Tensor)
                 else torch.tensor(vector, dtype=torch.float32))

        nearest_vec, nearest_meta, nearest_dist = self.tree.find_nearest(
            vec_t, max_depth=4)

        if (nearest_vec is not None and nearest_dist < self.DEDUP_RADIUS
                and 'pressure' in nearest_meta):
            meta = dict(nearest_meta)
            meta['pressure']     = min(meta['pressure'] + self.PRESSURE_GAIN,
                                       self.MAX_PRESSURE)
            meta['attempts']    += 1
            meta['last_tau']     = tau
            meta['last_delta_e'] = float(delta_e)
            self.tree.write_concept(nearest_vec.to(vec_t.device), meta)
        else:
            self.tree.write_concept(vec_t, {
                'pressure':     self.PRESSURE_GAIN,
                'attempts':     1,
                'first_tau':    tau,
                'last_tau':     tau,
                'last_delta_e': float(delta_e),
                'source_text':  src_text,
                'timestamp':    time.time(),
            })

    def flush(self):
        """Flush queued writes and rebuild metadata index."""
        self.tree.flush()
        self._load_meta()

    def reconcile(self, soul, max_items=20, verbose=True):
        """
        Attempt to integrate high-score rejected inputs.
        Score = pressure / (1 + hyperbolic distance to current soul state).
        Higher score = more insistent AND geometrically closer = try first.
        """
        if not self._meta:
            self._load_meta()
        if not self._meta:
            return 0, 0

        state     = soul.state
        file_size = os.path.getsize(self.path)
        node_size = self.tree.node_size
        scored    = []

        for idx, meta in self._meta.items():
            offset = idx * node_size
            if offset + node_size > file_size:
                continue
            self.tree._file.seek(offset)
            data = self.tree._file.read(node_size)
            has_data, vec, _ = self.tree._unpack_node(data)
            if not has_data or vec is None:
                continue
            vec   = vec.to(state.device)
            dist  = poincare_distance(state, vec).item()
            score = meta['pressure'] / (1.0 + dist)
            scored.append((score, idx, vec, meta))

        scored.sort(key=lambda x: -x[0])
        integrated = 0
        to_clear   = []

        for score, idx, vec, meta in scored[:max_items]:
            scale = max(math.exp(-meta['pressure'] * 0.5), self.PRESSURE_FLOOR)
            orig_am  = soul.adaptive_margin
            orig_str = soul._rejection_streak
            soul.adaptive_margin   = lambda: ENERGY_MARGIN * scale
            soul._rejection_streak = 0
            accepted = soul.evolve(vec, source=b"dream_reconcile")
            soul.adaptive_margin   = orig_am
            soul._rejection_streak = orig_str

            if accepted:
                src_text = meta.get('source_text', '')
                if src_text:
                    try:
                        soul.language_index.add(soul.state, src_text)
                    except Exception:
                        pass
                to_clear.append(idx)
                integrated += 1
                if verbose:
                    label = meta.get('source_text', '?')[:50]
                    print(f"  [RejectedLog] integrado '{label}' "
                          f"score={score:.3f} p={meta['pressure']:.2f}")
            else:
                meta['pressure'] = min(meta['pressure'] + self.PRESSURE_GAIN,
                                       self.MAX_PRESSURE)
                meta['attempts'] += 1
                self.tree.write_concept(vec, meta)

        for idx in to_clear:
            offset = idx * node_size
            self.tree._write_queue[offset] = b'\x00' * node_size
            self._meta.pop(idx, None)

        self.flush()
        remaining = len(self._meta)
        if verbose and integrated > 0:
            print(f"[RejectedLog] {integrated} integrado(s), {remaining} pendente(s)")
        return integrated, remaining

    def size(self):
        return len(self._meta)

    def close(self):
        self.tree.flush()
        self.tree.close()

    def clear(self):
        self._meta = {}
        self.tree.close()
        if os.path.exists(self.path):
            os.remove(self.path)
        self.tree = HyperbolicTree(self.path)


def curvature_at(v_prev, v_curr, v_next):
    """
    Hyperbolic angle at v_curr formed by geodesics v_prev->v_curr and v_curr->v_next.
    Uses hyperbolic law of cosines: cos θ = (cosh a · cosh b - cosh c) / (sinh a · sinh b)
    Returns 0 if any segment is degenerate.
    """
    a = poincare_distance(v_prev, v_curr).item()
    b = poincare_distance(v_curr, v_next).item()
    c = poincare_distance(v_prev, v_next).item()
    if a < EPS or b < EPS:
        return 0.0
    cos_theta = (math.cosh(a) * math.cosh(b) - math.cosh(c)) / (
                 math.sinh(a) * math.sinh(b))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


class FragmentAccumulator:
    """
    Incremental accumulator for a narrative fragment — a thread of bytes
    that converges toward a local attractor in the hyperbolic field.

    Each fragment is a local compression unit: a sequence of bytes whose
    trajectory converges. The natural cut point is where the thread ends —
    detected when fragment energy grows consistently for CONFIRM_WINDOW steps.

    CONFIRM_WINDOW = 3 is the geometric minimum: curvature requires 3 points,
    so 3 steps is the minimum to distinguish fluctuation from divergence.

    Energy: E = d(centroid, soul_state) * total_length
    The centroid is the Fréchet mean (incremental approximation).
    total_length is the geodesic length of the trajectory — it captures
    both order and structure, making explicit positional encoding unnecessary:
    "gato" and "otag" have different trajectories and different lengths.

    No arbitrary parameters. The geometry decides the cuts.
    """

    CONFIRM_WINDOW = 3

    def __init__(self, first_point):
        self.centroid      = first_point.clone()
        self.total_length  = 0.0
        self.total_curv    = 0.0
        self.last_point    = first_point.clone()
        self.penultimate   = None
        self.count         = 1
        self._prev_energy  = float('inf')
        self._growing      = 0   # consecutive steps where energy grew

    def add_point(self, point):
        # Geodesic length
        dist = poincare_distance(self.last_point, point).item()
        self.total_length += dist

        # Curvature at previous point (requires 3 points)
        if self.penultimate is not None:
            self.total_curv += curvature_at(
                self.penultimate, self.last_point, point)

        # Incremental Fréchet mean: one gradient step toward new point
        # Step size 1/(n+1) ensures convergence to true Fréchet mean
        step = (1.0 / (self.count + 1)) * log_map(self.centroid, point)
        self.centroid = to_poincare(exp_map(self.centroid, step))

        self.penultimate = self.last_point
        self.last_point  = point.clone()
        self.count      += 1

    def energy(self, soul_state):
        """
        Fragment energy relative to soul state.
        E = d(centroid, state) * total_length

        total_length already encodes trajectory structure (order, curvature).
        No alpha weighting needed — curvature is implicit in the path length.
        A straight trajectory has minimum length for its displacement.
        A curved trajectory is longer — higher energy — naturally.
        """
        if self.total_length < EPS:
            return 0.0
        d = poincare_distance(self.centroid, soul_state).item()
        return d * self.total_length

    def should_cut(self, soul):
        """
        Narrative thread termination — anchored in Recursive Coherence.

        Three independent criteria, each geometrically or dynamically
        founded. No arbitrary thresholds.

        1. Velocity excess: |dE/dt| > V_max = Θ/τ
           The fragment is changing faster than the system can process.
           Θ = adaptive_margin() (fragility threshold)
           τ = soul.tau (accumulated tension — processing history)

        2. Phase misalignment: cos(frag_dir, soul.velocity) < -0.5
           cos(-0.5) = 120° — fragment actively opposing identity vector.
           Beyond this angle the fragment is outside the Beverly Band.

        3. Sustained energy growth: CONFIRM_WINDOW=3 consecutive steps.
           Geometric minimum: curvature requires 3 points.
           Distinguishes fluctuation from divergence.
        """
        E  = self.energy(soul.state)
        if self._prev_energy == float('inf'):
            self._prev_energy = E
            return False
        dE = E - self._prev_energy

        # ── Criterion 1: velocity excess ─────────────────────────────
        V_max = soul.adaptive_margin() / (soul.tau + EPS)
        if dE > V_max:   # only cut on growth — drops mean improving compatibility
            self._growing = 0
            self._prev_energy = E
            return True

        # ── Criterion 2: phase misalignment ──────────────────────────
        # Direction from soul to fragment centroid — global alignment,
        # not just last step. Measures whether the fragment as a whole
        # is pulling the soul away from its identity vector.
        if self.count > 1 and torch.norm(soul.velocity) > EPS:
            dir_to_frag = log_map(soul.state, self.centroid)
            if torch.norm(dir_to_frag) > EPS:
                cos_psi = F.cosine_similarity(
                    dir_to_frag.unsqueeze(0),
                    soul.velocity.unsqueeze(0)).item()
                if cos_psi < -0.5:   # outside Beverly Band (>120°)
                    self._growing = 0
                    self._prev_energy = E
                    return True

        # ── Criterion 3: sustained energy growth ─────────────────────
        if dE > 0:
            self._growing += 1
        else:
            self._growing     = 0
            self._prev_energy = E
        return self._growing >= self.CONFIRM_WINDOW

    def parallel_transport_dir(self):
        """
        Direction of this fragment's trajectory — used to pass narrative
        context to the next fragment via approximate parallel transport.
        Returns the log_map from penultimate to last point (tangent vector).
        """
        if self.penultimate is None:
            return None
        return log_map(self.penultimate, self.last_point)


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
            # Evict highest-pressure item (most processed), not oldest
            # This keeps young items alive for assimilation
            evict = max(self.items.keys(),
                        key=lambda k: self.items[k]['pressure'])
            del self.items[evict]
        self.items[self.next_id] = {
            'vector':    vector.clone(),
            'metadata':  metadata,
            'attempts':  0,
            'added_tau': tau,
            'timestamp': time.time(),
            'pressure':  0.0,   # grows with time + attempts
        }
        self.next_id += 1

    def field_force(self, state):
        if not self.items:
            return torch.zeros(DIM, device=state.device)
        items_list = list(self.items.values())
        vecs     = torch.stack([it['vector'] for it in items_list])
        d        = poincare_distance_batch(state, vecs)             # [N]
        # Pressure amplifies weight: persistent items pull harder
        p        = torch.tensor([it.get('pressure', 0.0) for it in items_list],
                                device=state.device)                # [N]
        w        = (torch.exp(-d) * (1.0 + p)).unsqueeze(1)        # [N,1]
        tangents = torch.stack([log_map(state, v) for v in vecs])   # [N,DIM]
        force    = (w * tangents).sum(dim=0)                        # [DIM]
        return MEMORY_FORCE * force / (len(self.items) + EPS)

    def competing_attractor(self, state):
        """
        Sample one buffer item by proximity — creates internal conflict.
        Instead of averaging all memories, one competes for dominance.
        """
        if not self.items:
            return torch.zeros(DIM, device=state.device)
        vecs  = torch.stack([it['vector'] for it in self.items.values()])
        d     = poincare_distance_batch(state, vecs)
        probs = torch.softmax(-d, dim=0)
        idx   = torch.multinomial(probs, 1).item()
        return MEMORY_FORCE * log_map(state, vecs[idx])

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

    def _item_key(self, item):
        for k, v in self.items.items():
            if v is item:
                return k
        return None

    def reassess(self, soul, radius):
        for item in self.find_nearby(soul.state, radius):
            if soul.evolve(item['vector'], source='buffer'):
                k = self._item_key(item)
                if k is not None:
                    del self.items[k]
            else:
                item['attempts'] += 1
                item['pressure'] = item.get('pressure', 0.0) + 0.15
                if item['attempts'] >= self.max_attempts:
                    k = self._item_key(item)
                    if k is not None:
                        del self.items[k]

    def assimilate(self, soul, max_items=5):
        """
        Active assimilation loop: try to integrate high-pressure buffer items
        using a temporarily lowered energy threshold.
        pressure → threshold * exp(-pressure), floored at PRESSURE_FLOOR.
        High-pressure items = things the user insisted on but system resisted.
        Called after dreaming to prevent post-dream isolation.
        """
        if not self.items:
            return 0
        # Sort by pressure descending — most insistent items first
        # Also filter by distance: no point attempting items very far from state
        state = soul.state
        candidates = []
        for k, item in self.items.items():
            if item.get('pressure', 0.0) < 0.1:
                continue
            dist = poincare_distance(state, item['vector']).item()
            candidates.append((k, item, dist))
        # Within assimilation radius (2x normal reassess radius) or top-pressure
        candidates.sort(key=lambda x: -x[1].get('pressure', 0.0))
        integrated = 0
        for k, item, dist in candidates[:max_items]:
            # Skip items too distant even for assimilation
            # (pressure opens margin but cannot bridge large semantic gaps)
            if dist > 2.0:
                item['pressure'] += 0.05  # still accumulate, slower
                continue
            # Temporarily lower margin proportional to pressure
            scale  = math.exp(-item['pressure'])
            scale  = max(scale, 0.25)  # never colapsar coerência
            orig   = soul._rejection_streak
            soul._rejection_streak = 0  # reset streak for clean attempt
            # Patch adaptive_margin temporarily
            orig_margin = ENERGY_MARGIN
            import builtins
            _orig_am = soul.adaptive_margin
            soul.adaptive_margin = lambda: orig_margin * scale
            accepted = soul.evolve(item['vector'], source=b'assimilation')
            soul.adaptive_margin = _orig_am
            soul._rejection_streak = orig
            if accepted:
                raw_src = item['metadata'].get('source', b'')
                if isinstance(raw_src, str):
                    text = raw_src
                else:
                    text = raw_src.decode('utf-8', errors='replace')
                text = text.rstrip('\x00').strip()
                if text:
                    soul.language_index.add(soul.state, text)
                #_text_a = item.rstrip(b'\x00').decode('utf-8', errors='replace').strip()
                #if _text_a:
                #    soul.language_index.add(soul.state, _text_a)
                del self.items[k]
                integrated += 1
                print(f"[Assimilação] item integrado (pressão={item['pressure']:.2f})")
            else:
                item['pressure'] += 0.15  # keep growing
        return integrated

    def save(self, filename):
        def _ser(v):
            return v.hex() if isinstance(v, (bytes, bytearray)) else v
        data = [{'vector':    item['vector'].tolist(),
                 'metadata':  {k: _ser(v) for k, v in item['metadata'].items()},
                 'attempts':  item['attempts'],
                 'added_tau': item['added_tau'],
                 'timestamp': item.get('timestamp', time.time()),
                 'pressure':  item.get('pressure', 0.0)}
                for item in self.items.values()]
        # Atomic write: temp file + rename prevents corruption on crash
        tmp = filename + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        os.replace(tmp, filename)

    def load(self, filename):
        if not os.path.exists(filename):
            return
        try:
            with open(filename, encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Buffer] Ficheiro corrompido ({e}) — a ignorar e a fazer backup.")
            import shutil
            shutil.copy(filename, filename + ".corrupt")
            os.remove(filename)
            return
        self.items = {i: {'vector':    torch.tensor(d['vector'], device=DEVICE),
                          'metadata':  d['metadata'],
                          'attempts':  d['attempts'],
                          'added_tau': d['added_tau'],
                          'timestamp': d.get('timestamp', time.time()),
                          'pressure':  d.get('pressure', 0.0)}
                      for i, d in enumerate(data)}
        self.next_id = len(data)

    def size(self):
        return len(self.items)


# =============================================================================
# SEMANTIC DECODER
# =============================================================================

class SemanticDecoder:
    """
    Collapses a continuous state vector into the nearest discrete symbol
    (word/concept) using Poincaré distance — the inverse of bytes_to_vector.

    This is the bridge between continuous hyperbolic geometry and discrete
    language. Without this, reflection loops operate on ASCII noise.
    With it, the system has semantic pressure: states are pulled toward
    known linguistic attractors.
    """

    def __init__(self, vocab_list, device=None):
        if device is None:
            device = DEVICE
        self.device = device
        self.words  = vocab_list

        if not vocab_list:
            raise ValueError("SemanticDecoder requires a non-empty vocabulary.")

        print(f"[SemanticDecoder] Construindo embedding de {len(vocab_list)} tokens...")
        self.word_vectors = batch_bytes_to_vector(vocab_list, device=device)
        print(f"[SemanticDecoder] Pronto. Shape: {self.word_vectors.shape}")

    def decode(self, state_vector, temperature=1.0):
        """
        Sample a vocabulary token using temperature over hyperbolic distances.
        temperature=0  → greedy (nearest token)
        temperature=1  → soft sampling proportional to proximity
        temperature>1  → more random / exploratory
        """
        dists = poincare_distance_batch(state_vector, self.word_vectors)
        if temperature < 1e-6:
            idx = torch.argmin(dists).item()
        else:
            # logits = -dist/T, softmax → probability over vocabulary
            logits = -dists / temperature
            probs  = torch.softmax(logits, dim=0)
            idx    = torch.multinomial(probs, 1).item()
        return self.words[idx], dists[idx].item()

    def top_k(self, state_vector, k=5):
        """Returns the k nearest vocabulary tokens with distances."""
        dists  = poincare_distance_batch(state_vector, self.word_vectors)
        topk   = torch.topk(dists, k=min(k, len(self.words)), largest=False)
        return [(self.words[i], dists[i].item()) for i in topk.indices]

    @classmethod
    def from_file(cls, filepath, device=None):
        """Load vocabulary from a text file (one word/phrase per line)."""
        with open(filepath, encoding='utf-8', errors='replace') as f:
            words = [l.strip() for l in f if l.strip()]
        print(f"[SemanticDecoder] Carregado {len(words)} tokens de '{filepath}'")
        return cls(words, device=device)

    @classmethod
    def from_corpus(cls, text, min_freq=2, max_words=5000, device=None):
        """
        Build vocabulary from raw text.
        Extracts words, filters by frequency, returns decoder.
        """
        import re
        from collections import Counter
        words   = re.findall(r"\b[a-záàâãéèêíïóôõöúüçñA-Z][a-záàâãéèêíïóôõöúüçñA-Z'\-]{1,}\b",
                             text)
        counts  = Counter(w.lower() for w in words)
        vocab   = [w for w, c in counts.most_common(max_words) if c >= min_freq]
        print(f"[SemanticDecoder] Vocabulário extraído: {len(vocab)} tokens "
              f"(min_freq={min_freq})")
        return cls(vocab, device=device)

# =============================================================================
# SOUL  (core dynamics) – LIGEIRAMENTE ADAPTADA PARA USAR A ÁRVORE
# =============================================================================


class RGOutput:
    """
    Emergent output layer. Vocabulary is built dynamically from accepted
    chunks — only content that the system integrated enters the lexicon.

    Generate: state → temperature sampling over hyperbolic distances → token
    Feedback: generated token → bytes_to_vector → re-enters as input

    The loop runs until state stabilises (Δs < threshold).
    This creates self-consistency pressure: the system learns to generate
    output that it can also accept as input — grounding language in geometry.
    """

    MAX_VOCAB = 10000

    def __init__(self, device=None):
        self.device   = device or DEVICE
        self._tokens  = []          # list of str (decoded text)
        self._vectors = []          # list of tensors [DIM] on device
        self._decoder = None        # SemanticDecoder, rebuilt on demand
        self._dirty   = False       # rebuild needed?

    def ingest(self, raw_bytes):
        """
        Extract tokens from accepted chunk bytes and add to vocabulary.
        Called every time a chunk is accepted.
        """
        if not raw_bytes:
            return
        text = raw_bytes.rstrip(b'\x00').decode('utf-8', errors='replace').strip()
        if not text:
            return

        import re
        # Extract word-like tokens (2+ chars, contain a letter)
        tokens = re.findall(
            r'[\w\u00c0-\u024f][\w\u00c0-\u024f\'\.\-]{1,}',
            text)
        for tok in tokens:
            tok = tok.strip('.\'-').lower()
            if len(tok) < 2:
                continue
            if tok not in self._tokens:
                if len(self._tokens) >= self.MAX_VOCAB:
                    # Drop oldest token (FIFO — recent experience dominates)
                    self._tokens.pop(0)
                    self._vectors.pop(0)
                self._tokens.append(tok)
                self._vectors.append(
                    bytes_to_vector(tok.encode('utf-8'), device=self.device))
                self._dirty = True

    def _ensure_decoder(self):
        if self._dirty or self._decoder is None:
            if not self._tokens:
                return False
            self._decoder = SemanticDecoder(self._tokens, device=self.device)
            self._dirty = False
        return True

    def generate(self, state, temperature=0.8, n_tokens=8):
        """
        Generate a sequence of tokens from the current state.
        Each token is sampled via temperature over hyperbolic distances.
        Returns (text, list_of_tokens).
        """
        if not self._ensure_decoder():
            return "", []

        tokens = []
        # Walk through token generation — state drifts slightly with each token
        cur_state = state.clone()
        for _ in range(n_tokens):
            word, dist = self._decoder.decode(cur_state, temperature=temperature)
            tokens.append(word)
            # Nudge state toward generated token (mild — just for next token context)
            tok_vec   = bytes_to_vector(word.encode('utf-8'), device=self.device)
            cur_state = to_poincare(cur_state + 0.1 * (tok_vec - cur_state))

        text = ' '.join(tokens)
        return text, tokens

    def feedback_loop(self, soul, temperature=0.8, n_tokens=8,
                      conv_threshold=0.002, max_iter=6):
        """
        Generate output from state, re-enter it as input, repeat until stable.

        Each iteration:
          1. Generate text from current state
          2. Encode text back to vector
          3. Apply internal feedback step
          4. Check convergence by state distance

        The system learns to produce output consistent with its own geometry.
        Returns final generated text.
        """
        if not self._ensure_decoder():
            return ""

        prev_state = soul.state.clone()
        last_text  = ""

        for iteration in range(max_iter):
            # Generate text from current state
            text, tokens = self.generate(soul.state, temperature=temperature,
                                         n_tokens=n_tokens)
            last_text = text

            if not text.strip():
                break

            # Re-encode and feed back as internal step
            feedback_vec = bytes_to_vector(
                text.encode('utf-8'), device=soul.state.device)
            soul._feedback_step(feedback_vec)

            # Convergence: state distance between iterations
            state_delta = poincare_distance(soul.state, prev_state).item()
            prev_state  = soul.state.clone()

            if state_delta < conv_threshold:
                break

        return last_text

    def vocab_size(self):
        return len(self._tokens)

def metropolis_accept(delta_e, tau):
    """Metropolis criterion T=1/τ. dE<0 always accept; dE>=0 P=exp(-dE*tau)."""
    if delta_e < 0.0:
        return True
    return torch.rand(1).item() < math.exp(min(0.0, -delta_e * tau))


class RGAnalyticsLogger:
    """Per-operation analytics for abstraction and atomization."""
    def __init__(self, jsonl_path=None, verbose=True):
        self.jsonl_path = jsonl_path; self.verbose = verbose
        self._merges = []; self._splits = []; self._cycle = 0
    def log_merge(self, d_a, d_b, d_c, delta_e, tau, accepted, label=''):
        p = 1.0 if delta_e < 0 else min(1.0, math.exp(min(0.0, -delta_e * tau)))
        self._merges.append({'d_a':round(d_a,4),'d_b':round(d_b,4),'d_c':round(d_c,4),
            'delta_e':round(delta_e,4),'p_accept':round(p,4),'tau':round(tau,4),
            'accepted':accepted,'label':label})
    def log_split(self, parent_text, frag_text, frag_len, d_parent, d_frag, accepted, tau):
        self._splits.append({'parent':parent_text[:40],'fragment':frag_text[:20],
            'frag_len':frag_len,'d_parent':round(d_parent,4),'d_frag':round(d_frag,4),
            'accepted':accepted,'tau':round(tau,4)})
    def summary(self):
        self._cycle += 1
        m, s = self._merges, self._splits
        if not m and not s: return
        tau_val = m[0]['tau'] if m else (s[0]['tau'] if s else 0)
        print(f"\n  +-- [RG Analytics] cycle={self._cycle} tau={tau_val:.3f}")
        if m:
            n_acc = sum(1 for x in m if x['accepted'])
            de_acc = [x['delta_e'] for x in m if x['accepted']]
            de_rej = [x['delta_e'] for x in m if not x['accepted']]
            print(f"  |  MERGE  {n_acc}/{len(m)} accepted")
            if de_acc: print(f"  |    dE acc : min={min(de_acc):.4f}  max={max(de_acc):.4f}  mean={sum(de_acc)/len(de_acc):.4f}")
            if de_rej: print(f"  |    dE rej : min={min(de_rej):.4f}  max={max(de_rej):.4f}  mean={sum(de_rej)/len(de_rej):.4f}")
            for x in sorted([x for x in m if x['accepted']], key=lambda x: x['delta_e'])[:3]:
                print(f"  |    + dE={x['delta_e']:+.4f}  d({x['d_a']:.3f},{x['d_b']:.3f})->{x['d_c']:.3f}" + (f" -> {x['label']}" if x['label'] else ''))
        if s:
            n_acc = sum(1 for x in s if x['accepted'])
            la = [x['frag_len'] for x in s if x['accepted']]
            lr = [x['frag_len'] for x in s if not x['accepted']]
            print(f"  |  SPLIT  {n_acc}/{len(s)} accepted")
            if la: print(f"  |    len acc : min={min(la)}  max={max(la)}  mean={sum(la)/len(la):.1f}")
            if lr: print(f"  |    len rej : min={min(lr)}  max={max(lr)}  mean={sum(lr)/len(lr):.1f}")
            for x in sorted([x for x in s if x['accepted']], key=lambda x: x['frag_len'])[:5]:
                print(f"  |    + [{x['frag_len']}b] '{x['parent']}' -> '{x['fragment']}'  d={x['d_frag']:.3f}")
        print(f"  +--------------------------------------------------")
        if self.jsonl_path:
            import json as _j
            with open(self.jsonl_path, 'a', encoding='utf-8') as f:
                _j.dump({'cycle':self._cycle,'merges':m,'splits':s}, f); f.write('\n')
        self._merges = []; self._splits = []


class Soul:
    def __init__(self, soul_file="soul.json", tree_file="tree.bin",
                 buffer_capacity=10000, buffer_file=None, save_interval=100,
                 tree_size_mb=10,
                 vocab_file="rg_vocab.txt",
                 lexicon_file="attractor_lexicon",
                 lang_file="language_index"):
        self.soul_file   = soul_file
        self.vocab_file   = vocab_file  # rg_vocab.txt
        self.lexicon_file = lexicon_file  # attractor_lexicon.npz
        self.lang_file    = lang_file     # language_index.npz
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
            self.tau     = getattr(self, '_loaded_tau', 0.0)
            self.t_exist = getattr(self, '_loaded_t_exist', 0.0)
            print(f"[Soul] Restaurado: τ={self.tau:.4f}, histórico={len(self.history)}")
        else:
            self.state   = to_poincare(torch.randn(DIM, device=DEVICE) * 0.1)
            self.history = deque(maxlen=50)
            self.tau     = 0.0
            self.t_exist = 0.0

        self.total_inputs = 0
        self.accepted = 0
        self.rejected = 0
        self._rejection_streak = 0
        self.rejected_log = None
        self.attractor_lexicon = AttractorLexicon.load(self.lexicon_file, device=DEVICE)
        if self.attractor_lexicon.size() > 0:
            print(f"[AttractorLexicon] Restaurado: {self.attractor_lexicon.active_size()} "
                  f"activos / {self.attractor_lexicon.size()} total")
        self.language_index = LanguageIndex.load(self.lang_file, device=DEVICE)
        if self.language_index.size() > 0:
            print(f"[LanguageIndex] Restaurado: {self.language_index.size()} entradas")
        self.rg_output = RGOutput(device=DEVICE)
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, encoding="utf-8") as _vf:
                _vocab = [line.strip() for line in _vf if line.strip()]
            for tok in _vocab:
                if tok not in self.rg_output._tokens:
                    self.rg_output._tokens.append(tok)
                    self.rg_output._vectors.append(
                        bytes_to_vector(tok.encode("utf-8"), device=DEVICE))
            if _vocab:
                self.rg_output._dirty = True
                print(f"[RGOutput] Vocabulário restaurado: {len(_vocab)} tokens")
        # ⭐ Velocity: cognitive inertia — smooths trajectory across steps
        loaded_vel    = getattr(self, '_loaded_velocity', None)
        self.velocity = (torch.tensor(loaded_vel, dtype=torch.float32, device=DEVICE)
                         if loaded_vel is not None
                         else torch.zeros(DIM, device=DEVICE))
        self._cached_expansion  = 1.0
        self._expansion_counter = 0

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
        self._loaded_t_exist  = data.get('t_exist', 0.0)
        self._loaded_velocity = data.get('velocity', None)

    def _save(self):
        self.tree.flush()
        # soul.json — always written first, atomically
        tmp = self.soul_file + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({
                'soul_id': self.soul_id,
                'created': self.created,
                'state':    self.state.tolist() if hasattr(self, 'state') else None,
                'history':  [s.tolist() for s in self.history]
                            if hasattr(self, 'history') else [],
                'tau':      self.tau if hasattr(self, 'tau') else 0.0,
                't_exist':  self.t_exist if hasattr(self, 't_exist') else 0.0,
                'velocity': self.velocity.tolist() if hasattr(self, 'velocity') else None,
            }, f)
        os.replace(tmp, self.soul_file)
        cwd = os.getcwd()

        #print(f"DEBUG _save: attractor_lexicon size={self.attractor_lexicon.size()}")
        #print(f"DEBUG _save: language_index size={self.language_index.size()}")
        #print(f"DEBUG _save: rg_output size={self.rg_output.vocab_size()}")

        # Secondary files — written independently, failures logged not raised
        if hasattr(self, "attractor_lexicon"):
            try:
                self.attractor_lexicon.save(self.lexicon_file)
                stem = self.lexicon_file[:-4] if self.lexicon_file.endswith('.npz') else self.lexicon_file
                print(f"[Save] attractor_lexicon.npz ({self.attractor_lexicon.size()} entries) -> {os.path.join(cwd, stem+'.npz')}")
            except Exception as e:
                print(f"[Save] attractor_lexicon error: {e}")
        if hasattr(self, "language_index"):
            try:
                self.language_index.save(self.lang_file)
                print(f"[Save] language_index.bin ({self.language_index.size()} entries)")
            except Exception as e:
                print(f"[Save] language_index error: {e}")
        if hasattr(self, "rg_output"):
            try:
                tmp_v = self.vocab_file + ".tmp"
                with open(tmp_v, "w", encoding="utf-8") as vf:
                    vf.write("\n".join(self.rg_output._tokens))
                os.replace(tmp_v, self.vocab_file)
            except Exception as e:
                print(f"[Save] rg_vocab error: {e}")
        if self.buffer_file:
            try:
                self.buffer.save(self.buffer_file)
            except Exception as e:
                print(f"[Save] buffer error: {e}")
                
        #print(f"DEBUG after _save: attractor_lexicon size={self.attractor_lexicon.size()}")
        #print(f"DEBUG after _save: language_index size={self.language_index.size()}")
        #print(f"DEBUG after _save: rg_output size={self.rg_output.vocab_size()}")

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

        last = self.history[-1]  # [DIM]

        # Phase term: hyperbolic distance scaled by Fisher weight
        d_phase    = poincare_distance_batch(last, candidates)  # [N]
        phase_term = d_phase / (d_phase + 1.0)
        fi         = fisher_norm_batch(candidates)              # [N] gamma*I[rho]
        phase_term = phase_term * (1.0 + fi)                   # curvature-aware

        # Curvature term — fully vectorised
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
        """
        E = COHERENCE_WEIGHT * phase_term * (1 + I[candidate])
          + CURVATURE_WEIGHT * curvature
        where I[x] = fisher_norm(x) is the normalised Fisher Information
        weight (conformal factor of Poincare disk).
        This implements the gamma*I[rho] influence on lambda*R_COH
        from the ToAE equation: near-boundary states cost more.
        """
        if not self.history:
            return torch.tensor(0.0, device=candidate.device)
        last       = self.history[-1]
        d_phase    = poincare_distance(candidate, last)
        phase_term = d_phase / (d_phase + 1.0)
        # Fisher weight: scales energy with local curvature
        fi         = fisher_norm(candidate)           # gamma*I[rho] proxy
        phase_term = phase_term * (1.0 + fi)         # lambda scales with curvature
        if len(self.history) >= 3:
            d1 = poincare_distance(self.history[-1], self.history[-2])
            d2 = poincare_distance(candidate, self.history[-1])
            curvature = torch.abs(d2 - d1)
        else:
            curvature = torch.tensor(0.0, device=candidate.device)
        return COHERENCE_WEIGHT * phase_term + CURVATURE_WEIGHT * curvature

    def compute_susceptibility(self):
        """
        Susceptibility chi = mean_displacement * acceptance_rate / (1 + fisher)
          chi very low  -> rigid narrative lock (post-dream refractory)
          chi rising    -> attractor weakening, window opening
          chi > ~0.03   -> learning window open
        """
        if len(self.history) < 2:
            return 0.0
        hist = list(self.history)
        displacements = [
            poincare_distance(hist[i], hist[i-1]).item()
            for i in range(1, len(hist))
        ]
        mean_move       = sum(displacements) / len(displacements)
        acceptance_rate = self.accepted / max(1, self.total_inputs)
        fi              = fisher_norm(self.state).item()
        return mean_move * acceptance_rate / (1.0 + fi)

    def adaptive_margin(self):
        """
        Acceptance margin adapted to Jacobian regime (cached every 50 accepts).
        collapse (rho<0.8): open margin to force exploration
        critical (0.8-1.2): nominal
        chaos    (rho>1.2): closed margin to stabilise
        """
        self._expansion_counter += 1
        if self._expansion_counter >= 50:
            self._cached_expansion  = self.directional_expansion_rate()
            self._expansion_counter = 0
        rate = self._cached_expansion
        base = ENERGY_MARGIN + self._rejection_streak * 0.01
        if rate < 0.8:
            scale = 1.0 + (0.8 - rate) * 2.5   # collapse: open margin
        elif rate > 1.2:
            scale = max(0.3, 1.0 - (rate - 1.2) * 1.5)  # chaos: close margin
        else:
            scale = 1.0
        return min(base * scale, ENERGY_MARGIN * 5.0)

    def tick(self, dt=1.0, entropy_rate=0.001, gravity_rate=0.003):
        """
        Advance existence time.
        Two forces:
          entropy_rate: small isotropic noise — cost of staying frozen
          gravity_rate: semantic gravity toward buffered items
                        weighted by their pressure (user insistence)
        x_{t+1} = x_t + gravity·Σ w_i·log_map(x_t, b_i) + noise
        """
        self.t_exist += dt
        r       = torch.norm(self.state)
        damping = (1.0 - r ** 2).clamp(min=0.01)
        # Semantic gravity: drift toward high-pressure buffer items
        gravity = torch.zeros_like(self.state)
        if self.buffer.size() > 0:
            items = list(self.buffer.items.values())
            # Only items with meaningful pressure contribute to gravity
            # Prevents unrelated zero-pressure items from averaging out the pull
            active = [it for it in items if it.get('pressure', 0.0) >= 0.1]
            if active:
                vecs  = torch.stack([it['vector'] for it in active])
                pres  = torch.tensor([it['pressure'] for it in active],
                                      device=self.state.device)
                w       = pres / (pres.sum() + EPS)
                dirs    = torch.stack([log_map(self.state, v) for v in vecs])
                gravity = (w.unsqueeze(1) * dirs).sum(dim=0)
        # Isotropic entropy noise
        noise = torch.randn_like(self.state) * entropy_rate * dt
        delta = (gravity_rate * dt * gravity + noise) * damping
        self.state = to_poincare(self.state + delta)
        return float(torch.norm(gravity).item())

    def evolve(self, input_vec, source="user"):
        self.total_inputs += 1

        # ⭐ direction = input pull + buffer field + momentum
        # Radial damping: (1-r²) prevents boundary runaway
        _r         = torch.norm(self.state)
        _damping   = (1.0 - _r ** 2).clamp(min=0.01)
        direction  = log_map(self.state, input_vec) * _damping
        if self.buffer.size() > 1 and torch.rand(1).item() < 0.2:
            buf_force = self.buffer.competing_attractor(self.state)
        else:
            buf_force = self.buffer.field_force(self.state)
        direction += buf_force * _damping
        direction += MOMENTUM * self.velocity

        candidate  = exp_map(self.state, ENERGY_STEP * direction)
        energy_new = self.coherence_energy(candidate)
        energy_old = self.coherence_energy(self.state)
        margin  = self.adaptive_margin()
        delta_e = energy_new - energy_old

        if delta_e <= margin:
            self._rejection_streak = 0
            self.history.append(self.state.clone())
            dist = poincare_distance(candidate, self.state)
            self.tau     += dist.item()
            self.velocity = log_map(self.state, candidate)
            self.state    = candidate
            _src_hex = source.hex() if isinstance(source, (bytes, bytearray)) else str(source)[:128]
            self.tree.write_concept(self.state, {
                'source':    _src_hex,
                'energy':    energy_new.item(),
                'tau':       self.tau,
                'timestamp': time.time(),
            })
            # Add to semantic RAM index and RGOutput vocabulary
            # Internal sources (buffer reassess, assimilation, dream) must not
            # pollute the vocabulary — 'buffer' as a token dominates generation
            _INTERNAL = {'buffer', 'self_reflection', 'dream',
                         'abstraction', 'atomization', 'assimilation', 'dream_wakeup'}
            _INTERNAL_BYTES = {b'buffer', b'assimilation', b'self_reflection',
                               b'dream_reconcile', b'dream', b'abstraction',
                               b'atomization', b'dream_wakeup'}
            _is_internal = (source in _INTERNAL or
                            (isinstance(source, (bytes, bytearray)) and
                             source in _INTERNAL_BYTES))
            raw = source if isinstance(source, (bytes, bytearray)) else source.encode() if isinstance(source, str) else b""
            if not _is_internal:
                self.rg_output.ingest(raw)
                # LanguageIndex: archive text at accepted state position
                _text = raw.rstrip(b'\x00').decode('utf-8', errors='replace').strip()
                if _text:
                    self.language_index.add(self.state, _text)
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
                if self.rejected_log is not None:
                    self.rejected_log.record(source, candidate, delta_e.item(), self.tau)
                _raw = source if isinstance(source, (bytes, bytearray)) else source.encode() if isinstance(source, str) else b""
                if _raw:
                    self.rg_output.ingest(_raw)
                    _text_r = _raw.rstrip(b'\x00').decode('utf-8', errors='replace').strip()
                    if _text_r:
                        self.language_index.add(candidate, _text_r)
            self.rejected += 1
            print(f"[Buffer] ΔE={delta_e.item():.4f} streak={self._rejection_streak}")
            return False

    def _feedback_step(self, feedback_vec):
        """
        Single internal feedback step — does NOT write to
        or tree, does NOT count as accept/reject. Pure state update.
        Applies radial damping to prevent boundary drift.
        """
        r          = torch.norm(self.state)
        damping    = (1.0 - r ** 2).clamp(min=0.01)

        direction  = log_map(self.state, feedback_vec) * damping
        direction += self.buffer.field_force(self.state) * damping
        direction += MOMENTUM * self.velocity

        candidate  = exp_map(self.state, ENERGY_STEP * direction)
        energy_new = self.coherence_energy(candidate)
        energy_old = self.coherence_energy(self.state)

        if (energy_new - energy_old) <= ENERGY_MARGIN:
            self.history.append(self.state.clone())
            dist          = poincare_distance(candidate, self.state)
            # feedback steps do not increment τ — internal resonance only
            self.velocity = log_map(self.state, candidate)
            self.state    = candidate
            return True, (energy_new - energy_old).item()
        return False, 0.0


    def evolve_sequence(self, text, source="user"):
        """
        Process text as a sequence of narrative threads.

        Each thread is a fragment whose trajectory converges toward a
        local attractor. The cut point is detected geometrically — when
        the thread's energy grows consistently for CONFIRM_WINDOW steps,
        the thread has ended and a new one begins.

        Narrative continuity: each fragment receives the directional
        context of the previous one via approximate parallel transport.
        This preserves the discourse thread across fragment boundaries.

        Architecture:
          bytes -> byte vectors -> FragmentAccumulator -> evolve()
          The byte vectors are NOT averaged flat — they form a trajectory.
          total_length encodes order implicitly: "gato" != "otag".
        """
        raw_bytes = text.encode('utf-8')
        if not raw_bytes:
            return False

        # Level 0: register individual bytes in language tree
        _li = self.language_index
        for _b in raw_bytes:
            _bchar = chr(_b) if 32 <= _b < 127 else f'\\x{_b:02x}'
            _bvec  = bytes_to_vector(bytes([_b]), device=self.state.device)
            _near  = _li.nearest(_bvec, k=1)
            if not _near or _near[0] != _bchar:
                _li.add(_bvec, _bchar)

        first_vec = bytes_to_vector(bytes([raw_bytes[0]]),
                                    device=self.state.device)
        acc          = FragmentAccumulator(first_vec)
        accepted_any = False
        prev_dir     = None   # parallel transport context from previous fragment
        frag_start   = 0

        def _commit_fragment(vec, raw_slice, is_last=False):
            nonlocal accepted_any, prev_dir

            # Apply narrative context from previous fragment
            # Approximate parallel transport: nudge centroid along prev direction
            input_vec = vec
            if prev_dir is not None:
                context_step = prev_dir * 0.15   # small weight — context, not override
                input_vec = to_poincare(
                    exp_map(vec, context_step))

            _INTERNAL_SOURCES = {
                'assimilation', 'buffer', 'dream', 'abstraction',
                'atomization', 'self_reflection', 'dream_reconcile',
                'dream_wakeup', 'dream_fragment'}
            _src_str = (source.decode('utf-8', errors='replace')
                        if isinstance(source, (bytes, bytearray)) else str(source))
            if _src_str not in _INTERNAL_SOURCES:
                _frag_text = raw_slice.decode('utf-8', errors='replace').strip()
                if _frag_text:
                    _li.add(input_vec, _frag_text)
            if self.evolve(input_vec, source=source):
                accepted_any = True
            # rejection handling (buffer + rejected_log) is done inside evolve()

        for i in range(1, len(raw_bytes)):
            byte_vec = bytes_to_vector(bytes([raw_bytes[i]]),
                                       device=self.state.device)
            acc.add_point(byte_vec)

            if acc.should_cut(self):
                prev_dir = acc.parallel_transport_dir()
                _commit_fragment(
                    acc.centroid,
                    raw_bytes[frag_start:i])
                # Start new fragment with current byte
                acc        = FragmentAccumulator(byte_vec)
                frag_start = i

        # Commit last fragment
        if acc.count > 0:
            prev_dir = acc.parallel_transport_dir()
            _commit_fragment(
                acc.centroid,
                raw_bytes[frag_start:],
                is_last=True)

        return accepted_any

    def evolve_with_feedback(self, input_vec, source,
                              conv_threshold=0.001, max_inner=10):
        """
        Regulatory feedback loop.

        After an external input is accepted, the system retrieves
        the nearest memory chunk and resonates with it internally
        until the state stabilises.

        Convergence criterion: distance between consecutive states
        (not energy delta) — more robust against oscillation.

        Feedback steps are internal — they do not write to tree or
, preventing feedback-pollution of the index.
        """
        accepted = self.evolve(input_vec, source=source)
        if not accepted:
            return accepted

        # Phase 1: memory-based feedback (if semantic memory has content)
            prev_state = self.state.clone()
            for iteration in range(max_inner):
                if not results:
                    break
                raw_bytes, dist = results[0]
                feedback_vec = bytes_to_vector(raw_bytes, device=self.state.device)
                self._feedback_step(feedback_vec)
                state_delta = poincare_distance(self.state, prev_state).item()
                prev_state  = self.state.clone()
                if state_delta < conv_threshold:
                    if iteration > 0:
                        print(f"[Feedback] Estável em {iteration+1} iter "
                              f"(Δs={state_delta:.5f})")
                    break

        # Phase 2: RGOutput feedback loop — generate text, re-enter, stabilise
        if self.rg_output.vocab_size() > 0:
            text = self.rg_output.feedback_loop(
                self,
                conv_threshold=conv_threshold,
                max_iter=max_inner,
            )
            if text and text.strip():
                print(f"[RGOutput] {text.strip()}")

        return accepted

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
                _r2       = torch.norm(self.state)
                _damp2    = (1.0 - _r2 ** 2).clamp(min=0.01)
                direction = log_map(self.state, vec) * _damp2 + buffer_force * _damp2 + MOMENTUM * self.velocity
                candidate = exp_map(self.state, ENERGY_STEP * direction)
                candidates.append(candidate)

            cand_tensor  = torch.stack(candidates)
            energies_new = self.coherence_energy_batch(cand_tensor)

            # 🚨 FIX A: recompute energy_old after each acceptance
            for i, (vec, src, candidate) in enumerate(
                    zip(batch_vecs, batch_srcs, candidates)):
                self.total_inputs += 1
                energy_old = self.coherence_energy(self.state)
                margin  = self.adaptive_margin()
                delta_e = energies_new[i] - energy_old

                if delta_e <= margin:
                    self._rejection_streak = 0
                    self.history.append(self.state.clone())
                    dist = poincare_distance(candidate, self.state)
                    self.tau     += dist.item()
                    self.velocity = log_map(self.state, candidate)
                    self.state    = candidate
                    _src_hex = src.hex() if isinstance(src, (bytes, bytearray)) else str(src)[:128]
                    self.tree.write_concept(self.state, {
                        'source':    _src_hex,
                        'energy':    energies_new[i].item(),
                        'tau':       self.tau,
                        'timestamp': time.time(),
                    })
                    # Add to semantic RAM index and RGOutput vocabulary
                    _INTERNAL_B = {'buffer', 'self_reflection', 'dream'}
                    _is_int_b   = (src in _INTERNAL_B or
                                   (isinstance(src, (bytes, bytearray)) and
                                    src in {b'buffer', b'assimilation', b'self_reflection'}))
                    _raw = src if isinstance(src, (bytes, bytearray)) else src.encode() if isinstance(src, str) else b""
                    if not _is_int_b:
                        self.rg_output.ingest(_raw)
                        _text_b = _raw.rstrip(b'\x00').decode('utf-8', errors='replace').strip()
                        if _text_b:
                            self.language_index.add(self.state, _text_b)                        
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
                        if self.rejected_log is not None:
                            self.rejected_log.record(src, candidate, delta_e.item(), self.tau)
                        _raw_r = src if isinstance(src, (bytes, bytearray)) else src.encode() if isinstance(src, str) else b""
                        if _raw_r:
                            self.rg_output.ingest(_raw_r)
                            _text_r2 = _raw_r.rstrip(b'\x00').decode('utf-8', errors='replace').strip()
                            if _text_r2:
                                self.language_index.add(candidate, _text_r2)
                    self.rejected += 1
                    print(f"[Buffer] ΔE={delta_e.item():.4f} streak={self._rejection_streak}")
                    results.append(False)

        return results

    def reassess_buffer(self, radius, assimilate=True):
        before = self.buffer.size()
        self.buffer.reassess(self, radius)
        if assimilate:
            integrated = self.buffer.assimilate(self)
        else:
            integrated = 0
        after = self.buffer.size()
        if before != after:
            extra = f", {integrated} assimilado(s)" if integrated else ""
            print(f"[Buffer] Reavaliado: {before} -> {after} itens{extra}")


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
            self.velocity = log_map(self.state, candidate)
            self.state    = candidate
            return True
        return False

    def abstract_memory(self, merge_radius=0.15, samples=200):
        """RG abstraction — geodesic merge with local Metropolis criterion."""
        merged = 0
        for _ in range(samples):
            vec, meta, dist = self.tree.find_nearest(self.state)
            if vec is None: break
            vec = vec.to(self.state.device)
            noise = torch.randn_like(vec) * 0.01
            perturbed = to_poincare(vec + noise)
            vec2, meta2, _ = self.tree.find_nearest(perturbed)
            if vec2 is None: continue
            vec2 = vec2.to(self.state.device)
            actual_dist = poincare_distance(vec, vec2).item()
            if actual_dist < 1e-4: continue
            if actual_dist < merge_radius:
                centroid = geodesic_midpoint(vec, vec2)
                d_a      = poincare_distance(vec,      self.state).item()
                d_b      = poincare_distance(vec2,     self.state).item()
                d_c      = poincare_distance(centroid, self.state).item()
                delta_e  = d_c - (d_a + d_b)
                if not metropolis_accept(delta_e, self.tau):
                    if hasattr(self, 'rg_analytics'):
                        self.rg_analytics.log_merge(d_a, d_b, d_c, delta_e, self.tau, accepted=False)
                    self.state = to_poincare(self.state + torch.randn_like(self.state) * 0.05)
                    continue
                self.tree.write_concept(centroid, {
                    'source': 'abstraction', 'tau': self.tau,
                    'merged': True, 'dist': float(actual_dist),
                    'delta_e': float(delta_e), 'timestamp': time.time(),
                })
                _label = ""
                _ro = self.rg_output
                if _ro._tokens and _ro._vectors:
                    _n = min(len(_ro._tokens), len(_ro._vectors))
                    if _n > 0:
                        try:
                            _tv = torch.stack(_ro._vectors[:_n]).to(centroid.device)
                            _label = _ro._tokens[int(torch.argmin(poincare_distance_batch(centroid, _tv)).item())]
                        except Exception: pass
                self.attractor_lexicon.rg_merge(vec, vec2, centroid, _label)
                if hasattr(self, 'rg_analytics'):
                    self.rg_analytics.log_merge(d_a, d_b, d_c, delta_e, self.tau, accepted=True, label=_label)
                merged += 1
                self.history.append(self.state.clone())
                _fd = poincare_distance(self.state, centroid).item()
                self._last_fusion_norm = getattr(self, '_last_fusion_norm', 0.0) * 0.9 + _fd * 0.1
                self.velocity = log_map(self.state, centroid)
                self.state    = centroid
            else:
                self.state = to_poincare(self.state + torch.randn_like(self.state) * 0.05)
        if merged > 0: print(f"[Dream/RG] abstracoes formadas: {merged}")
        return merged

    def atomize_memory(self, samples=50, _seen=None):
        """RG downward — fragments go to buffer as dream_fragment."""
        DEDUP_RADIUS = 0.08
        def byte_fragments(raw_bytes):
            n = len(raw_bytes)
            if n < 2: return []
            vecs  = [bytes_to_vector(bytes([b]), device=self.state.device) for b in raw_bytes]
            steps = [poincare_distance(vecs[i], vecs[i+1]).item() for i in range(n-1)]
            if not steps: return []
            mean_d = sum(steps)/len(steps)
            std_d  = (sum((s-mean_d)**2 for s in steps)/len(steps))**0.5
            thresh = mean_d + std_d
            cuts   = [0]
            for i, s in enumerate(steps):
                if s > thresh: cuts.append(i+1)
            cuts.append(n)
            frags = []
            for a, b in zip(sorted(set(cuts)), sorted(set(cuts))[1:]):
                chunk = raw_bytes[a:b]
                if chunk: frags.append(chunk)
            return frags
        li = self.language_index
        if li.size() == 0: return 0
        seen_texts = _seen if _seen is not None else set()
        candidates = []
        for txt in li.nearest(self.state, k=samples):
            if not txt or txt in seen_texts: continue
            seen_texts.add(txt)
            raw = txt.encode('utf-8', errors='replace')
            vec = bytes_to_vector(raw, device=self.state.device)
            t   = poincare_distance(self.state, vec).item()
            candidates.append((t, raw, vec, txt))
        candidates.sort(key=lambda x: -x[0])
        atomized = 0; total_queued = 0; frag_dists = []
        for _tension, raw, parent_vec, parent_text in candidates:
            frags = byte_fragments(raw)
            if not frags: continue
            orig_text    = raw.decode('utf-8', errors='replace')
            frags_scored = []
            for frag_raw in frags:
                frag_text = frag_raw.decode('utf-8', errors='replace')
                if frag_text == orig_text: continue
                fv   = bytes_to_vector(frag_raw, device=self.state.device)
                near = li.nearest(fv, k=1)
                if near:
                    nv = bytes_to_vector(near[0].encode('utf-8', errors='replace'), device=self.state.device)
                    if poincare_distance(fv, nv).item() < DEDUP_RADIUS: continue
                ft = poincare_distance(self.state, fv).item()
                frags_scored.append((len(frag_raw), -ft, frag_raw, fv, frag_text))
            frags_scored.sort(key=lambda x: (x[0], x[1]))
            n_queued = 0
            for frag_len, _, frag_raw, frag_vec, frag_text in frags_scored:
                self.buffer.add(frag_vec,
                    {'source':'dream_fragment','fragment_text':frag_text,
                     'parent_text':parent_text[:60],'frag_len':frag_len,'tau':self.tau},
                    tau=self.tau)
                if hasattr(self, 'rg_analytics'):
                    self.rg_analytics.log_split(orig_text, frag_text, frag_len,
                        d_parent=poincare_distance(self.state, parent_vec).item(),
                        d_frag=poincare_distance(self.state, frag_vec).item(),
                        accepted=True, tau=self.tau)
                frag_dists.append(poincare_distance(self.state, frag_vec).item())
                n_queued += 1; total_queued += 1
            if n_queued > 0: atomized += 1
        self._last_fission_norm = (sum(frag_dists)/len(frag_dists) if frag_dists else 0.0)
        if atomized > 0:
            print(f"[Dream/Atomize] {atomized} conceitos -> {total_queued} fragmentos no buffer")
        return atomized

    def conversational_probe(self, input_vec, steps=3):
        """
        Explore input without integrating it.
        probe_state = geodesic_midpoint(current_state, input_vec),
        evolved lightly for `steps` steps toward input goal.
        Generates from probe state — does NOT alter soul.state or any
        persistent field. Two voices: [CoTa] = consolidated state,
        [Probe] = what the system would say if it were already there.
        """
        probe     = geodesic_midpoint(self.state, input_vec)
        probe_vel = log_map(self.state, probe)

        for _ in range(steps):
            r         = torch.norm(probe)
            damping   = (1.0 - r ** 2).clamp(min=0.01)
            force     = self.buffer.field_force(probe)
            d_input   = log_map(probe, input_vec) * 0.3
            direction = (force + d_input) * damping + MOMENTUM * probe_vel
            new_probe = exp_map(probe, ENERGY_STEP * direction)
            probe_vel = log_map(probe, new_probe)
            probe     = new_probe

        has_vocab = (self.rg_output.vocab_size() > 0 or
                     (hasattr(self, 'attractor_lexicon') and
                      self.attractor_lexicon.active_size() > 0))
        if not has_vocab:
            return ""

        _saved_state, _saved_vel = self.state, self.velocity
        self.state    = probe
        self.velocity = probe_vel
        try:
            narrator = NarrativeGenerator(
                self, input_vec,
                max_tokens  = getattr(self, '_narrative_max_tokens',  20),
                temperature = getattr(self, '_narrative_temperature', 0.9),
            )
            text = narrator.generate()
        finally:
            self.state    = _saved_state
            self.velocity = _saved_vel
        return text

    def dream(self, cycles=500, merge_radius=0.15, rg_every=40,
             atomize_every=None,
             decoder=None, think_every=100, think_steps=3,
             verbose=True, diagnostics=None):
        """
        Full dream cycle.

        cycles       : total relaxation steps
        merge_radius : attractor fusion threshold
        rg_every     : run abstract_memory every N cycles
        decoder      : SemanticDecoder — if provided, runs think() every think_every cycles
        think_every  : run metacognitive loopback every N cycles
        think_steps  : steps per think() call

        Phases per cycle:
          every step         → _dream_step() (relaxation)
          every 10 steps     → buffer.reassess() (reconciliation)
          every rg_every     → abstract_memory() (RG / abstraction formation)
          every think_every  → think() (metacognitive loopback, if decoder set)
          every 200 steps    → checkpoint save
        """
        print(f"\n[Dream] Iniciando {cycles} ciclos "
              f"| τ={self.tau:.4f} | buffer={self.buffer.size()} "
              f"| merge_radius={merge_radius}"
              + (f" | vocab={len(decoder.words)}" if decoder else ""))

        moved    = 0
        fused    = 0
        thoughts = []

        for k in range(1, cycles + 1):

            # Phase 0 — existence tick (semantic gravity acts during dream)
            self.tick(dt=0.1)  # small dt — dream is slower than waking
            if diagnostics is not None:
                diagnostics.record_step(self)

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

            # Phase 3b — atomization (fission / downward RG)
            _atom_every = atomize_every if atomize_every else rg_every * 3
            if k % _atom_every == 0:
                na = self.atomize_memory(
                    samples=min(50, cycles // 10))
                if na > 0:
                    self.tree.flush()
                    _fusion_norm  = getattr(self, "_last_fusion_norm", 0.0)
                    _fission_norm = getattr(self, "_last_fission_norm", 0.001)
                    _ratio = _fusion_norm / (_fission_norm + EPS)
                    _regime = ("critical" if 0.7 < _ratio < 1.4
                               else "collapse" if _ratio >= 1.4
                               else "fragmentation")
                    if verbose:
                        print(f"  [RG] fusion={_fusion_norm:.3f} "
                              f"fission={_fission_norm:.3f} "
                              f"ratio={_ratio:.3f} [{_regime}]")
                    if diagnostics is not None:
                        diagnostics.record_rg_ratio(
                            k, fused, na, _ratio, _regime)

            # Phase 4 — metacognitive loopback
            if decoder is not None and k % think_every == 0:
                r = self.think(decoder, steps=think_steps, write_stable=True)
                thoughts.append(r['word'])
                if verbose:
                    print(f"  [Dream/{k}] Pensamento: '{r['word']}' "
                          f"estabilidade={r['stability']:.0%}")

            # Checkpoint
            if k % 200 == 0:
                self._save()
                print(f"  [Dream/{k}/{cycles}] τ={self.tau:.4f} "
                      f"moved={moved} fused={fused} "
                      + (f"last_thought='{thoughts[-1]}'" if thoughts else ""))

        # Phase 5: reconcile persistent rejected log
        rl_integrated = 0
        if self.rejected_log is not None and self.rejected_log.size() > 0:
            print(f"\n[Dream/Phase5] Reconciliando {self.rejected_log.size()} inputs rejeitados...")
            rl_integrated, _ = self.rejected_log.reconcile(self, max_items=20, verbose=verbose)

        # Phase 6: semantic purge of LanguageIndex
        # Fuse cold, redundant entries not supporting any active concept
        li_fused = 0
        if hasattr(self, "language_index") and self.language_index.size() > 0:
            li_fused, _ = self.language_index.purge_semantic(
                self.tree, verbose=verbose)
        # Phase 7 — wakeup
        wakeup_processed = 0; wakeup_accepted = 0
        _buf_snap = sorted(
            [(k, dict(item)) for k, item in list(self.buffer.items.items())
             if item.get("metadata",{}).get("source") == "dream_fragment"],
            key=lambda x: x[1].get("metadata",{}).get("frag_len", 999))
        if _buf_snap:
            print(f"\n[Dream/Wakeup] {len(_buf_snap)} fragmentos para processar")
            for k, item in _buf_snap:
                frag_text = item.get("metadata",{}).get("fragment_text","")
                if not frag_text: continue
                self.buffer.items.pop(k, None)
                accepted = self.evolve_sequence(frag_text, source="dream_wakeup")
                wakeup_processed += 1
                if accepted:
                    wakeup_accepted += 1
                    try:
                        _mode = "generative" if self.language_index.size() > 0 else "retrieval"
                        produce_output(self, _mode, input_vec=self.state.clone(), probe=False)
                    except Exception as _e:
                        if verbose: print(f"  [Wakeup] output error: {_e}")
            print(f"[Dream/Wakeup] processados={wakeup_processed} aceites={wakeup_accepted} τ={self.tau:.4f}")

        self._save()
        print(f"[Dream] Concluído | moved={moved} | fused={fused} | τ={self.tau:.4f}"
              + (f" | thoughts={thoughts[-5:]}" if thoughts else "")
              + (f" | reconciled={rl_integrated}" if rl_integrated else "")
              + (f" | li_purged={li_fused}" if li_fused else ""))
        result = {'moved': moved, 'fused': fused, 'tau': self.tau,
                  'thoughts': thoughts, 'reconciled': rl_integrated,
                  'li_purged': li_fused,
                  'wakeup_processed': wakeup_processed,
                  'wakeup_accepted':  wakeup_accepted}
        if diagnostics is not None:
            diagnostics.snapshot(self, dream_result=result)
        return result


    # =========================================================================
    # METACOGNITIVE LOOPBACK
    # =========================================================================

    def think(self, decoder, steps=5, write_stable=True):
        """
        Metacognitive loopback: the system speaks to itself and evaluates
        whether its own words are coherent with its current trajectory.

        Unlike evolve(), think() does NOT count toward total_inputs/accepted
        statistics and does NOT write to the tree unless the reflection is
        stable (accepted in >= half the steps). This keeps the tree clean.

        steps       : number of self-reflection iterations
        write_stable: if True, writes the final echo to the tree when stable
        """
        print(f"\n[Reflexão] τ={self.tau:.4f} | buffer={self.buffer.size()}")

        accepted_count = 0
        last_word      = None
        last_echo      = None

        for i in range(steps):
            # 1. Collapse continuous state → nearest discrete symbol
            word, dist = decoder.decode(self.state)

            # 2. Re-encode the symbol (the "echo")
            echo_vec = bytes_to_vector(word)

            # 3. Evaluate coherence WITHOUT side effects
            direction  = echo_vec - self.state
            direction += self.buffer.field_force(self.state)
            direction += MOMENTUM * self.velocity
            candidate  = exp_map(self.state, ENERGY_STEP * direction)

            energy_new = self.coherence_energy(candidate)
            energy_old = self.coherence_energy(self.state)
            margin = self.adaptive_margin()
            delta_e    = (energy_new - energy_old).item()
            accepted   = delta_e <= margin

            if accepted:
                accepted_count += 1
                # Update state with echo — this IS the metacognitive loop
                self.history.append(self.state.clone())
                dist_moved     = poincare_distance(candidate, self.state)
                self.tau      += dist_moved.item()
                self.velocity  = candidate - self.state
                self.state     = candidate

            status    = "✓" if accepted else "✗"
            stability = f"ΔE={delta_e:+.4f}"
            print(f"  ({i+1}/{steps}) '{word}' {status} {stability} "
                  f"dist_vocab={dist:.4f}")

            last_word = word
            last_echo = echo_vec

        # Write to tree only if reflection was stable (majority accepted)
        stability_ratio = accepted_count / steps
        if write_stable and stability_ratio >= 0.5 and last_echo is not None:
            self.tree.write_concept(self.state, {
                'source':    'self_reflection',
                'word':      last_word,
                'stability': stability_ratio,
                'tau':       self.tau,
                'timestamp': time.time(),
            })
            self.tree.flush()
            # Language as geometric operator: stable concept warps local metric
            if decoder is not None:
                word_vec      = bytes_to_vector(last_word.encode("utf-8"), device=self.state.device)
                warp_strength = stability_ratio * 0.08
                warp_dir      = log_map(self.state, word_vec)
                r             = torch.norm(self.state)
                damp          = (1.0 - r ** 2).clamp(min=0.01)
                self.state    = to_poincare(self.state + warp_strength * warp_dir * damp)
                prev          = self.history[-1] if self.history else self.state
                self.velocity = log_map(prev, self.state)
                print(f"  -> [{last_word}] deformou espaco local (forca={warp_strength:.3f})")
            else:
                print(f"  -> Estavel ({stability_ratio:.0%}) — gravado: {last_word}")
        elif stability_ratio < 0.5:
            print(f"  → Instável ({stability_ratio:.0%}) — estado em zona vazia, "
                  f"rejection_streak={self._rejection_streak}")

        return {
            'word':      last_word,
            'accepted':  accepted_count,
            'steps':     steps,
            'stability': stability_ratio,
            'tau':       self.tau,
        }

    def reflect(self, decoder, iterations=3, think_steps=5):
        """
        Full reflection cycle: repeated think() passes.
        Each pass can change state, so the next pass reflects on the updated state.

        This creates a fixed-point dynamic: if the state is semantically stable,
        the same word will appear repeatedly and be reinforced.
        If unstable, the state drifts toward the nearest semantic attractor.
        """
        print(f"\n[Reflect] {iterations} iterações de auto-reflexão")
        results = []
        for k in range(iterations):
            print(f"\n  [Reflect {k+1}/{iterations}]")
            r = self.think(decoder, steps=think_steps)
            results.append(r)
            # If perfectly stable (all accepted), we've reached a fixed point
            if r['stability'] == 1.0:
                print(f"  → Fixed point atingido em iteração {k+1}: '{r['word']}'")
                break
        return results

    def directional_expansion_rate(self, eps=1e-3):
        """
        Estimate local expansion rate along one random direction.
        Computes ||J·v|| / ||v|| for a single noise vector v — this is
        a directional derivative, not the true spectral radius (which
        would require power iteration or SVD). Sufficient as a regime
        heuristic:
          rate < 0.8  → collapse (contracting)
          rate ≈ 1.0  → critical (edge of chaos)
          rate > 1.2  → chaos (expanding)
        Previously misnamed jacobian_spectral_radius.
        """
        if not self.history:
            return 1.0  # assume critical when no history
        s       = self.state
        noise   = torch.randn_like(s) * eps
        s_pert  = to_poincare(s + noise)
        f_s     = s - self.history[-1]
        f_pert  = s_pert - self.history[-1]
        J_col   = (f_pert - f_s) / eps
        return torch.norm(J_col).item()

    def get_stats(self):
        rate = self.directional_expansion_rate()
        regime = ("collapse" if rate < 0.8
                  else "critical" if rate < 1.2
                  else "chaos")
        return {
            'tau': self.tau,
            't_exist': self.t_exist,
            'history_len': len(self.history),
            'state_norm': torch.norm(self.state).item(),
            'total_inputs': self.total_inputs,
            'accepted': self.accepted,
            'rejected': self.rejected,
            'buffer_size': self.buffer.size(),
            'rg_vocab_size': self.rg_output.vocab_size(),
            'expansion_rate':  rate,
            'regime':          regime,
            'susceptibility':  round(self.compute_susceptibility(), 6),
        }



class AttractorLexicon:
    """
    Vocabulary as RG fixed points.
    Each entry: centroid vector + stability score + text label.
    Labels assigned by nearest heard token (SemanticMemory proximity).
    Entries only crystallise after multi-dream confirmation.
    """
    STABILITY_THRESHOLD = 2.0   # min stability to emit in generation
    MERGE_RADIUS        = 0.12  # centroids closer than this are merged
    MAX_ENTRIES         = 2000  # cap on lexicon size

    def __init__(self):
        self._centroids  = []   # list of tensors [DIM]
        self._labels     = []   # list of str
        self._stability  = []   # list of float — grows each confirmation
        self._n_merges   = []   # list of int — how many RG steps formed this

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def register(self, centroid, label, merge_increment=1.0):
        """
        Register or strengthen an attractor centroid.
        If centroid is within MERGE_RADIUS of existing entry: strengthen it.
        Otherwise: create new provisional entry (stability=0).
        """
        if not self._centroids:
            self._add(centroid, label)
            return
        vecs  = torch.stack(self._centroids).to(centroid.device)
        dists = poincare_distance_batch(centroid, vecs)
        idx   = torch.argmin(dists).item()
        if dists[idx].item() < self.MERGE_RADIUS:
            # Strengthen existing — update centroid as geodesic midpoint
            self._centroids[idx] = geodesic_midpoint(
                self._centroids[idx].to(centroid.device), centroid)
            self._stability[idx] += merge_increment
            self._n_merges[idx]  += 1
            # Update label if new one is different (reflects semantic drift)
            if label and label != self._labels[idx]:
                # Keep label that matches geometry better
                pass  # label stable once set — prevents churn
        else:
            if len(self._centroids) >= self.MAX_ENTRIES:
                # Evict lowest-stability entry
                worst = int(torch.tensor(self._stability).argmin().item())
                self._centroids.pop(worst)
                self._labels.pop(worst)
                self._stability.pop(worst)
                self._n_merges.pop(worst)
            self._add(centroid, label)

    def _add(self, centroid, label):
        self._centroids.append(centroid.detach().clone())
        self._labels.append(label)
        self._stability.append(0.0)
        self._n_merges.append(1)

    def rg_merge(self, vec_a, vec_b, centroid, label):
        """
        Called when abstract_memory() fuses vec_a and vec_b into centroid.
        Removes the closest entries to vec_a and vec_b from the lexicon
        (the parents), accumulates their stability into the new centroid.
        This keeps the lexicon aligned with the actual RG tree.
        """
        if not self._centroids:
            self._add(centroid, label)
            return

        vecs = torch.stack(self._centroids).to(centroid.device)

        # Find closest entry to each parent
        dists_a = poincare_distance_batch(vec_a, vecs)
        dists_b = poincare_distance_batch(vec_b, vecs)
        idx_a   = int(torch.argmin(dists_a).item())
        idx_b   = int(torch.argmin(dists_b).item())

        # Only remove if they are genuinely close (within MERGE_RADIUS)
        accumulated_stability = 0.0
        accumulated_merges    = 0
        to_remove = set()
        if dists_a[idx_a].item() < self.MERGE_RADIUS:
            accumulated_stability += self._stability[idx_a]
            accumulated_merges    += self._n_merges[idx_a]
            to_remove.add(idx_a)
        if dists_b[idx_b].item() < self.MERGE_RADIUS and idx_b != idx_a:
            accumulated_stability += self._stability[idx_b]
            accumulated_merges    += self._n_merges[idx_b]
            to_remove.add(idx_b)

        # Remove parents in reverse index order to preserve indices
        for i in sorted(to_remove, reverse=True):
            self._centroids.pop(i)
            self._labels.pop(i)
            self._stability.pop(i)
            self._n_merges.pop(i)

        # Register centroid with inherited stability
        if len(self._centroids) >= self.MAX_ENTRIES:
            worst = int(torch.tensor(self._stability).argmin().item())
            self._centroids.pop(worst)
            self._labels.pop(worst)
            self._stability.pop(worst)
            self._n_merges.pop(worst)

        self._centroids.append(centroid.detach().clone())
        self._labels.append(label)
        self._stability.append(accumulated_stability + 1.0)
        self._n_merges.append(accumulated_merges + 1)

    def active_entries(self):
        """Return (centroids_tensor, labels) for entries above threshold."""
        if not self._centroids:
            return None, []
        stable_idx = [i for i, s in enumerate(self._stability)
                      if s >= self.STABILITY_THRESHOLD]
        if not stable_idx:
            return None, []
        vecs   = torch.stack([self._centroids[i] for i in stable_idx])
        labels = [self._labels[i] for i in stable_idx]
        return vecs, labels

    def all_entries(self):
        """Return all entries including provisional."""
        if not self._centroids:
            return None, []
        return torch.stack(self._centroids), list(self._labels)

    def size(self):
        return len(self._centroids)

    def active_size(self):
        return sum(1 for s in self._stability if s >= self.STABILITY_THRESHOLD)

    # ------------------------------------------------------------------
    # Persistence (.npz)
    # ------------------------------------------------------------------

    def save(self, path):
        stem  = str(path)[:-4] if str(path).endswith(".npz") else str(path)
        final = stem + ".npz"
        tmp   = stem + ".tmp"
        if not self._centroids:
            vecs_np = np.zeros((0, DIM), dtype=np.float32)
            labels  = np.array([], dtype=object)
            stab    = np.array([], dtype=np.float32)
            nmerge  = np.array([], dtype=np.int32)
        else:
            vecs_np = np.stack([c.cpu().numpy() for c in self._centroids]).astype(np.float32)
            labels  = np.array(self._labels, dtype=object)
            stab    = np.array(self._stability, dtype=np.float32)
            nmerge  = np.array(self._n_merges, dtype=np.int32)
        np.savez_compressed(tmp, vecs=vecs_np, labels=labels,
                            stability=stab, n_merges=nmerge)
        os.replace(tmp + ".npz", final)

    @classmethod
    def load(cls, path, device):
        obj     = cls()
        npz_path = str(path)[:-4] if str(path).endswith(".npz") else str(path)
        npz_path += ".npz"
        if not os.path.exists(npz_path):
            return obj
        try:
            data   = np.load(npz_path, allow_pickle=True)
            vecs   = data["vecs"]
            labels = data["labels"]
            stab   = data["stability"]
            nmerge = data["n_merges"]
            for v, l, s, n in zip(vecs, labels, stab, nmerge):
                obj._centroids.append(
                    torch.tensor(v, dtype=torch.float32, device=device))
                obj._labels.append(str(l))
                obj._stability.append(float(s))
                obj._n_merges.append(int(n))
        except Exception as e:
            print(f"[AttractorLexicon] erro ao carregar: {e}")
        return obj


class LanguageIndex:
    """
    Persistent linguistic surface — HyperbolicTree format.

    Architecture:
      Buffer <- Input -> LanguageIndex -> ConceptTree

    The LI is the log of all observed language. It does NOT gatekeep —
    everything heard enters (accepted or rejected). The ConceptTree is
    the semantic filter; the LI is the linguistic witness.

    Key differences from the old .npz implementation:
      - Binary tree storage (language_index.bin) — same format as tree.bin
      - Beam search k-nearest (find_k_nearest) instead of full scan
      - Geometric deduplication on add() — near-duplicates strengthen
        existing entry (access_count++) rather than creating new node
      - No eviction — grows as log. Purge is explicit (purge_semantic())
      - Lang field in metadata (ISO 639-1) — future multilingual routing
      - Pressure field: entries queried often have high pressure and
        survive purge; cold entries are candidates for fusion

    Purge policy (called at end of dream):
      Entries not accessed in N dream cycles AND far from any active
      concept in the ConceptTree are fused with their nearest LI neighbour
      (geodesic midpoint). Text is concatenated as evidence trail.
      This keeps the LI bounded without losing linguistic coverage.

    Migration: if language_index.npz exists but .bin does not,
    migrate_from_npz() is called automatically on first load.
    """

    DEDUP_RADIUS  = 0.06   # strengthen existing if closer than this
    PURGE_RADIUS  = 0.04   # fuse cold entries closer than this during purge
    COLD_THRESHOLD = 3     # access_count below this = cold candidate

    def __init__(self, path="language_index.bin", tree_size_mb=20):
        self.path = path
        self.tree = HyperbolicTree(path, initial_size_mb=tree_size_mb)
        self._size = 0
        self._scan_size()

    def _meta_path(self):
        return self.path + '.meta'

    def _load_size(self):
        try:
            with open(self._meta_path(), 'r') as f:
                self._size = int(f.read().strip())
        except Exception:
            self._size = 0

    def _save_size(self):
        try:
            with open(self._meta_path(), 'w') as f:
                f.write(str(self._size))
        except Exception:
            pass

    def _scan_size(self):
        """Load size from .meta; full scan only on first run."""
        if os.path.exists(self._meta_path()):
            self._load_size(); return
        if not os.path.exists(self.path):
            self._size = 0; return
        file_size = os.path.getsize(self.path)
        node_size = self.tree.node_size
        count = 0
        self.tree._file.seek(0)
        for i in range(file_size // node_size):
            self.tree._file.seek(i * node_size)
            data = self.tree._file.read(node_size)
            if len(data) < node_size: break
            has_data, _, _ = self.tree._unpack_node(data)
            if has_data: count += 1
        self._size = count
        self._save_size()

    def add(self, vec, text, lang=None):
        """
        Add (vec, text) to the linguistic surface.
        If a near-duplicate exists (distance < DEDUP_RADIUS), strengthen
        it (increment access_count, append text evidence) instead of
        creating a new node.
        """
        if not text or not text.strip():
            return
        text = text.strip()

        # Geometric dedup
        _, nearest_meta, nearest_dist = self.tree.find_nearest(vec, max_depth=5)
        if nearest_dist < self.DEDUP_RADIUS and nearest_meta.get('text'):
            # Strengthen existing — re-write with incremented access
            nearest_meta['access_count'] = nearest_meta.get('access_count', 0) + 1
            nearest_meta['last_text']    = text[:120]
            self.tree.write_concept(vec, nearest_meta)
            return

        meta = {
            'text':         text[:200],
            'lang':         lang or '',
            'access_count': 0,
            'tau_added':    0.0,   # filled by soul if available
            'timestamp':    time.time(),
        }
        self.tree.write_concept(vec, meta)
        self._size += 1
        self._save_size()

    def add_with_tau(self, vec, text, tau, lang=None):
        """Variant that records tau_added — used by evolve_batch."""
        if not text or not text.strip():
            return
        text = text.strip()

        _, nearest_meta, nearest_dist = self.tree.find_nearest(vec, max_depth=5)
        if nearest_dist < self.DEDUP_RADIUS and nearest_meta.get('text'):
            nearest_meta['access_count'] = nearest_meta.get('access_count', 0) + 1
            nearest_meta['last_text']    = text[:120]
            self.tree.write_concept(vec, nearest_meta)
            return

        self.tree.write_concept(vec, {
            'text':         text[:200],
            'lang':         lang or '',
            'access_count': 0,
            'tau_added':    float(tau),
            'timestamp':    time.time(),
        })
        self._size += 1
        self._save_size()

    def nearest(self, query_vec, k=3, lang=None):
        """
        Return up to k nearest texts by hyperbolic distance.
        Uses beam search (find_k_nearest) — O(k·log n) vs old O(n) scan.
        Increments access_count for retrieved entries.
        lang filter: if provided, skips entries with non-matching lang field
        ('' and None always pass — unknown lang is not filtered out).
        """
        results = self.tree.find_k_nearest(query_vec, k=k * 2, max_depth=8)
        texts   = []
        for dist, vec, meta in results:
            if not meta.get('text'):
                continue
            entry_lang = meta.get('lang', '')
            if lang and entry_lang and entry_lang != lang:
                continue
            # Increment access_count in-place via write_queue
            meta['access_count'] = meta.get('access_count', 0) + 1
            self.tree.write_concept(vec.to(query_vec.device), meta)
            texts.append(meta['text'])
            if len(texts) >= k:
                break
        return texts

    def purge_semantic(self, soul_tree, min_access=None, verbose=True):
        """
        Fuse cold, redundant LI entries during dream.

        Cold entry: access_count < COLD_THRESHOLD (or min_access if given)
        AND hyperbolic distance to nearest active ConceptTree node > 0.3
        (i.e. not supporting any active concept).

        Cold entries closer than PURGE_RADIUS to each other are fused:
          - new vec = geodesic_midpoint(a, b)
          - text = a.text + ' | ' + b.text (evidence trail)
          - access_count = max(a, b)

        This keeps LI bounded without losing linguistic coverage.
        Returns (fused_count, removed_count).
        """
        threshold = min_access if min_access is not None else self.COLD_THRESHOLD
        file_size = os.path.getsize(self.path)
        node_size = self.tree.node_size

        # Collect cold entries
        cold = []
        self.tree._file.seek(0)
        for idx in range(file_size // node_size):
            offset = idx * node_size
            self.tree._file.seek(offset)
            data = self.tree._file.read(node_size)
            if len(data) < node_size:
                break
            has_data, vec, meta = self.tree._unpack_node(data)
            if not has_data or not meta.get('text'):
                continue
            if meta.get('access_count', 0) < threshold:
                # Check distance to nearest concept in soul_tree
                _, _, dist_to_concept = soul_tree.find_nearest(
                    vec, max_depth=5)
                if dist_to_concept > 0.3:
                    cold.append((idx, vec, meta))

        fused = 0
        cleared = set()
        for i in range(len(cold)):
            if i in cleared:
                continue
            idx_a, vec_a, meta_a = cold[i]
            for j in range(i + 1, len(cold)):
                if j in cleared:
                    continue
                idx_b, vec_b, meta_b = cold[j]
                if poincare_distance(vec_a, vec_b).item() < self.PURGE_RADIUS:
                    # Fuse: geodesic midpoint, concatenate text evidence
                    mid  = geodesic_midpoint(vec_a, vec_b)
                    text = (meta_a['text'][:100] + ' | ' +
                            meta_b['text'][:100])[:200]
                    new_meta = {
                        'text':         text,
                        'lang':         meta_a.get('lang') or meta_b.get('lang', ''),
                        'access_count': max(meta_a.get('access_count', 0),
                                            meta_b.get('access_count', 0)),
                        'tau_added':    meta_a.get('tau_added', 0.0),
                        'timestamp':    time.time(),
                        'fused':        True,
                    }
                    # Zero out both, write fused node
                    self.tree._write_queue[idx_a * node_size] = (
                        b'\x00' * node_size)
                    self.tree._write_queue[idx_b * node_size] = (
                        b'\x00' * node_size)
                    self.tree.write_concept(mid, new_meta)
                    cleared.add(i)
                    cleared.add(j)
                    fused += 1
                    self._size = max(0, self._size - 1)
                    break

        self.tree.flush()
        self._save_size()
        if verbose and fused > 0:
            print(f"[LanguageIndex] purge: {fused} pares fundidos, "
                  f"{len(cleared)} entradas consolidadas")
        return fused, len(cleared)

    def size(self):
        return self._size

    def flush(self):
        self.tree.flush()
        self._save_size()

    def close(self):
        self.tree.flush()
        self._save_size()
        self.tree.close()

    @classmethod
    def load(cls, path, device=None, tree_size_mb=20):
        """
        Load or create LanguageIndex from path.
        If path.npz exists but path.bin does not, migrate automatically.
        device is accepted for API compatibility but ignored (tree is device-agnostic).
        """
        # Resolve .bin path
        if path.endswith('.npz'):
            bin_path = path[:-4] + '.bin'
            npz_path = path
        elif path.endswith('.bin'):
            bin_path = path
            npz_path = path[:-4] + '.npz'
        else:
            bin_path = path + '.bin'
            npz_path = path + '.npz'

        obj = cls(bin_path, tree_size_mb=tree_size_mb)

        # Auto-migrate from .npz if .bin is empty and .npz exists
        if (os.path.exists(npz_path)
                and os.path.getsize(bin_path) < obj.tree.node_size * 10):
            obj.migrate_from_npz(npz_path, device=device)

        return obj

    def migrate_from_npz(self, npz_path, device=None):
        """
        One-shot migration from old .npz format to HyperbolicTree .bin.
        Called automatically by load() when .npz exists but .bin is empty.
        """
        try:
            data   = __import__('numpy').load(npz_path, allow_pickle=True)
            vecs   = data['vecs']
            texts  = data['texts']
            langs  = (data['langs'] if 'langs' in data
                      else [None] * len(texts))
            access = (data['access'] if 'access' in data
                      else [0] * len(texts))
            n = len(vecs)
            print(f"[LanguageIndex] migrando {n} entradas de {npz_path} ...")
            dev = device or 'cpu'
            for i, (v, t, a, l) in enumerate(zip(vecs, texts, access, langs)):
                vec = __import__('torch').tensor(
                    v, dtype=__import__('torch').float32).to(dev)
                self.tree.write_concept(vec, {
                    'text':         str(t)[:200],
                    'lang':         str(l) if l and str(l) else '',
                    'access_count': int(a),
                    'tau_added':    0.0,
                    'timestamp':    time.time(),
                })
                self._size += 1
                if i % 10000 == 0 and i > 0:
                    self.tree.flush()
                    print(f"  ... {i}/{n}")
            self.tree.flush()
            self._save_size()
            print(f"[LanguageIndex] migração concluída: {self._size} entradas")
        except Exception as e:
            print(f"[LanguageIndex] erro na migração: {e}")

    # ── save() kept for API compatibility — just flushes the tree ──────────
    def save(self, path=None):
        self.tree.flush()

class NarrativeGenerator:
    """
    Language generation as energy minimisation in hyperbolic space.
    Each step selects the candidate text from LanguageIndex that most
    reduces poincare_distance(simulated_state, goal).
    Stops when average energy gain < ENERGY_THRESHOLD (saturated).
    """
    ENERGY_THRESHOLD  = 0.005
    TRANSITION_ALPHA  = 0.35
    K_CANDIDATES      = 10
    MAX_STEPS         = 40
    MIN_STEPS         = 2
    SATURATION_WINDOW = 3

    def __init__(self, soul, input_vec, max_tokens=40, temperature=0.7,
                 min_tokens=2, coherence_patience=3):
        self.soul               = soul
        self.input_vec          = input_vec
        self.max_tokens         = min(max_tokens, self.MAX_STEPS)
        self.temperature        = temperature
        self.min_tokens         = min_tokens
        self.coherence_patience = coherence_patience

    def _simulate(self, state, candidate_vec):
        direction = log_map(state, candidate_vec)
        return exp_map(state, direction * self.TRANSITION_ALPHA)

    def _energy(self, state):
        return poincare_distance(state, self.input_vec).item()

    def _effective_temperature(self):
        rho        = getattr(self.soul, '_cached_expansion', 1.0)
        tau_norm   = min(self.soul.tau / 1000.0, 1.0)
        collapse_b = max(0.0, (0.8 - rho)) * 1.5 + (1.0 - tau_norm) * 0.3
        chaos_p    = max(0.0, (rho - 1.2)) * 1.0
        T = self.temperature * (1.0 + collapse_b) / (1.0 + chaos_p)
        return max(0.05, T)

    def _select_candidate(self, state, candidates, T, exclude):
        E_now = self._energy(state)
        gains = []
        for text, vec in candidates:
            if text in exclude:
                gains.append(float('-inf'))
                continue
            new_state = self._simulate(state, vec)
            gains.append(E_now - self._energy(new_state))
        if all(g == float('-inf') for g in gains):
            gains = []
            for text, vec in candidates:
                gains.append(E_now - self._energy(self._simulate(state, vec)))
        gains_t = torch.tensor(gains, dtype=torch.float32)
        if T < 1e-6:
            idx = int(torch.argmax(gains_t).item())
        else:
            probs = torch.softmax(gains_t / T, dim=0)
            idx   = int(torch.multinomial(probs, 1).item())
        text, vec = candidates[idx]
        new_state = self._simulate(state, vec)
        return text, vec, new_state, E_now - self._energy(new_state)

    def _get_candidates(self, state):
        li = getattr(self.soul, 'language_index', None)
        if li is not None and li.size() > 0:
            texts = li.nearest(state, k=self.K_CANDIDATES)
            if texts:
                seen = set()
                cands = []
                for txt in texts:
                    if txt in seen: continue
                    seen.add(txt)
                    vec = bytes_to_vector(txt.encode('utf-8', errors='replace'),
                                         device=state.device)
                    cands.append((txt, vec))
                if cands:
                    return cands
        lex = getattr(self.soul, 'attractor_lexicon', None)
        if lex is not None:
            vecs, labels = lex.active_entries()
            if vecs is not None and len(labels) > 0:
                vecs = vecs.to(state.device)
                return [(lbl, vecs[i]) for i, lbl in enumerate(labels)]
        ro = self.soul.rg_output
        n  = min(len(ro._tokens), len(ro._vectors))
        if n > 0:
            try:
                vecs = torch.stack(ro._vectors[:n]).to(state.device)
                return [(ro._tokens[i], vecs[i]) for i in range(n)]
            except Exception:
                pass
        return []

    def generate(self):
        state        = self.soul.state.clone()
        T            = self._effective_temperature()
        output       = []
        exclude      = set()
        gains_window = []
        for step in range(self.max_tokens):
            candidates = self._get_candidates(state)
            if not candidates:
                break
            text, vec, new_state, gain = self._select_candidate(
                state, candidates, T, exclude)
            output.append(text)
            exclude.add(text)
            state = new_state
            gains_window.append(gain)
            if len(gains_window) > self.SATURATION_WINDOW:
                gains_window.pop(0)
            if step >= self.MIN_STEPS:
                if sum(gains_window) / len(gains_window) < self.ENERGY_THRESHOLD:
                    break
        return " ".join(output)


# =============================================================================
# DIAGNOSTICS
# =============================================================================

class DiagnosticsLogger:
    """
    Captures system snapshots at end of each dream cycle.
    Writes simultaneously to:
      - .jsonl  (one JSON object per line — machine-readable, appendable)
      - .csv    (one row per snapshot — easy to open in pandas/Excel)

    Metrics captured:
      tau              — geometric displacement (cognitive age)
      t_exist          — wall-clock existence time
      tau_rate         — accepts / total_inputs (learning rate)
      state_norm       — ||state|| (position in disk; near 1 = near boundary)
      fisher           — fisher_norm(state) (local curvature weight)
      state_drift      — hyperbolic distance moved since last snapshot
      energy_mean      — mean coherence energy over last window
      expansion_rate   — directional expansion (collapse/critical/chaos)
      regime           — string label
      buffer_size      — pending items
      pressure_mean    — mean pressure in buffer (tension with exterior)
      pressure_max     — max single-item pressure
      gravity_norm     — ||semantic gravity vector|| (pull from buffer)
      vocab_size        — RGOutput token count
      vocab_entropy    — -Σ p_i log p_i over token distances (diversity)
      dream_moved      — steps where state moved during dream
      dream_fused      — RG fusions during dream
      dream_thoughts   — words surfaced during think() calls
      timestamp        — Unix time
      session          — dream cycle index
    """

    CSV_FIELDS = [
        'session', 'timestamp', 'tau', 't_exist', 'tau_rate',
        'state_norm', 'fisher', 'state_drift',
        'expansion_rate', 'regime',
        'buffer_size', 'pressure_mean', 'pressure_max', 'gravity_norm',
        'vocab_size', 'vocab_entropy',
        'dream_moved', 'dream_fused', 'dream_reconciled',
        'fisher_vel_mean', 'fisher_vel_std', 'fv_regime',
        'plasticity_mean', 'plasticity_std',
        'susceptibility_mean', 'susceptibility_max',
    ]

    def __init__(self, base_path):
        """
        base_path: stem without extension, e.g. 'diagnostics'
        Creates diagnostics.jsonl and diagnostics.csv
        """
        self.jsonl_path = base_path + '.jsonl'
        self.csv_path   = base_path + '.csv'
        self._session       = 0
        self._last_state    = None
        self._last_fisher   = None   # for fisher_velocity
        self._energy_window = []     # rolling window of coherence energy values
        # Per-step metrics accumulated during dream — reset each snapshot
        self.step_metrics   = {      # public: dream loop writes here
            "fisher":         [],
            "fisher_velocity": [],
            "plasticity":     [],
            "expansion_rate": [],
            "acceptance_ratio": [],
            "susceptibility":   [],
        }
        self._write_csv_header()

    def record_step(self, soul):
        """
        Called every dream step. Accumulates per-step observables.
        fisher_velocity = F_t - F_{t-1}  (rate of change of curvature)
        plasticity_index = acceptance_rate / (1 + fisher)  (cognitive dashboard)
          high rate + low fisher  → learning
          low rate  + high fisher → narrative rigidity
          low both               → confusion
          high both              → deep reorganisation
        """
        f      = fisher_norm(soul.state).item()
        rate   = soul.accepted / max(1, soul.total_inputs)
        p_idx  = rate / (1.0 + f)
        fv     = (f - self._last_fisher) if self._last_fisher is not None else 0.0
        exp    = getattr(soul, "_cached_expansion", 1.0)
        self._last_fisher = f
        self.step_metrics["fisher"].append(f)
        self.step_metrics["fisher_velocity"].append(fv)
        self.step_metrics["plasticity"].append(p_idx)
        self.step_metrics["expansion_rate"].append(exp)
        chi = soul.compute_susceptibility()
        self.step_metrics["acceptance_ratio"].append(rate)
        self.step_metrics["susceptibility"].append(chi)

    def _write_csv_header(self):
        """Write CSV header if file is new."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', encoding='utf-8') as f:
                f.write(','.join(self.CSV_FIELDS) + '\n')

    def _vocab_entropy(self, soul):
        """
        Shannon entropy of token distribution by proximity to current state.
        H = -Σ p_i log p_i  where p_i = softmax(-dist_i)
        Low entropy = state near one token (focused)
        High entropy = state equidistant from many (diffuse)
        """
        ro = soul.rg_output
        if not ro._tokens or len(ro._tokens) < 2:
            return 0.0
        vecs  = torch.stack(ro._vectors).to(soul.state.device)
        dists = poincare_distance_batch(soul.state, vecs)
        probs = torch.softmax(-dists, dim=0)
        # Clamp to avoid log(0)
        probs = probs.clamp(min=1e-9)
        entropy = -(probs * torch.log(probs)).sum().item()
        return entropy

    def _gravity_norm(self, soul):
        """Magnitude of semantic gravity vector from buffer."""
        if soul.buffer.size() == 0:
            return 0.0
        items  = list(soul.buffer.items.values())
        active = [it for it in items if it.get('pressure', 0.0) >= 0.1]
        if not active:
            return 0.0
        vecs  = torch.stack([it['vector'] for it in active])
        pres  = torch.tensor([it['pressure'] for it in active],
                              device=soul.state.device)
        w     = pres / (pres.sum() + EPS)
        dirs  = torch.stack([log_map(soul.state, v) for v in vecs])
        grav  = (w.unsqueeze(1) * dirs).sum(dim=0)
        return torch.norm(grav).item()

    def _pressure_stats(self, soul):
        if soul.buffer.size() == 0:
            return 0.0, 0.0
        pressures = [it.get('pressure', 0.0) for it in soul.buffer.items.values()]
        return float(sum(pressures) / len(pressures)), float(max(pressures))

    def record_rg_ratio(self, step, n_fused, n_atomized, ratio, regime):
        """Record fusion/fission balance — written to .jsonl."""
        try:
            row = {"type": "rg_ratio", "step": step,
                   "n_fused": n_fused, "n_atomized": n_atomized,
                   "ratio": round(ratio, 4), "regime": regime,
                   "timestamp": time.time()}
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def snapshot(self, soul, dream_result=None):
        """
        Capture a full diagnostic snapshot.
        dream_result: dict returned by soul.dream() — optional.
        """
        self._session += 1
        stats        = soul.get_stats()
        fisher       = fisher_norm(soul.state).item()
        vocab_ent    = self._vocab_entropy(soul)
        grav         = self._gravity_norm(soul)
        p_mean, p_max = self._pressure_stats(soul)

        # State drift since last snapshot
        if self._last_state is not None:
            drift = poincare_distance(soul.state, self._last_state).item()
        else:
            drift = 0.0
        self._last_state = soul.state.clone()

        tau_rate = (stats['accepted'] / max(1, stats['total_inputs']))

        # Aggregate per-step metrics
        def _mean(lst): return round(sum(lst)/len(lst), 4) if lst else 0.0
        def _std(lst):
            if len(lst) < 2: return 0.0
            m = sum(lst)/len(lst)
            return round((sum((x-m)**2 for x in lst)/len(lst))**0.5, 4)

        fisher_vel_mean  = _mean(self.step_metrics["fisher_velocity"])
        fisher_vel_std   = _std(self.step_metrics["fisher_velocity"])
        plasticity_mean     = _mean(self.step_metrics["plasticity"])
        plasticity_std      = _std(self.step_metrics["plasticity"])
        susceptibility_mean = _mean(self.step_metrics.get("susceptibility", []))
        susceptibility_max  = max(self.step_metrics.get("susceptibility", [0.0]), default=0.0)

        # Classify fisher_velocity regime
        fv = fisher_vel_mean
        if   abs(fv) < 0.001:              fv_regime = "stable"
        elif fv > 0.005:                   fv_regime = "crystallising"
        elif fv < -0.005:                  fv_regime = "reorganising"
        elif fisher_vel_std > 0.01:        fv_regime = "transitioning"
        else:                              fv_regime = "drifting"

        # Reset for next cycle
        for k in self.step_metrics:
            self.step_metrics[k].clear()

        record = {
            'session':        self._session,
            'timestamp':      time.time(),
            'tau':            round(stats['tau'], 4),
            't_exist':        round(stats['t_exist'], 2),
            'tau_rate':       round(tau_rate, 4),
            'state_norm':     round(stats['state_norm'], 4),
            'fisher':         round(fisher, 4),
            'state_drift':    round(drift, 4),
            'expansion_rate': round(stats['expansion_rate'], 4),
            'regime':         stats['regime'],
            'buffer_size':    stats['buffer_size'],
            'pressure_mean':  round(p_mean, 4),
            'pressure_max':   round(p_max, 4),
            'gravity_norm':   round(grav, 4),
            'vocab_size':     stats['rg_vocab_size'],
            'lexicon_size':   getattr(soul, 'attractor_lexicon', None) and soul.attractor_lexicon.active_size() or 0,
            'vocab_entropy':  round(vocab_ent, 4),
            'dream_moved':      dream_result.get('moved', 0) if dream_result else 0,
            'dream_fused':      dream_result.get('fused', 0) if dream_result else 0,
            'dream_thoughts':     dream_result.get('thoughts', []) if dream_result else [],
            'dream_reconciled':   dream_result.get('reconciled', 0) if dream_result else 0,
            'fisher_vel_mean':  fisher_vel_mean,
            'fisher_vel_std':   fisher_vel_std,
            'fv_regime':        fv_regime,
            'plasticity_mean':  plasticity_mean,
            'plasticity_std':      plasticity_std,
            'susceptibility_mean': round(susceptibility_mean, 6),
            'susceptibility_max':  round(susceptibility_max, 6),
        }

        self._append_jsonl(record)
        self._append_csv(record)
        self._print_summary(record)
        return record

    def _append_jsonl(self, record):
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _append_csv(self, record):
        row = [str(record.get(f, '')) for f in self.CSV_FIELDS]
        with open(self.csv_path, 'a', encoding='utf-8') as f:
            f.write(','.join(row) + '\n')

    def _print_summary(self, r):
        print(
            f"[Diag #{r['session']}] "
            f"τ={r['tau']:.2f} rate={r['tau_rate']:.2f} "
            f"drift={r['state_drift']:.3f} fisher={r['fisher']:.3f} "
            f"regime={r['regime']} "
            f"buf={r['buffer_size']} p̄={r['pressure_mean']:.2f} "
            f"grav={r['gravity_norm']:.3f} "
            f"vocab={r['vocab_size']} H={r['vocab_entropy']:.3f} "
            f"moved={r['dream_moved']} fused={r['dream_fused']} | "
            f"dF/dt={r['fisher_vel_mean']:+.4f}±{r['fisher_vel_std']:.4f} "
            f"[{r['fv_regime']}] "
            f"plasticity={r['plasticity_mean']:.4f}±{r['plasticity_std']:.4f} "
            f"chi={r['susceptibility_mean']:.5f}(max={r['susceptibility_max']:.5f})"
        )

# =============================================================================
# OUTPUT
# =============================================================================

def decode_source(source_str):
    """
    Decode a source field from metadata back to human-readable text.
    If it looks like hex (binary chunk), decode bytes to UTF-8.
    Otherwise return as-is (text source).
    """
    if not source_str:
        return ""
    # Hex-encoded binary chunk: even length, all hex chars
    s = source_str.strip()
    if len(s) % 2 == 0 and all(c in "0123456789abcdefABCDEF" for c in s):
        try:
            raw = bytes.fromhex(s)
            # Strip null padding, decode as UTF-8 with replacement
            text = raw.rstrip(b'\x00').decode('utf-8', errors='replace')
            return text if text.strip() else repr(raw[:16])
        except Exception:
            pass
    return s

def nearest_concept(soul):
    """
    Retrieval via LanguageIndex — nearest text to current soul state.
    Skips internal source labels.
    """
    _INTERNAL = {
        'assimilation', 'buffer', 'dream', 'abstraction',
        'atomization', 'self_reflection', 'dream_reconcile',
        'dream_wakeup', 'dream_fragment',
    }
    li = soul.language_index
    li._scan_size()  # ensure size is accurate
    _li_sz = li.size()
    _raw   = li.tree.find_k_nearest(soul.state, k=20, max_depth=12)
    print(f"[DBG nearest_concept] li.size={_li_sz} raw_results={len(_raw)}")
    for dist, vec, meta in _raw:
        print(f"  [DBG] dist={dist:.4f} meta={list(meta.items())[:3]}")
    if _li_sz > 0:
        for dist, vec, meta in _raw:
            txt = (meta.get('text', '') or meta.get('source', '')).strip()
            if txt and txt.lower() not in _INTERNAL:
                return txt, float(dist)
    # Fallback: tree
    vec, meta, dist = soul.tree.find_nearest(soul.state)
    if vec is not None:
        text = decode_source(meta.get('source', ''))
        return text, dist
    return None, float('inf')

class SessionLogger:
    """Append-only log — soul.log next to soul.json. One JSON line per event."""
    def __init__(self, soul_path):
        base = soul_path[:-5] if soul_path.endswith('.json') else soul_path
        self.path = base + '.log'
        self._f = open(self.path, 'a', encoding='utf-8', buffering=1)
        self._write({'event': 'session_start', 'ts': time.time()})
    def log_input(self, text, accepted, tau):
        self._write({'event':'input','text':text[:500],'accepted':accepted,'tau':round(tau,4),'ts':time.time()})
    def log_output(self, kind, text, tau):
        self._write({'event':'output','kind':kind,'text':text[:500],'tau':round(tau,4),'ts':time.time()})
    def log_dream(self, result):
        self._write({'event':'dream','moved':result.get('moved'),'fused':result.get('fused'),
                     'reconciled':result.get('reconciled'),'li_purged':result.get('li_purged'),
                     'tau':round(result.get('tau',0),4),'ts':time.time()})
    def _write(self, obj):
        try: self._f.write(json.dumps(obj, ensure_ascii=False)+'\n')
        except Exception: pass
    def close(self):
        self._write({'event':'session_end','ts':time.time()})
        self._f.close()


def produce_output(soul, mode, input_vec=None, probe=False, logger=None):
    if mode == 'none':
        return
    if mode in ('retrieval', 'both'):
        text, dist = nearest_concept(soul)
        if text is not None:
            print(f'[Retrieval] dist={dist:.4f} -> "{text}"')
            if logger: logger.log_output('retrieval', text, soul.tau)
        else:
            print("[Retrieval] Memória vazia.")
    if mode in ('generative', 'both'):
        has_vocab = (soul.rg_output.vocab_size() > 0 or
                     (hasattr(soul, 'attractor_lexicon') and
                      soul.attractor_lexicon.active_size() > 0))
        if not has_vocab:
            print("[CoTa] vocabulario vazio -- ingere texto primeiro")
            return
        goal     = input_vec if input_vec is not None else soul.state
        narrator = NarrativeGenerator(
            soul, goal,
            max_tokens  = getattr(soul, '_narrative_max_tokens',  20),
            temperature = getattr(soul, '_narrative_temperature', 0.7),
        )
        text = narrator.generate()
        if text and text.strip():
            print(f"[CoTa] {text}")
            if logger: logger.log_output('generative', text, soul.tau)
        else:
            print(f"[CoTa] (sem output -- vocab={soul.rg_output.vocab_size()} tokens)")
        # Probe: generate from simulated midpoint without integrating
        if probe and input_vec is not None and has_vocab:
            probe_text = soul.conversational_probe(input_vec)
            if probe_text and probe_text.strip() and probe_text != text:
                print(f"[Probe] {probe_text}")
                if logger: logger.log_output('probe', probe_text, soul.tau)

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
    parser.add_argument("--rejected-file", default="rejected.bin",
                        help="Tree de inputs rejeitados (default: rejected.bin)")
    parser.add_argument("--rg-vocab-file", default="rg_vocab.txt",
                        help="Ficheiro de vocabulário RGOutput (default: rg_vocab.txt)")
    parser.add_argument("--lexicon-file", default="attractor_lexicon",
                        help="Ficheiro .npz do AttractorLexicon (default: attractor_lexicon)")
    parser.add_argument("--lang-file", default="language_index",
                        help="Ficheiro .npz do LanguageIndex (default: language_index)")
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--reassess-radius", type=float, default=0.5)
    parser.add_argument("--reassess-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--output-mode", default="none",
                        choices=["none", "retrieval", "generative", "both"])
    parser.add_argument("--no-logfile", action="store_true", default=False,
                        help="Desactivar log de inputs/outputs (default: activo)")
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
    parser.add_argument("--dream-atomize-every", type=int, default=None,
                        help="Frequência de atomização (default: 3x rg_every)")
    # Metacognition
    parser.add_argument("--vocab-file", type=str, default=None,
                        help="Vocabulário para reflexão semântica (um token por linha)")
    parser.add_argument("--think-every", type=int, default=100,
                        help="Reflexão metacognitiva a cada N ciclos de sonho (default: 100)")
    parser.add_argument("--think-steps", type=int, default=5,
                        help="Passos por ciclo de reflexão (default: 5)")
    parser.add_argument("--reflect", action="store_true",
                        help="Executa reflexão autónoma após processamento")
    parser.add_argument("--reflect-iterations", type=int, default=3,
                        help="Iterações de reflexão (default: 3)")
    parser.add_argument("--narrative-token-weight", type=float, default=0.5,
                        help="Peso do token emitido na trajectória (default: 0.5)")
    parser.add_argument("--narrative-input-weight", type=float, default=0.5,
                        help="Peso do input na trajectória (default: 0.5)")
    parser.add_argument("--narrative-temperature", type=float, default=0.7,
                        help="Temperatura de amostragem (default: 0.7)")
    parser.add_argument("--narrative-max-tokens", type=int, default=20,
                        help="Máximo de tokens por narrativa (default: 20)")
    parser.add_argument("--diagnostics-file", type=str, default=None,
                        help="Stem para ficheiros de diagnóstico (ex: diag → diag.jsonl + diag.csv)")
    args = parser.parse_args()

    # Ajusta a constante PATH_BITS global (simplificação)
    global PATH_BITS
    PATH_BITS = args.path_bits

    if args.init:
        sem_npz = args.sem_file if args.sem_file.endswith(".npz") else args.sem_file + ".npz"
        lex_npz  = args.lexicon_file if args.lexicon_file.endswith(".npz") else args.lexicon_file + ".npz"
        lang_npz = args.lang_file if args.lang_file.endswith(".npz") else args.lang_file + ".npz"
        for f in [args.soul, args.tree, args.buffer_file,
                  sem_npz, args.rg_vocab_file, args.rejected_file, lex_npz,
                  lang_npz, lang_npz.replace(".npz", ".bin")]:
            if os.path.exists(f):
                os.remove(f)

    soul = Soul(args.soul, args.tree,
                buffer_capacity=args.buffer_capacity,
                buffer_file=args.buffer_file,
                save_interval=args.save_interval,
                tree_size_mb=args.tree_size_mb,
                vocab_file=args.rg_vocab_file,
                lexicon_file=args.lexicon_file,
                lang_file=args.lang_file)
    _rg_log = args.soul.replace(".json", "_rg_analytics.jsonl")
    soul.rg_analytics = RGAnalyticsLogger(jsonl_path=_rg_log, verbose=True)

    # Purge internal source tokens from vocabulary and LI
    _PURGE = {"dream_reconcile","dream","buffer","abstraction",
              "atomization","assimilation","self_reflection",
              "dream_wakeup","dream_fragment"}
    if hasattr(soul.rg_output, "_tokens"):
        _ct = [(t,v) for t,v in zip(soul.rg_output._tokens,
                                    soul.rg_output._vectors)
               if t.strip().lower() not in _PURGE]
        soul.rg_output._tokens  = [x[0] for x in _ct]
        soul.rg_output._vectors = [x[1] for x in _ct]
    if hasattr(soul, "language_index"):
        _li = soul.language_index
        try:
            _li.tree.flush()
            _NODE_SIZE = _li.tree.node_size
            _fname     = _li.tree.filename
            _n_nodes   = os.path.getsize(_fname) // _NODE_SIZE
            _removed   = 0
            with open(_fname, "r+b") as _f:
                for _i in range(_n_nodes):
                    _f.seek(_i * _NODE_SIZE)
                    _data = _f.read(_NODE_SIZE)
                    if len(_data) < _NODE_SIZE: break
                    _hd, _vec, _meta = _li.tree._unpack_node(_data)
                    if not _hd or _meta is None: continue
                    _txt = (_meta.get("text","") or _meta.get("source","")).strip().lower()
                    if _txt in _PURGE:
                        _f.seek(_i * _NODE_SIZE)
                        _f.write(b'\x00' * _NODE_SIZE)
                        _removed += 1
            if _removed: print(f"[Purge] LI: {_removed} internal labels removed")
        except Exception as _pe:
            print(f"[Purge] LI error: {_pe}")

    soul._narrative_token_weight = args.narrative_token_weight
    soul._narrative_input_weight = args.narrative_input_weight
    soul._narrative_temperature  = args.narrative_temperature
    soul._narrative_max_tokens   = args.narrative_max_tokens

    soul.rejected_log = RejectedLog(args.rejected_file, tree_size_mb=2)
    print(f"[RejectedLog] {soul.rejected_log.size()} entradas pendentes em '{args.rejected_file}'")

    # Ensure output files exist from first run
    if not os.path.exists(args.rg_vocab_file):
        open(args.rg_vocab_file, 'w').close()
    lex_npz2  = args.lexicon_file if args.lexicon_file.endswith(".npz") else args.lexicon_file + ".npz"
    lang_npz2 = args.lang_file if args.lang_file.endswith(".npz") else args.lang_file + ".npz"
    if not os.path.exists(lex_npz2):
        soul.attractor_lexicon.save(args.lexicon_file)
    if not os.path.exists(lang_npz2):
        soul.language_index.save(args.lang_file)

    diagnostics = (DiagnosticsLogger(args.diagnostics_file)
                   if args.diagnostics_file else None)
    logger = None if args.no_logfile else SessionLogger(args.soul)

    print(f"[CoTa Hypernode] soul={args.soul} tree={args.tree}")
    print(f"tree_size_mb={args.tree_size_mb}, path_bits={args.path_bits}")

    # Load semantic decoder if vocab provided
    decoder = None
    if args.vocab_file:
        decoder = SemanticDecoder.from_file(args.vocab_file)
    elif (args.file and not args.binary and
          (args.start_dreaming or args.reflect)):
        # Auto-build vocab from corpus if no explicit vocab given
        with open(args.file, encoding='utf-8', errors='replace') as _f:
            _corpus = _f.read()
        decoder = SemanticDecoder.from_corpus(_corpus, min_freq=2, max_words=3000)

    input_count = 0

    def process(vec, source):
        nonlocal input_count
        input_count += 1
        if soul.evolve(vec, source=source):
            produce_output(soul, args.output_mode, input_vec=vec)
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
                # Store raw bytes so retrieval can recover original content
                sources.append(chunk)

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
                            produce_output(soul, args.output_mode,
                                           input_vec=vecs[j])
                input_count += len(b_chunks)
                if input_count % args.reassess_interval < BATCH:
                    soul.reassess_buffer(args.reassess_radius)
            print()
        else:
            with open(args.file, encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f if l.strip()]

            for b_start in range(0, len(lines), BATCH):
                b_lines = lines[b_start:b_start+BATCH]
                for line in b_lines:
                    accepted = soul.evolve_sequence(line, source=line[:50])
                    status = "[Accepted]" if accepted else "[Buffered]"
                    print(f"{status} >> {line[:80]}")
                    if logger: logger.log_input(line, accepted, soul.tau)
                    if args.output_mode != 'none' and accepted:
                        _vec = bytes_to_vector(line.encode('utf-8', errors='replace'))
                        produce_output(soul, args.output_mode, input_vec=_vec,
                                       logger=logger)
                input_count += len(b_lines)
                if input_count % args.reassess_interval < BATCH:
                    soul.reassess_buffer(args.reassess_radius)
    else:
        # Interactive mode: regulatory feedback loop
        print("Modo interactivo. Ctrl+D para sair.\n")
        try:
            while True:
                line = input("> ")
                if line:
                    vec      = bytes_to_vector(line)
                    accepted = soul.evolve_sequence(line, source=line)
                    if logger: logger.log_input(line, accepted, soul.tau)
                    if not accepted:
                        chi = soul.compute_susceptibility()
                        if chi < 0.03:
                            soul._dream_step(force_scale=0.3)
                        soul.reassess_buffer(
                            radius=args.reassess_radius * 1.5,
                            assimilate=True)
                        print(f"[chi={chi:.4f}] micro-dream + reassess "
                              f"(buf={soul.buffer.size()})")
                    produce_output(soul, 'generative', input_vec=vec, probe=True)
        except EOFError:
            pass

    if args.reflect and decoder:
        soul.reflect(decoder,
                     iterations=args.reflect_iterations,
                     think_steps=args.think_steps)

    if args.start_dreaming:
        _dream_res = soul.dream(
            cycles=args.dream_cycles,
            merge_radius=args.dream_merge_radius,
            rg_every=args.dream_rg_every,
            atomize_every=args.dream_atomize_every,
            decoder=decoder,
            think_every=args.think_every,
            think_steps=args.think_steps,
            diagnostics=diagnostics,
        )
        if logger: logger.log_dream(_dream_res)

    stats = soul.get_stats()
    print("\n" + "="*50)
    print(f"τ final:        {stats['tau']:.4f}")
    print(f"Total inputs:   {stats['total_inputs']}")
    print(f"Aceites:        {stats['accepted']} ({stats['accepted']/max(1,stats['total_inputs'])*100:.1f}%)")
    print(f"Rejeitados:     {stats['rejected']}")
    print(f"Buffer size:    {stats['buffer_size']}")
    print(f"Norma do estado:{stats['state_norm']:.4f}")
    print(f"Expansion rate:     {stats['expansion_rate']:.4f} ({stats['regime']})")
    print(f"RGOutput vocab: {stats['rg_vocab_size']} tokens")
    print("="*50)

    if hasattr(soul, "rejected_log") and soul.rejected_log:
        soul.rejected_log.close()
    if logger: logger.close()
    soul.close()

if __name__ == "__main__":
    main()