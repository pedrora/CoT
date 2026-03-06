#!/usr/bin/env python3
"""
CoTa Minimal Hypernode com Buffer Hiperbólico
=============================================

Implementação mínima de um nó CoTa com:
- Espaço hiperbólico (disco de Poincaré)
- Alma com estado, história e tempo próprio τ
- Coerência (alinhamento de fase + curvatura)
- Buffer de reavaliação hiperbólico para inputs rejeitados
- Integração de input via bytes UTF-8 (sem dependências externas)
- Armazenamento persistente em grelha radial‑angular

Uso:
    python cota_minimal.py [--init] [--file corpus.txt] [--buffer-capacity N] [--reassess-radius R]

Se --init for usado, uma nova alma é criada (ou a existente é apagada).
Se --file for fornecido, o ficheiro é processado linha a linha; caso contrário,
lê do stdin.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from datetime import datetime

# =============================================================================
# CONSTANTES
# =============================================================================
DIM = 64                # Dimensão do espaço de estados (bytes por vector)
EPOCH = datetime(2025, 1, 12, 23, 57, 0)  # Época de referência (não usada directamente aqui)
COHERENCE_THRESHOLD = 0.72
CURVATURE_THRESHOLD = 0.20
EPS = 1e-8

# Parâmetros da grelha de armazenamento (HyperbolicStore)
NUM_RINGS = 16
ANGLES_PER_RING = 32
RECORD_SIZE = DIM * 4 + 256  # vector (float32) + metadados (bytes)

# =============================================================================
# FUNÇÕES HIPERBÓLICAS
# =============================================================================

def to_poincare(x):
    """Projecta um vector euclidiano no disco de Poincaré."""
    norm = torch.norm(x)
    if norm < EPS:
        return x
    return torch.tanh(norm) * x / norm

def poincare_distance(u, v):
    """Distância geodésica entre dois pontos no disco de Poincaré."""
    uu = torch.sum(u * u).item()
    vv = torch.sum(v * v).item()
    diff = torch.sum((u - v) ** 2).item()
    denom = (1 - uu) * (1 - vv) + EPS
    arg = 1 + 2 * diff / denom
    return float(np.arccosh(max(1 + EPS, arg)))

# =============================================================================
# FUNÇÃO DE TEXTO PARA VECTOR (BASEADA EM BYTES)
# =============================================================================

def text_to_vector(text):
    """
    Converte texto num vector de DIM dimensões usando os bytes UTF-8.
    - Obtém a sequência de bytes.
    - Trunca ou preenche para ter exactamente DIM bytes.
    - Normaliza cada byte para [-0.5, 0.5].
    - Projecta no disco de Poincaré.
    """
    raw = torch.tensor([b for b in text.encode('utf-8')], dtype=torch.float32)
    if len(raw) < DIM:
        raw = F.pad(raw, (0, DIM - len(raw)))
    else:
        raw = raw[:DIM]
    raw = raw / 255.0 - 0.5  # agora em [-0.5, 0.5]
    return to_poincare(raw)

# =============================================================================
# ARMAZENAMENTO HIPERBÓLICO (GRELHA RADIAL‑ANGULAR)
# =============================================================================

class HyperbolicStore:
    """
    Armazena conceitos num ficheiro mapeado em memória,
    organizado por anéis radiais e sectores angulares.
    """

    def __init__(self, filename, size_mb=10):
        self.filename = filename
        self.size = size_mb * 1024 * 1024
        self.record_size = RECORD_SIZE

        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                f.truncate(self.size)

        self.f = open(filename, 'r+b')
        self.mmap = np.memmap(self.f, dtype=np.uint8, mode='r+', shape=(self.size,))

        self.rings = NUM_RINGS
        self.sectors_per_ring = [ANGLES_PER_RING] * self.rings
        self._compute_offsets()
        self.next_slot = {}  # dicionário (ring, sec) -> próximo slot livre

    def _compute_offsets(self):
        self.offsets = {}
        current = 0
        for ring in range(self.rings):
            for sec in range(self.sectors_per_ring[ring]):
                self.offsets[(ring, sec)] = current
                current += self.record_size * 10  # 10 slots por sector (simples)
        assert current < self.size, "Ficheiro pequeno demais para a grelha"

    def coordinate_to_sector(self, coord):
        r = torch.norm(coord).item()
        if r < 1e-6:
            ring = 0
        else:
            ring = int(r * self.rings)
            ring = min(self.rings - 1, ring)

        theta = torch.atan2(coord[1], coord[0]).item()
        if theta < 0:
            theta += 2 * np.pi
        sec = int(theta / (2 * np.pi) * self.sectors_per_ring[ring])
        sec = min(self.sectors_per_ring[ring] - 1, sec)
        return ring, sec

    def write_concept(self, coord, metadata):
        ring, sec = self.coordinate_to_sector(coord)
        key = (ring, sec)
        offset = self.offsets[key]
        slot = self.next_slot.get(key, 0)
        if slot * self.record_size >= (self.record_size * 10):
            return None  # sector cheio
        record_offset = offset + slot * self.record_size

        vec_bytes = coord.numpy().astype(np.float32).tobytes()
        meta_json = json.dumps(metadata).encode('utf-8')
        if len(meta_json) > 256:
            meta_json = meta_json[:256]
        else:
            meta_json = meta_json.ljust(256, b'\x00')
        record = vec_bytes + meta_json
        self.mmap[record_offset:record_offset + self.record_size] = np.frombuffer(record, dtype=np.uint8)

        self.next_slot[key] = slot + 1
        return (ring, sec, slot)

    def read_concept(self, ring, sec, slot):
        key = (ring, sec)
        offset = self.offsets[key]
        record_offset = offset + slot * self.record_size
        record = self.mmap[record_offset:record_offset + self.record_size]
        vec = np.frombuffer(record[:DIM*4], dtype=np.float32)
        vec = torch.from_numpy(vec)
        meta_json = record[DIM*4:].tobytes().rstrip(b'\x00')
        try:
            metadata = json.loads(meta_json)
        except:
            metadata = {}
        return vec, metadata

    def close(self):
        self.f.close()

# =============================================================================
# BUFFER HIPERBÓLICO DE REAVALIAÇÃO
# =============================================================================

class HyperbolicBuffer:
    """
    Buffer que armazena inputs rejeitados, organizados pela sua posição no disco.
    A reavaliação procura itens próximos do estado actual e tenta integrá‑los.
    """

    def __init__(self, capacity=1000, max_attempts=5):
        self.capacity = capacity
        self.max_attempts = max_attempts
        self.items = {}  # id -> {vector, metadata, attempts, added_tau}
        self.next_id = 0

    def add(self, vector, metadata):
        """Adiciona um vector rejeitado ao buffer."""
        if len(self.items) >= self.capacity:
            # Remove o mais antigo (menor id) – política simples
            oldest = min(self.items.keys())
            del self.items[oldest]
        self.items[self.next_id] = {
            'vector': vector.clone(),
            'metadata': metadata,
            'attempts': 0,
            'added_tau': None  # podemos guardar o τ no momento da adição futuramente
        }
        self.next_id += 1

    def find_nearby(self, target, radius):
        """
        Retorna uma lista de itens cuja distância hiperbólica a target é < radius.
        """
        nearby = []
        for item in self.items.values():
            dist = poincare_distance(target, item['vector'])
            if dist < radius:
                nearby.append((dist, item))
        nearby.sort(key=lambda x: x[0])
        return [item for _, item in nearby]

    def reassess(self, soul, radius):
        """
        Reavalia itens próximos do estado actual da alma.
        Para cada um, tenta integrar; se aceite, remove do buffer.
        """
        nearby = self.find_nearby(soul.state, radius)
        for item in nearby:
            delta = (item['vector'] - soul.state) * 0.1
            if soul.integrate(delta, source=item['metadata'].get('source', 'buffer')):
                # Integrado com sucesso: remover do buffer
                # (é ineficiente, mas para já serve)
                for key, val in list(self.items.items()):
                    if val is item:
                        del self.items[key]
                        break
            else:
                item['attempts'] += 1
                if item['attempts'] >= self.max_attempts:
                    # Descartar após muitas tentativas
                    for key, val in list(self.items.items()):
                        if val is item:
                            del self.items[key]
                            break

    def size(self):
        return len(self.items)

    def clear(self):
        self.items.clear()
        self.next_id = 0

# =============================================================================
# ALMA (SOUL)
# =============================================================================

class Soul:
    def __init__(self, soul_file="soul.json", store_file="store.bin", buffer_capacity=1000):
        self.soul_file = soul_file
        self.store_file = store_file
        self.store = HyperbolicStore(store_file)
        self.buffer = HyperbolicBuffer(capacity=buffer_capacity)

        if os.path.exists(soul_file):
            self._load()
        else:
            self._create()

        # Estado corrente e histórico
        self.state = to_poincare(torch.randn(DIM) * 0.1)
        self.history = deque(maxlen=100)
        self.tau = 0.0
        self.last_state = self.state.clone()

        # Contadores para estatísticas
        self.total_inputs = 0
        self.accepted = 0
        self.rejected = 0

    def _create(self):
        self.soul_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + "_" + str(os.getpid())
        self.created = datetime.utcnow().isoformat()
        self._save()

    def _load(self):
        with open(self.soul_file, 'r') as f:
            data = json.load(f)
        self.soul_id = data['soul_id']
        self.created = data['created']

    def _save(self):
        with open(self.soul_file, 'w') as f:
            json.dump({
                'soul_id': self.soul_id,
                'created': self.created,
            }, f)

    def coherence_score(self, candidate):
        """Calcula coerência entre candidato e histórico recente."""
        if not self.history:
            return 1.0, 0.0
        last = self.history[-1]
        phase_sim = F.cosine_similarity(candidate.flatten(), last.flatten(), dim=0).item()
        phase_sim = max(0.0, phase_sim)

        if len(self.history) >= 3:
            d1 = poincare_distance(self.history[-1], self.history[-2])
            d2 = poincare_distance(self.history[-2], self.history[-3])
            d3 = poincare_distance(candidate, self.history[-1])
            curvature = abs(d3 - d1) / (d1 + d2 + d3 + EPS)
        else:
            curvature = 0.0

        score = 0.6 * phase_sim + 0.4 * (1 - min(curvature, 1.0))
        return score, curvature

    def integrate(self, delta, source="user"):
        """
        Tenta integrar um delta (vector). Retorna True se aceite.
        """
        self.total_inputs += 1
        candidate = self.state + delta
        candidate = to_poincare(candidate)

        score, curv = self.coherence_score(candidate)

        if score >= COHERENCE_THRESHOLD and curv <= CURVATURE_THRESHOLD:
            # Aceite
            self.history.append(self.state.clone())
            self.tau += poincare_distance(candidate, self.state)
            self.state = candidate

            metadata = {
                'source': source,
                'score': score,
                'curvature': curv,
                'tau': self.tau,
                'timestamp': time.time()
            }
            self.store.write_concept(self.state, metadata)
            self.accepted += 1
            print(f"[Aceite] τ={self.tau:.4f} score={score:.3f} curv={curv:.3f}")
            return True
        else:
            # Rejeitado: guarda no buffer
            self.buffer.add(candidate, {'source': source, 'original': source})
            self.rejected += 1
            print(f"[Rejeitado] score={score:.3f} curv={curv:.3f}")
            return False

    def reassess_buffer(self, radius=0.3):
        """Reavalia o buffer com um determinado raio."""
        before = self.buffer.size()
        self.buffer.reassess(self, radius)
        after = self.buffer.size()
        if before != after:
            print(f"[Buffer] Reavaliação: {before} -> {after} itens")

    def get_stats(self):
        return {
            'tau': self.tau,
            'history_len': len(self.history),
            'state_norm': torch.norm(self.state).item(),
            'total_inputs': self.total_inputs,
            'accepted': self.accepted,
            'rejected': self.rejected,
            'buffer_size': self.buffer.size()
        }

# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="CoTa Minimal Hypernode com Buffer Hiperbólico")
    parser.add_argument("--init", action="store_true", help="Cria nova alma")
    parser.add_argument("--file", type=str, help="Processa ficheiro linha a linha")
    parser.add_argument("--soul", default="soul.json", help="Ficheiro da alma")
    parser.add_argument("--store", default="store.bin", help="Ficheiro de armazenamento")
    parser.add_argument("--buffer-capacity", type=int, default=500, help="Capacidade do buffer")
    parser.add_argument("--reassess-radius", type=float, default=0.3, help="Raio de reavaliação")
    parser.add_argument("--reassess-interval", type=int, default=5, help="Reavaliar a cada N inputs")
    args = parser.parse_args()

    if args.init:
        for f in [args.soul, args.store]:
            if os.path.exists(f):
                os.remove(f)
        soul = Soul(args.soul, args.store, buffer_capacity=args.buffer_capacity)
        print("Nova alma criada.")
    else:
        soul = Soul(args.soul, args.store, buffer_capacity=args.buffer_capacity)
        print("Alma carregada.")

    print(f"Buffer capacity: {args.buffer_capacity}, reassess radius: {args.reassess_radius}")

    # Processamento de input
    input_count = 0
    if args.file:
        with open(args.file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            input_count += 1
            print(f"\n>> {line}")
            vec = text_to_vector(line)
            delta = (vec - soul.state) * 0.1
            soul.integrate(delta, source=line[:50])

            if input_count % args.reassess_interval == 0:
                soul.reassess_buffer(radius=args.reassess_radius)
    else:
        # Modo interactivo
        print("Modo interactivo. Escreve frases (Ctrl+D para sair).")
        try:
            while True:
                line = input("> ")
                if not line:
                    continue
                input_count += 1
                vec = text_to_vector(line)
                delta = (vec - soul.state) * 0.1
                soul.integrate(delta, source=line[:50])

                if input_count % args.reassess_interval == 0:
                    soul.reassess_buffer(radius=args.reassess_radius)
        except EOFError:
            pass

    # Estatísticas finais
    stats = soul.get_stats()
    print("\n" + "="*50)
    print("ESTATÍSTICAS FINAIS")
    print(f"Tempo próprio τ: {stats['tau']:.4f}")
    print(f"Total inputs: {stats['total_inputs']}")
    print(f"Aceites: {stats['accepted']} ({stats['accepted']/max(1,stats['total_inputs'])*100:.1f}%)")
    print(f"Rejeitados: {stats['rejected']}")
    print(f"Tamanho final do buffer: {stats['buffer_size']}")
    print(f"Norma do estado: {stats['state_norm']:.4f}")
    print("="*50)

    soul.store.close()

if __name__ == "__main__":
    main()