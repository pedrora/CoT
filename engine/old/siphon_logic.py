import torch
import torch.nn.functional as F

EPS = 1e-8


# -------------------------------------------------------
# 1. POINCARÉ PROJECTION
# -------------------------------------------------------

def to_poincare(x):
    """
    Project Euclidean embedding into Poincaré ball.
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    return torch.tanh(norm) * x / (norm + EPS)


# -------------------------------------------------------
# 2. HYPERBOLIC DISTANCE
# -------------------------------------------------------

def poincare_distance(u, v):
    """
    Geodesic distance in Poincaré ball.
    """
    uu = torch.sum(u*u, dim=-1)
    vv = torch.sum(v*v, dim=-1)
    diff = torch.sum((u - v)**2, dim=-1)

    denom = (1 - uu) * (1 - vv) + EPS
    argument = 1 + 2 * diff / denom

    return torch.acosh(torch.clamp(argument, min=1+EPS))


# -------------------------------------------------------
# 3. PHASE CONTINUITY (proto-phase transport)
# -------------------------------------------------------

def phase_continuity(x_t, x_prev):
    """
    Measures directional continuity (parallel transport proxy).
    """
    return F.cosine_similarity(x_t, x_prev, dim=-1)


# -------------------------------------------------------
# 4. GEODESIC SMOOTHNESS
# -------------------------------------------------------

def geodesic_smoothness(current, history):
    """
    Penalizes large jumps in hyperbolic space.
    """
    if len(history) == 0:
        return torch.tensor(1.0)

    dists = [
        poincare_distance(current, h)
        for h in history[-3:]   # short memory window
    ]

    dists = torch.stack(dists)
    return torch.exp(-torch.mean(dists))


# -------------------------------------------------------
# 5. CURVATURE NOISE
# -------------------------------------------------------

def curvature_noise(current, history):
    """
    Detects incoherent trajectory curvature.
    """
    if len(history) < 3:
        return torch.tensor(0.0)

    d = [
        poincare_distance(history[i], history[i-1])
        for i in range(1, len(history))
    ]

    d = torch.stack(d[-5:])
    return torch.var(d)


# -------------------------------------------------------
# 6. RC / SIPHON SCORE
# -------------------------------------------------------

def siphon_score(current, history,
                 w_geo=1.0,
                 w_phase=1.0,
                 w_curv=1.0):

    geo = geodesic_smoothness(current, history)

    if len(history) > 0:
        phase = phase_continuity(current, history[-1])
    else:
        phase = torch.tensor(1.0)

    curv = curvature_noise(current, history)

    score = (
        w_geo * geo
        + w_phase * phase
        - w_curv * curv
    )

    diagnostics = {
        "geodesic": geo.item(),
        "phase": phase.item(),
        "curvature": curv.item(),
        "RC_score": score.item()
    }

    return score, diagnostics
