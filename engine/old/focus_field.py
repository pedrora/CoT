import torch
import torch.nn.functional as F

EPS = 1e-8


# -------------------------------------------------------
# 1. ESTIMATE GEODESIC DIRECTION
# -------------------------------------------------------

def dominant_direction(history):
    """
    Estimate dominant trajectory direction.
    """
    if len(history) < 2:
        return None

    directions = []

    for i in range(1, len(history)):
        d = history[i] - history[i-1]
        d = d / (torch.norm(d) + EPS)
        directions.append(d)

    return torch.mean(torch.stack(directions), dim=0)


# -------------------------------------------------------
# 2. FOCUS FIELD FORCE
# -------------------------------------------------------

def focus_force(current, history, strength=0.05):
    """
    Computes a gentle attraction toward coherent direction.
    """
    g = dominant_direction(history)

    if g is None:
        return current

    g = g / (torch.norm(g) + EPS)

    # projection onto dominant direction
    proj = torch.dot(current, g) * g

    # gentle correction
    corrected = (1 - strength) * current + strength * proj

    return corrected


# -------------------------------------------------------
# 3. NORMALIZE BACK INTO POINCARÉ BALL
# -------------------------------------------------------

def renormalize_poincare(x):
    norm = torch.norm(x)
    if norm >= 1.0:
        x = x / (norm + EPS) * 0.999
    return x
