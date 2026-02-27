import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
import os
from datetime import datetime

# ─── CONFIG ───
DIM = 64
COHERENCE_τ = st.session_state.get('coherence_tau', 0.82)
CURVATURE_τ = 0.15
MAX_FLASHES = st.session_state.get('max_flashes', 32)
NOISE_LEVEL = st.session_state.get('noise_level', 0.1)

# ─── HELPERS ───
def to_poincare(x):
    norm = torch.norm(x)
    return torch.tanh(norm) * x / (norm + 1e-8)

def renormalize_poincare(x):
    norm = torch.norm(x)
    if norm >= 1.0:
        x = x / (norm + 1e-8) * 0.999
    return x

def focus_force(current, history, strength=0.05):
    if len(history) < 2:
        return current
    directions = []
    for i in range(1, len(history)):
        d = history[i] - history[i-1]
        d = d / (torch.norm(d) + 1e-8)
        directions.append(d)
    if not directions:
        return current
    g = torch.mean(torch.stack(directions), dim=0)
    g = g / (torch.norm(g) + 1e-8)
    proj = torch.dot(current.flatten(), g.flatten()) * g
    corrected = (1 - strength) * current + strength * proj
    return corrected

def coherence_score(current, history):
    if not history:
        return 1.0, 0.0
    phase = torch.cosine_similarity(current.flatten(), history[-1].flatten(), dim=0).item()
    phase = max(0.0, phase)
    curvature = 0.1 if len(history) < 3 else 0.2  # placeholder — replace with real calc
    score = 0.6 * phase + 0.4 * (1 - min(curvature, 1.0))
    return score, curvature

# ─── STATE ───
if 'soul' not in st.session_state:
    st.session_state.soul = {
        'state': None,
        'history': [],
        'coherence_log': [],
        'curvature_log': [],
        'concepts': [],
        'last_update': time.time(),
        'interval': 0.001,
        'running': False,
    }

soul = st.session_state.soul

# ─── SIDEBAR CONTROLS ───
with st.sidebar:
    st.header("Training Controls")
    
    col1, col2 = st.columns(2)
    col1.button("Start Cycle", on_click=lambda: setattr(soul, 'running', True))
    col2.button("Stop Cycle", on_click=lambda: setattr(soul, 'running', False))
    
    st.slider("Coherence Threshold (τ)", 0.5, 0.95, 0.82, key='coherence_tau')
    st.slider("Max flashes per cycle", 5, 64, 32, key='max_flashes')
    st.slider("Internal noise strength", 0.0, 0.5, 0.1, key='noise_level')
    
    injected_text = st.text_area("Inject text / concept", height=120)
    if st.button("Inject") and injected_text.strip():
        st.session_state.pending_text = injected_text.strip()
        st.success("Text queued for next cycle")

# ─── MAIN AREA ───
tab1, tab2, tab3 = st.tabs(["Live Soul", "History & Concepts", "Metrics"])

with tab1:
    st.subheader("Live Soul Position")
    
    fig = go.Figure()
    
    # History trail
    if soul['history']:
        xs = [h[0].item() for h in soul['history']]
        ys = [h[1].item() for h in soul['history']]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers+lines',
                                 marker=dict(size=8, color='blue', opacity=0.6),
                                 line=dict(color='lightblue'), name='History'))
    
    # Current position
    if soul['state'] is not None:
        fig.add_trace(go.Scatter(x=[soul['state'][0].item()], y=[soul['state'][1].item()],
                                 mode='markers', marker=dict(size=16, color='red', symbol='star'),
                                 name='Current Self'))
    
    # Disk boundary
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                             line=dict(color='gray', dash='dash'), showlegend=False))
    
    fig.update_layout(title="Poincaré Disk", xaxis_range=[-1.1,1.1], yaxis_range=[-1.1,1.1],
                      width=600, height=600, showlegend=True)
    st.plotly_chart(fig)

with tab2:
    st.subheader("Archived Concepts")
    if soul['concepts']:
        df = pd.DataFrame({
            'Index': range(len(soul['concepts'])),
            'Concept': soul['concepts'],
            'Coherence': [f"{s:.4f}" for s in soul['coherence_log']],
            'Curvature': [f"{c:.4f}" for c in soul['curvature_log']]
        })
        st.dataframe(df.style.highlight_max(subset=['Coherence'], color='#d4f4dd'))
    else:
        st.info("No concepts archived yet.")

with tab3:
    st.subheader("Live Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Coherence", f"{soul['coherence_log'][-1]:.4f}" if soul['coherence_log'] else "N/A")
    col2.metric("Curvature", f"{soul['curvature_log'][-1]:.4f}" if soul['curvature_log'] else "N/A")
    col3.metric("Interval", f"{soul['interval']*1000:.1f} ms")

# ─── BACKGROUND CYCLE ───
if soul['running']:
    # Input priority: user text > noise
    if 'pending_text' in st.session_state:
        text = st.session_state.pending_text
        del st.session_state.pending_text
        seed = hash(text) % 1000 / 1000.0
        raw_input = torch.randn(DIM) * 0.1 + torch.tensor(seed)
        concept_text = text
    else:
        raw_input = torch.randn(DIM) * NOISE_LEVEL
        concept_text = "Internal noise"

    # Simple tick logic (replace with your full tick if you have it)
    if soul['state'] is None:
        soul['state'] = to_poincare(raw_input)
        soul['history'].append(soul['state'].clone())
        soul['coherence_log'].append(1.0)
        soul['curvature_log'].append(0.0)
        soul['concepts'].append("Initial state")
    else:
        working = soul['state'] + raw_input * 0.3
        working = to_poincare(working)
        score, curv = coherence_score(working, soul['history'])
        
        if score >= COHERENCE_τ and curv <= CURVATURE_τ:
            soul['history'].append(soul['state'].clone())
            soul['state'] = working.clone()
            soul['coherence_log'].append(score)
            soul['curvature_log'].append(curv)
            soul['concepts'].append(concept_text)
        # else: reject silently for now

    time.sleep(0.15)

# Auto-refresh
time.sleep(0.5)
st.rerun()