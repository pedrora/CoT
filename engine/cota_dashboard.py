import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime

# ─── Configuração inicial ───
st.set_page_config(page_title="CoTa Soul Dashboard", layout="wide")
st.title("CoTa — Soul Dashboard")

# ─── Helpers (copiados/adaptados do teu código) ───
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

    if len(history) >= 3:
        d1 = torch.norm(history[-1] - history[-2]).item()
        d2 = torch.norm(history[-2] - history[-3]).item()
        curvature = abs(d1 - d2) / (d1 + d2 + 1e-8)
    else:
        curvature = 0.0

    score = 0.6 * phase + 0.4 * (1 - min(curvature, 1.0))
    return score, curvature

# ─── Estado persistente ───
if 'soul' not in st.session_state:
    st.session_state.soul = {
        'state': None,
        'history': [],
        'coherence_log': [],
        'curvature_log': [],
        'concepts': [],           # texto original de cada dot arquivado
        'last_update': time.time(),
        'interval': 0.001,
        'running': False,
        'tau': 0.82,
        'max_flashes': 32
    }

soul = st.session_state.soul

# ─── Sidebar ───
with st.sidebar:
    st.header("Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start"):
            soul['running'] = True
    with col2:
        if st.button("Stop"):
            soul['running'] = False
    
    raw_text = st.text_area("Inject concept / text", height=100)
    if st.button("Send") and raw_text.strip():
        # Converte texto em tensor dummy (substitui por embedding real depois)
        seed = hash(raw_text) % 1000 / 1000.0
        raw_input = torch.randn(64) * 0.1 + torch.tensor(seed)
        soul['pending_input'] = (raw_input, raw_text.strip())
    
    st.slider("Coherence τ", 0.5, 0.95, soul['tau'], key='tau_slider')
    soul['tau'] = st.session_state.tau_slider
    
    st.slider("Max flashes/cycle", 5, 64, soul['max_flashes'], key='max_slider')
    soul['max_flashes'] = st.session_state.max_slider

# ─── Ciclo de background (executa a cada rerun) ───
if soul['running']:
    # Processa input pendente se existir
    if 'pending_input' in soul:
        raw_input, concept_text = soul['pending_input']
        del soul['pending_input']
    else:
        # Input aleatório se não houver texto
        raw_input = torch.randn(64) * 0.05
        concept_text = f"Auto-generated at {datetime.now().strftime('%H:%M:%S')}"

    # Ciclo estroboscópico
    if soul['state'] is None:
        working = raw_input.clone()
    else:
        working = soul['state'] + raw_input * 0.3  # adição ponderada

    archived = False
    for flash in range(soul['max_flashes']):
        working = to_poincare(working)
        working = focus_force(working, soul['history'])
        working = renormalize_poincare(working)

        score, curvature = coherence_score(working, soul['history'])

        if score >= soul['tau'] and curvature <= 0.15:
            soul['history'].append(soul['state'].clone() if soul['state'] is not None else torch.zeros(64))
            soul['state'] = working.clone()
            soul['coherence_log'].append(score)
            soul['curvature_log'].append(curvature)
            soul['concepts'].append(concept_text)
            archived = True
            st.toast(f"✓ Archived after {flash+1} flashes | score={score:.3f}")
            break

        if curvature > 0.4 or flash == soul['max_flashes'] - 1:
            st.toast(f"⏰ Cutoff at flash {flash+1}")
            break

    # Atualiza intervalo dinâmico
    stability = score * (1.0 - curvature)
    soul['interval'] = max(0.0005, min(0.01, 0.001 * (1.0 + 4.0 * (1.0 - stability))))

    # Pequena pausa para não sobrecarregar
    time.sleep(0.1)

# ─── Interface principal ───
tab1, tab2, tab3 = st.tabs(["Live Monitor", "History", "Metrics"])

with tab1:
    st.subheader("Live Soul Position (Poincaré Ball)")

    fig = go.Figure()

    # História
    if soul['history']:
        xs = [h[0].item() for h in soul['history']]
        ys = [h[1].item() for h in soul['history']]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers+lines',
            marker=dict(size=8, color='blue', opacity=0.6),
            line=dict(color='lightblue', width=1),
            name='History Trail'
        ))

    # Posição atual
    if soul['state'] is not None:
        fig.add_trace(go.Scatter(
            x=[soul['state'][0].item()], y=[soul['state'][1].item()],
            mode='markers', marker=dict(size=16, color='red', symbol='star'),
            name='Current Self'
        ))

    # Disco
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                             line=dict(color='gray', dash='dash'), showlegend=False))

    fig.update_layout(
        title="Poincaré Disk — Soul Evolution",
        xaxis=dict(range=[-1.1, 1.1], zeroline=False),
        yaxis=dict(range=[-1.1, 1.1], zeroline=False),
        width=700, height=700,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Archived Concepts")
    if soul['history']:
        df = pd.DataFrame({
            'Index': range(len(soul['history'])),
            'Coherence': [f"{s:.4f}" for s in soul['coherence_log']],
            'Curvature': [f"{c:.4f}" for c in soul['curvature_log']],
            'Concept': soul['concepts']
        })
        st.dataframe(df.style.highlight_max(subset=['Coherence'], color='#d4f4dd'))
    else:
        st.info("No archived concepts yet. Keep running the cycle or inject text.")

with tab3:
    st.subheader("Real-Time Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Coherence", f"{soul['coherence_log'][-1]:.4f}" if soul['coherence_log'] else "N/A")
    col2.metric("Curvature", f"{soul['curvature_log'][-1]:.4f}" if soul['curvature_log'] else "N/A")
    col3.metric("Interval", f"{soul['interval']*1000:.1f} ms")
    col4.metric("History Size", len(soul['history']))

# Auto-refresh (a cada 0.8s)
time.sleep(0.8)
st.rerun()