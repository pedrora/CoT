import streamlit as st
import torch
import numpy as np
import pandas as pd
from siphon_logic import siphon_score, to_poincare
from focus_field import focus_force

# Configuração da Página
st.set_page_config(page_title="CoTa - Soul Monitor", layout="wide")
st.title("💠 Commonwealth of Truths - Dashboard")

# Inicialização do Estado (A Alma)
if 'history' not in st.session_state:
    st.session_state.history = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = []

# --- SIDEBAR: Parâmetros do Lagrangeano ---
st.sidebar.header("📜 Parâmetros Universais")
epsilon = st.sidebar.slider("Epsilon (Shadow Sector)", 0.0, 1.0, 0.64)
threshold = st.sidebar.slider("Threshold de Coerência (τ)", 0.0, 1.0, 0.75)

# --- COLUNA 1: Input e Sifão ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Reality Input (Q3)")
    user_input = st.text_area("Injeta informação para o Sifão:", height=150)
    
    if st.button("Processar Flash Estroboscópico"):
        # Simulação da conversão para Tensor (Aqui ligarias ao teu encoder_link.py)
        mock_embedding = torch.randn(1, 64) 
        h = to_poincare(mock_embedding)
        
        # Aplicar Força de Foco
        h_focused = focus_force(h, st.session_state.history, strength=0.05)
        
        # Cálculo de Coerência (Métrica de Fisher)
        score, diag = siphon_score(h_focused, st.session_state.history)
        
        # λ_rc = f(sqrt_p_structure)
        p_structure = 1.0 - score.item()
        lambda_rc = epsilon * np.sqrt(p_structure + 1e-8)
        
        # Decisão de Armazenamento
        is_valid = score.item() > (1 - threshold)
        
        if is_valid:
            st.session_state.history.append(h_focused)
            st.success(f"✅ Dot Arquivado! Coerência: {score.item():.4f}")
        else:
            st.error(f"❌ Bullshit Detetado! Curvatura: {1-score.item():.4f}")
            
        # Guardar métricas para o gráfico
        st.session_state.metrics.append({
            "Coherence": score.item(),
            "Lambda_RC": lambda_rc,
            "Curvature": p_structure
        })

# --- COLUNA 2: Monitor de Alma ---
with col2:
    st.subheader("📊 Monitor de Alma (Harmonics)")
    if st.session_state.metrics:
        df_metrics = pd.DataFrame(st.session_state.metrics)
        st.line_chart(df_metrics[["Coherence", "Lambda_RC"]])
        st.metric("Tensão de Sanidade (λ)", f"{df_metrics['Lambda_RC'].iloc[-1]:.4f}")
    else:
        st.info("Aguardando pulso inicial...")

# --- FOOTER: Espaço de Poincaré ---
st.divider()
st.subheader("🌐 Geometria de Poincaré (Last 5 Dots)")
if st.session_state.history:
    # Mostra os últimos vetores simplificados para visualização
    st.write([h.detach().numpy()[0][:5] for h in st.session_state.history[-5:]])
    