import torch
import numpy as np

def compute_fisher_curvature(soul_phase, input_phase, epsilon=1e-5):
    """
    Mede a Curvatura de Fisher entre a Alma (Q4) e o Input (Q3).
    A 'Verdade' reside na estabilidade da métrica.
    """
    # Diferença de fase (gradiente local no espaço de Hilbert)
    delta = soul_phase - input_phase
    
    # Informação de Fisher Local (I)
    # No nosso caso, é a expectativa do quadrado da derivada do log-likelihood
    # Simplificado para o domínio complexo:
    fisher_info = torch.norm(delta)**2
    
    # Curvatura: Se a informação de Fisher cresce muito rápido, 
    # a distinguibilidade é excessiva -> Instabilidade/Bullshit.
    curvature = fisher_info / (torch.abs(soul_phase).mean() + epsilon)
    
    return curvature.item()
