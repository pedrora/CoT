# loss_total = task_loss + lambda_rc * (1 - RC_score)
# lambda_rc = f(sqrt_p_structure)
# loss_total.backward()

def compute_dynamic_lambda(rc_score, epsilon=0.64):
    """
    Calcula o peso da coerência baseado na 'raiz da estrutura'.
    Se a estrutura está a divergir (Fisher alto), o lambda sobe 
    para forçar a correção.
    """
    # RC_score no teu código é a métrica de coerência (0 a 1)
    # A 'pressão' da estrutura é o inverso da coerência
    p_structure = 1.0 - rc_score
    
    # f(sqrt(p)) -> Seguindo a métrica de Fisher para amplitudes
    dynamic_lambda = epsilon * torch.sqrt(torch.tensor(p_structure) + 1e-8)
    
    return dynamic_lambda

# No loop de treino:
RC_score, _ = siphon_score(h, history)
lambda_rc = compute_dynamic_lambda(RC_score)

# Ação Lagrangeana Total: Minimização da Ação (S)
loss_total = task_loss + lambda_rc * (1 - RC_score)