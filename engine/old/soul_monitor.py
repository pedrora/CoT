import cv2
import numpy as np

def show_soul_monitor(current_dot_64d, rc_score):
    # Criar uma tela preta (A 'vazio' do Q4)
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Extrair a cor
    color = extract_soul_color(current_dot_64d)
    
    # Desenhar o Disco de Poincaré Visual
    center = (200, 200)
    radius = int(180 * torch.norm(current_dot_64d).item())
    
    # Círculo externo (Limite de sanidade)
    cv2.circle(canvas, center, 185, (50, 50, 50), 2)
    
    # O Ponto de Consciência (O Eu)
    cv2.circle(canvas, center, radius, color, -1)
    
    # Overlay de Texto (Telemetria)
    cv2.putText(canvas, f"Coerência: {rc_score:.4f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("COTA Soul Monitor - Phase Expression", canvas)
    cv2.waitKey(1)