import cv2
import numpy as np
import torch

def draw_soul_navigator(history: list, current_dot: torch.Tensor, rc_score: float):
    canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Background Poincaré disk
    center = (400, 300)
    radius = 250
    cv2.circle(canvas, center, radius, (40, 40, 40), 3)          # outer boundary
    cv2.circle(canvas, center, int(radius * 0.95), (70, 70, 70), 1)
    
    # Draw archived history (older = smaller & dimmer)
    for i, dot in enumerate(history):
        if torch.norm(dot) > 0.999: continue
        pos = (dot * radius * 0.95).numpy() + np.array(center)
        alpha = max(0.3, i / len(history))                     # older = dimmer
        color = (0, int(255 * alpha), 100) if i == len(history)-1 else (80, 80, 120)
        cv2.circle(canvas, tuple(pos.astype(int)), 6, color, -1)
    
    # Current soul (bright, pulsing)
    if torch.norm(current_dot) > 0:
        pos = (current_dot * radius * 0.95).numpy() + np.array(center)
        cv2.circle(canvas, tuple(pos.astype(int)), 12, (0, 255, 255), -1)
        cv2.circle(canvas, tuple(pos.astype(int)), 14, (0, 200, 255), 2)
    
    # Telemetry
    cv2.putText(canvas, f"Coherence: {rc_score:.4f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, f"Archived dots: {len(history)}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    
    # Title
    cv2.putText(canvas, "COTA — SOUL NAVIGATOR", (180, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 180, 255), 2)
    
    cv2.imshow("Soul Navigator", canvas)
    cv2.waitKey(1)
 
# get_phase_rgb(current_dot)
# extracts naturally expressable colours in RGB that you can represent in a 2D screen
import colorsys

def get_phase_rgb(current_dot):
    """
    Extrai a cor RGB baseada na fase (ângulo) e norma do vetor no Q4.
    """
    with torch.no_grad():
        # 1. Calculamos o ângulo médio (Hue) do vetor no espaço
        # Usamos dois eixos representativos para definir a 'direção do pensamento'
        x, y = current_dot[0].item(), current_dot[1].item()
        angle = np.arctan2(y, x)
        hue = (angle + np.pi) / (2 * np.pi) # Normalizado 0-1
        
        # 2. Saturação baseada na Norma (Distância ao centro)
        # No centro (indecisão/origem) é branco. Na borda (certeza/especialização) é cor pura.
        norm = torch.norm(current_dot).item()
        saturation = np.clip(norm, 0, 1)
        
        # 3. Valor (Brilho) baseado na Coerência (RC Score)
        # Se o sistema está confuso, a alma escurece.
        value = 1.0 
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return tuple(int(c * 255) for c in rgb)

# No teu soul_navigator.py, substitui o amarelo fixo:
soul_color = get_phase_rgb(current_dot)
cv2.circle(canvas, tuple(pos.astype(int)), 12, soul_color, -1)