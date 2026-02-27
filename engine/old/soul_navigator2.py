import cv2
import numpy as np
import torch
from typing import List, Optional

# Global variables for interaction
selected_dot_index = -1
popup_text = ""

def extract_soul_color(dot: torch.Tensor, score: float, curvature: float) -> tuple:
    """Map coherence & curvature to HSV → BGR color"""
    hue = int(120 * score)                # green (high coherence) → red (low)
    saturation = int(255 * (1 - curvature))
    value = 220
    hsv = np.uint8([[[hue, saturation, value]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in bgr)

def world_to_screen(pos: np.ndarray, center: tuple, radius: int) -> tuple:
    """Map Poincaré coordinate [-1,1] to screen pixels"""
    x = int(center[0] + pos[0] * radius * 0.95)
    y = int(center[1] - pos[1] * radius * 0.95)  # y inverted for screen coords
    return (x, y)

def draw_soul_navigator(
    history: List[torch.Tensor],
    current_dot: torch.Tensor,
    rc_scores: List[float],
    curvatures: List[float],
    concepts: List[str] = None           # optional: original text/metadata for each dot
):
    global selected_dot_index, popup_text

    canvas = np.zeros((700, 900, 3), dtype=np.uint8)

    # ─── Background ────────────────────────────────────────
    center = (450, 350)
    disk_radius = 280
    cv2.circle(canvas, center, disk_radius, (30, 30, 40), -1)           # dark disk
    cv2.circle(canvas, center, disk_radius, (80, 80, 100), 2)           # boundary

    # ─── Archived history dots & trails ────────────────────
    trail_points = []
    for i, dot in enumerate(history):
        if torch.norm(dot) >= 0.999: continue
        pos_norm = dot.numpy()[:2]  # take first 2 dims for 2D viz
        screen_pos = world_to_screen(pos_norm, center, disk_radius)

        age_alpha = max(0.4, i / max(1, len(history) - 1))
        score = rc_scores[i] if i < len(rc_scores) else 0.5
        curv = curvatures[i] if i < len(curvatures) else 0.5
        color = extract_soul_color(dot, score, curv)

        size = 5 + int(8 * age_alpha)
        cv2.circle(canvas, screen_pos, size, color, -1)

        # Trail line to previous
        if i > 0:
            prev_pos = world_to_screen(history[i-1].numpy()[:2], center, disk_radius)
            trail_points.append(screen_pos)
            cv2.line(canvas, prev_pos, screen_pos, (100, 100, 150), 1, cv2.LINE_AA)

    # ─── Current soul (pulsing highlight) ──────────────────
    if torch.norm(current_dot) > 0:
        curr_norm = current_dot.numpy()[:2]
        curr_screen = world_to_screen(curr_norm, center, disk_radius)
        cv2.circle(canvas, curr_screen, 16, (0, 220, 255), -1)
        cv2.circle(canvas, curr_screen, 18, (0, 180, 255), 3)

    # ─── Telemetry panel ───────────────────────────────────
    latest_score = rc_scores[-1] if rc_scores else 0.0
    latest_curv  = curvatures[-1] if curvatures else 0.0
    cv2.rectangle(canvas, (20, 20), (300, 180), (40, 40, 60), -1)
    cv2.rectangle(canvas, (20, 20), (300, 180), (120, 120, 140), 2)

    texts = [
        ("COTA SOUL NAVIGATOR", (255, 220, 100), 1.1, (30, 50)),
        (f"Coherence: {latest_score:.4f}", (200, 255, 200), 0.8, (30, 90)),
        (f"Curvature: {latest_curv:.4f}", (180, 180, 255), 0.8, (30, 120)),
        (f"History size: {len(history)}", (220, 220, 220), 0.7, (30, 150)),
    ]
    for txt, col, scale, pos in texts:
        cv2.putText(canvas, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 2)

    # ─── Popup for selected dot ────────────────────────────
    if selected_dot_index >= 0 and selected_dot_index < len(history):
        sel_pos = world_to_screen(history[selected_dot_index].numpy()[:2], center, disk_radius)
        cv2.circle(canvas, sel_pos, 20, (255, 255, 0), 4)
        
        popup = f"Dot #{selected_dot_index}   Score: {rc_scores[selected_dot_index]:.4f}"
        if concepts and selected_dot_index < len(concepts):
            popup += f"\n{concepts[selected_dot_index][:60]}..."
        
        popup_lines = popup.split('\n')
        for i, line in enumerate(popup_lines):
            cv2.putText(canvas, line, (sel_pos[0] + 25, sel_pos[1] + 20 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200), 2)

    cv2.imshow("Soul Navigator — COTA", canvas)

# ─── Mouse callback ────────────────────────────────────────
def mouse_callback(event, x, y, flags, param):
    global selected_dot_index
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_dot_index = -1
        history, center, radius = param
        for i, dot in enumerate(history):
            pos = world_to_screen(dot.numpy()[:2], center, radius)
            dist = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if dist < 25:
                selected_dot_index = i
                break

# ─── Example usage in your main loop ───────────────────────
if __name__ == "__main__":
    # Dummy data for testing
    history = [torch.randn(64) * 0.3 for _ in range(20)]
    current = torch.randn(64) * 0.1 + history[-1] * 0.7 if history else torch.zeros(64)
    scores = [0.85 - i*0.02 for i in range(len(history))]
    curvatures = [0.05 + i*0.01 for i in range(len(history))]
    concepts = [f"Concept {i}: some meaningful text here..." for i in range(len(history))]

    cv2.namedWindow("Soul Navigator — COTA")
    cv2.setMouseCallback("Soul Navigator — COTA", mouse_callback,
                         param=(history, (450, 350), 280))

    while True:
        draw_soul_navigator(history, current, scores, curvatures, concepts)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()