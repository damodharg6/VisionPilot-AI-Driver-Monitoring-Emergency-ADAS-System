import time
import cv2
import numpy as np


def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1, alpha=0.7):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius), (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_boot_sequence(frame, boot_steps, start_time):
    elapsed = time.time() - start_time
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (8, 10, 16), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cx = w // 2
    cy = h // 2 - 70
    cv2.putText(frame, "VISIONPILOT-AI PROTOTYPE", (max(20, cx - 210), cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 235, 235), 2)
    progress = min(1.0, elapsed / sum(step[1] for step in boot_steps))
    cv2.rectangle(frame, (cx - 190, cy + 32), (cx + 190, cy + 42), (32, 38, 48), -1)
    cv2.rectangle(frame, (cx - 190, cy + 32), (cx - 190 + int(380 * progress), cy + 42), (0, 210, 255), -1)

    t = 0.0
    y = cy + 82
    for text, duration in boot_steps:
        active = elapsed >= t
        done = elapsed >= t + duration
        color = (0, 210, 120) if done else ((0, 210, 255) if active else (80, 85, 95))
        prefix = "ONLINE" if done else ("SYNC" if active else "WAIT")
        cv2.putText(frame, f"{prefix}  {text}", (cx - 190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1)
        y += 24
        t += duration
    return progress >= 1.0

