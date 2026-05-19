import math
import time

import cv2
import numpy as np

from simulation.cinematic_effects import draw_rounded_rect


def fmt_time(secs):
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


class HudRenderer:
    def __init__(self):
        self.last_fps_time = time.time()
        self.frame_counter = 0
        self.fps = 0.0

    def update_fps(self):
        self.frame_counter += 1
        now = time.time()
        if now - self.last_fps_time >= 0.5:
            self.fps = self.frame_counter / (now - self.last_fps_time)
            self.last_fps_time = now
            self.frame_counter = 0
        return self.fps

    def draw_driver_panel(self, frame, metrics, cam_label, drowsy_events, total_sleep, session_start):
        h, w = frame.shape[:2]
        panel_w = min(330, max(292, w // 3))
        px1, py1, px2, py2 = w - panel_w - 14, 14, w - 14, h - 18
        draw_rounded_rect(frame, (px1, py1), (px2, py2), (8, 11, 17), radius=8, alpha=0.84)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (58, 74, 82), 1)
        cv2.line(frame, (px1 + 18, py1 + 48), (px2 - 18, py1 + 48), (42, 58, 65), 1)

        cv2.putText(frame, "DRIVER BIOMETRICS", (px1 + 18, py1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 225, 235), 1)
        state_color = self._state_color(metrics.state)
        cv2.putText(frame, metrics.state, (px1 + 18, py1 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.78, state_color, 2)

        y = py1 + 112
        self._bar(frame, px1 + 18, y, "FATIGUE CONFIDENCE", metrics.fatigue_score, (0, 120, 255), width=panel_w - 76); y += 46
        self._bar(frame, px1 + 18, y, "ATTENTION SCORE", metrics.attention_score, (0, 215, 125), width=panel_w - 76); y += 48

        left_x = px1 + 18
        right_x = px1 + panel_w // 2 + 8
        self._metric(frame, left_x, y, "EAR", f"{metrics.ear:.3f}", (235, 242, 242))
        self._metric(frame, right_x, y, "CALIBRATED", f"{metrics.ear_threshold:.3f}", (185, 205, 210)); y += 46
        self._metric(frame, left_x, y, "EYE CLOSURE", f"{metrics.closure_duration:.1f}s")
        self._metric(frame, right_x, y, "YAWN RATE", f"{metrics.yawn_frequency:.1f}/min"); y += 46
        self._metric(frame, left_x, y, "HEAD TILT", f"{metrics.head_tilt:+.1f} deg")
        self._metric(frame, right_x, y, "EVENTS", str(drowsy_events)); y += 46
        self._metric(frame, left_x, y, "SLEEP TIME", fmt_time(total_sleep))
        self._metric(frame, right_x, y, "UPTIME", fmt_time(time.time() - session_start)); y += 48

        cv2.line(frame, (px1 + 18, y), (px2 - 18, y), (42, 58, 65), 1)
        cv2.putText(frame, "CAMERA", (px1 + 18, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 136, 146), 1)
        cv2.putText(frame, cam_label[:24], (px1 + 18, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 238, 238), 1)

    def draw_bottom_telemetry(self, frame, adas):
        h, w = frame.shape[:2]
        py1, py2 = h - 82, h - 8
        draw_rounded_rect(frame, (258, py1), (w - 8, py2), (13, 15, 22), radius=8, alpha=0.88)
        cv2.rectangle(frame, (258, py1), (w - 8, py2), (55, 68, 78), 1)
        fps = self.update_fps()
        items = [
            ("SPEED", f"{adas.speed_kmh:05.1f} km/h"),
            ("ADAS", adas.mode),
            ("RISK", f"{adas.collision_risk * 100:03.0f}%"),
            ("TTC", f"{adas.ttc:04.1f}s"),
            ("SENSOR", adas.sensor_status),
            ("MANEUVER", adas.maneuver),
            ("STAGE", str(adas.stage)),
            ("FPS", f"{fps:04.1f}"),
        ]
        x = 274
        for label, value in items:
            cv2.putText(frame, label, (x, py1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (115, 125, 140), 1)
            cv2.putText(frame, value, (x, py1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (225, 230, 235), 1)
            x += max(78, len(value) * 9 + 24)

    def draw_alert_overlay(self, frame, adas, metrics):
        if adas.stage <= 0:
            return
        h, w = frame.shape[:2]
        alpha = 0.08 + 0.08 * min(1.0, adas.stage / 4.0)
        color = (0, 80, 255) if adas.stage < 3 else (0, 0, 190)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        pulse = int(90 + 90 * abs(math.sin(time.time() * 5.0)))
        if adas.stage >= 3:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, pulse), 5)
        text = ["CAUTION: FATIGUE", "WARNING: RESPOND", "ADAS TAKEOVER", "SAFE STOP PROTOCOL"][min(adas.stage, 4) - 1]
        cv2.putText(frame, text, (262, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 210, 255), 2)

    @staticmethod
    def _metric(frame, x, y, label, value, color=(230, 230, 235)):
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 126, 138), 1)
        cv2.putText(frame, str(value), (x, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

    @staticmethod
    def _bar(frame, x, y, label, value, color, width=194):
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 126, 138), 1)
        cv2.rectangle(frame, (x, y + 12), (x + width, y + 26), (30, 36, 45), -1)
        fill = int(width * max(0.0, min(1.0, value / 100.0)))
        cv2.rectangle(frame, (x, y + 12), (x + fill, y + 26), color, -1)
        cv2.rectangle(frame, (x, y + 12), (x + width, y + 26), (70, 84, 92), 1)
        cv2.putText(frame, f"{value:03.0f}", (x + width + 10, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 230, 230), 1)

    @staticmethod
    def _state_color(state):
        if "CRITICAL" in state or "DROWSY" in state:
            return (0, 80, 255)
        if "RISK" in state or "NO FACE" in state:
            return (0, 180, 255)
        return (0, 220, 120)
