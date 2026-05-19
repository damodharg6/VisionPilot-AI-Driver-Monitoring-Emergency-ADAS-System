from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np

from config import EAR_THRESH
from driver_monitoring.eye_tracking import eye_aspect_ratio, landmark_points


@dataclass
class DriverMetrics:
    face_detected: bool = False
    ear: float = 0.0
    ear_threshold: float = EAR_THRESH
    mouth_ratio: float = 0.0
    head_tilt: float = 0.0
    fatigue_score: float = 0.0
    attention_score: float = 100.0
    closure_duration: float = 0.0
    yawn_frequency: float = 0.0
    blink_rate: float = 0.0
    state: str = "NO FACE"


class FatigueAnalyzer:
    def __init__(self, base_ear_threshold=EAR_THRESH):
        self.base_ear_threshold = base_ear_threshold
        self.adaptive_threshold = base_ear_threshold
        self.open_ear_samples = deque(maxlen=180)
        self.ear_history = deque(maxlen=90)
        self.closed_since = None
        self.last_closed = False
        self.blink_times = deque(maxlen=60)
        self.yawn_times = deque(maxlen=20)
        self.yawn_active = False
        self.no_face_since = None

    def update(self, landmarks, width, height, left_eye, right_eye, mouth, tilt_points):
        now = time.time()
        if not landmarks:
            if self.no_face_since is None:
                self.no_face_since = now
            fatigue = min(100.0, 45.0 + (now - self.no_face_since) * 12.0)
            return DriverMetrics(
                face_detected=False,
                fatigue_score=fatigue,
                attention_score=max(0.0, 100.0 - fatigue),
                state="NO FACE",
            )

        self.no_face_since = None
        left = landmark_points(landmarks, left_eye, width, height)
        right = landmark_points(landmarks, right_eye, width, height)
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        self.ear_history.append(ear)

        closed = ear < self.adaptive_threshold
        if not closed:
            self.open_ear_samples.append(ear)
            if len(self.open_ear_samples) >= 40:
                open_baseline = float(np.percentile(self.open_ear_samples, 35))
                self.adaptive_threshold = float(np.clip(open_baseline * 0.72, 0.18, 0.30))

        if closed and not self.last_closed:
            self.closed_since = now
        if not closed and self.last_closed:
            duration = now - self.closed_since if self.closed_since else 0.0
            if 0.08 <= duration <= 0.65:
                self.blink_times.append(now)
            self.closed_since = None
        self.last_closed = closed
        closure_duration = now - self.closed_since if closed and self.closed_since else 0.0

        mouth_ratio = self._mouth_open_ratio(landmarks, mouth, width, height)
        yawning = mouth_ratio > 0.62
        if yawning and not self.yawn_active:
            self.yawn_times.append(now)
        self.yawn_active = yawning

        head_tilt = self._head_tilt_degrees(landmarks, tilt_points, width, height)
        blink_rate = self._events_per_minute(self.blink_times, now, window=60.0)
        yawn_frequency = self._events_per_minute(self.yawn_times, now, window=120.0)

        closure_risk = min(1.0, closure_duration / 3.2)
        ear_risk = np.clip((self.adaptive_threshold - ear) / 0.09, 0.0, 1.0)
        yawn_risk = min(1.0, yawn_frequency / 2.4)
        tilt_risk = np.clip((abs(head_tilt) - 10.0) / 22.0, 0.0, 1.0)
        blink_risk = np.clip((blink_rate - 24.0) / 18.0, 0.0, 1.0)
        instability = float(np.std(self.ear_history)) if len(self.ear_history) > 10 else 0.0
        stability_risk = np.clip(instability / 0.065, 0.0, 1.0)

        fatigue = 100.0 * (
            closure_risk * 0.34
            + ear_risk * 0.20
            + yawn_risk * 0.18
            + tilt_risk * 0.16
            + blink_risk * 0.06
            + stability_risk * 0.06
        )
        attention = np.clip(100.0 - fatigue - max(0.0, closure_duration - 0.45) * 8.0, 0.0, 100.0)

        if fatigue >= 78:
            state = "CRITICAL FATIGUE"
        elif fatigue >= 55:
            state = "DROWSY"
        elif fatigue >= 32:
            state = "FATIGUE RISK"
        else:
            state = "ATTENTIVE"

        return DriverMetrics(
            face_detected=True,
            ear=float(ear),
            ear_threshold=float(self.adaptive_threshold),
            mouth_ratio=float(mouth_ratio),
            head_tilt=float(head_tilt),
            fatigue_score=float(np.clip(fatigue, 0.0, 100.0)),
            attention_score=float(attention),
            closure_duration=float(closure_duration),
            yawn_frequency=float(yawn_frequency),
            blink_rate=float(blink_rate),
            state=state,
        )

    @staticmethod
    def _events_per_minute(events, now, window):
        while events and now - events[0] > window:
            events.popleft()
        return len(events) * (60.0 / window)

    @staticmethod
    def _mouth_open_ratio(landmarks, mouth, width, height):
        top, bottom, left, right = landmark_points(landmarks, mouth, width, height)
        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)
        return float(vertical / horizontal) if horizontal else 0.0

    @staticmethod
    def _head_tilt_degrees(landmarks, tilt_points, width, height):
        a, b = landmark_points(landmarks, tilt_points, width, height)
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

