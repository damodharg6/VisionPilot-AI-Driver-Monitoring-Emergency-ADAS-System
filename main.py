import sys
import time

import cv2
import imutils
import numpy as np

from adas.adas_engine import AdasDecisionEngine
from config import BOOT_STEPS, CAM_LABELS, FRAME_WIDTH, HEAD_TILT_POINTS, LEFT_EYE, MOUTH, RIGHT_EYE
from driver_monitoring.alert_system import AlarmSystem
from driver_monitoring.attention_scoring import AttentionSmoother
from driver_monitoring.eye_tracking import face_bbox, landmark_points
from driver_monitoring.face_detection import FaceDetector
from driver_monitoring.fatigue_analysis import DriverMetrics, FatigueAnalyzer
from simulation.cinematic_effects import draw_boot_sequence
from simulation.highway_renderer import HighwayRenderer
from simulation.hud_renderer import HudRenderer
from simulation.traffic_ai import TrafficAI

PREVIEW_W, PREVIEW_H = 280, 210
PADDING, HEADER_H = 12, 65


def scan_cameras(max_idx=5):
    found = []
    backend = cv2.CAP_MSMF if sys.platform.startswith("win") else cv2.CAP_ANY
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            found.append((i, CAM_LABELS.get(i, f"Camera {i}"), cap))
        else:
            cap.release()
    return found


def build_selector_frame(cameras, highlighted):
    n = len(cameras)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    tw = cols * (PREVIEW_W + PADDING) + PADDING
    th = HEADER_H + rows * (PREVIEW_H + PADDING + 30) + PADDING
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    canvas[:] = (16, 18, 26)

    cv2.putText(canvas, "SELECT CAMERA - AURORA ADAS PROTOTYPE", (PADDING, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 255), 2)
    cv2.putText(canvas, "Number key selects | Arrow/A/D browse | ENTER confirm | Q quit",
                (PADDING, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 155, 165), 1)

    for idx, (cam_idx, label, cap) in enumerate(cameras):
        col = idx % cols
        row = idx // cols
        x = PADDING + col * (PREVIEW_W + PADDING)
        y = HEADER_H + row * (PREVIEW_H + PADDING + 30)
        ret, frame = cap.read()
        if ret:
            thumb = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
        else:
            thumb = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
            cv2.putText(thumb, "No Signal", (72, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 75), 2)
        selected = idx == highlighted
        color = (0, 220, 120) if selected else (58, 64, 78)
        canvas[y:y + PREVIEW_H, x:x + PREVIEW_W] = thumb
        cv2.rectangle(canvas, (x - 2, y - 2), (x + PREVIEW_W + 2, y + PREVIEW_H + 2), color, 2 if selected else 1)
        cv2.putText(canvas, f"[{cam_idx}] {label}", (x, y + PREVIEW_H + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color if selected else (205, 210, 215), 1)
    return canvas


def choose_camera():
    print("[INFO] Scanning cameras...")
    cameras = scan_cameras()
    if not cameras:
        print("[ERROR] No cameras found.")
        return None, None, None

    highlighted = 0
    win = "Aurora ADAS - Camera Selector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow(win, build_selector_frame(cameras, highlighted))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            for _, _, cam in cameras:
                cam.release()
            cv2.destroyWindow(win)
            return None, None, None
        if key in (13, 32):
            cam_idx, label, chosen = cameras[highlighted]
            for i, (_, _, cam) in enumerate(cameras):
                if i != highlighted:
                    cam.release()
            cv2.destroyWindow(win)
            return cam_idx, label, chosen
        if ord("0") <= key <= ord("9"):
            num = key - ord("0")
            for i, (cam_idx, label, chosen) in enumerate(cameras):
                if cam_idx == num:
                    for j, (_, _, cam) in enumerate(cameras):
                        if j != i:
                            cam.release()
                    cv2.destroyWindow(win)
                    return cam_idx, label, chosen
        if key in (81, ord("a")):
            highlighted = (highlighted - 1) % len(cameras)
        if key in (83, ord("d")):
            highlighted = (highlighted + 1) % len(cameras)


def draw_face_tracking(frame, landmarks, metrics, adas):
    if not landmarks:
        return
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face_bbox(landmarks, w, h)
    color = (0, 220, 235) if adas.stage < 2 else ((0, 170, 255) if adas.stage < 3 else (0, 0, 255))
    corner = 18
    for sx, sy, dx, dy in [
        (x1, y1, corner, corner), (x2, y1, -corner, corner),
        (x1, y2, corner, -corner), (x2, y2, -corner, -corner),
    ]:
        cv2.line(frame, (sx, sy), (sx + dx, sy), color, 2)
        cv2.line(frame, (sx, sy), (sx, sy + dy), color, 2)
    label = "FACE LOCK" if metrics.face_detected else "SEARCHING"
    cv2.putText(frame, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    for eye in (LEFT_EYE, RIGHT_EYE):
        pts = landmark_points(landmarks, eye, w, h)
        cv2.drawContours(frame, [cv2.convexHull(pts.astype(np.int32))], -1, (0, 220, 120), 1)


def run_detection(cam_index, cam_label, cap, detector):
    alarm = AlarmSystem()
    fatigue = FatigueAnalyzer()
    attention = AttentionSmoother()
    adas_engine = AdasDecisionEngine()
    traffic = TrafficAI()
    highway = HighwayRenderer()
    hud = HudRenderer()

    session_start = time.time()
    boot_start = time.time()
    boot_duration = sum(step[1] for step in BOOT_STEPS)
    last_time = time.time()
    drowsy_events = 0
    was_drowsy = False
    total_sleep = 0.0
    sleep_start = None
    force_sleep_demo = False
    driver_win = "Aurora Driver Monitor"
    sim_win = "Aurora ADAS Simulation"
    cv2.namedWindow(driver_win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(sim_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(driver_win, 860, 620)
    cv2.resizeWindow(sim_win, 1180, 720)
    print(f"[INFO] Running on {cam_label}. Q=Quit, C=Change camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame grab failed.")
            break

        now = time.time()
        dt = max(0.001, min(0.08, now - last_time))
        last_time = now
        frame = imutils.resize(frame, width=820)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detector.detect(rgb)
        metrics = fatigue.update(landmarks, w, h, LEFT_EYE, RIGHT_EYE, MOUTH, HEAD_TILT_POINTS)
        metrics.attention_score = attention.update(metrics.attention_score)
        if force_sleep_demo:
            metrics.face_detected = True
            metrics.ear = min(metrics.ear, 0.12)
            metrics.fatigue_score = 96.0
            metrics.attention_score = 2.0
            metrics.closure_duration = max(metrics.closure_duration, 3.8)
            metrics.state = "CRITICAL FATIGUE"

        vehicles = traffic.update(dt, adas_engine.physics.speed_mps)
        adas = adas_engine.update(dt, metrics, vehicles)

        is_drowsy_event = adas.stage >= 2
        if is_drowsy_event and not was_drowsy:
            drowsy_events += 1
            sleep_start = now
        if not is_drowsy_event and was_drowsy and sleep_start:
            total_sleep += now - sleep_start
            sleep_start = None
        was_drowsy = is_drowsy_event
        visible_sleep = total_sleep + ((now - sleep_start) if sleep_start else 0.0)

        if adas.stage >= 2 and not adas.secured:
            alarm.start()
        else:
            alarm.stop()

        driver_frame = frame.copy()
        sim_frame = np.zeros((720, 1180, 3), dtype=np.uint8)

        draw_face_tracking(driver_frame, landmarks, metrics, adas)
        hud.draw_alert_overlay(driver_frame, adas, metrics)
        hud.draw_driver_panel(driver_frame, metrics, cam_label, drowsy_events, visible_sleep, session_start)

        highway.draw(sim_frame, vehicles, adas)

        if now - boot_start < boot_duration + 0.35:
            draw_boot_sequence(sim_frame, BOOT_STEPS, boot_start)

        demo_label = "S=End Sleep Demo" if force_sleep_demo else "S=Simulate Sleep"
        cv2.putText(driver_frame, f"Q=Quit  C=Change Camera  {demo_label}", (14, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (170, 180, 185), 1)
        cv2.imshow(driver_win, driver_frame)
        cv2.imshow(sim_win, sim_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            alarm.stop()
            cap.release()
            cv2.destroyAllWindows()
            return False
        if key == ord("c"):
            alarm.stop()
            cap.release()
            cv2.destroyWindow(driver_win)
            cv2.destroyWindow(sim_win)
            return True
        if key == ord("s"):
            force_sleep_demo = not force_sleep_demo

    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()
    return False


def main():
    detector = FaceDetector()
    try:
        while True:
            result = choose_camera()
            if result[0] is None:
                print("[INFO] Exiting.")
                break
            cam_idx, cam_label, cap = result
            if not run_detection(cam_idx, cam_label, cap, detector):
                break
    finally:
        detector.close()
        print("[INFO] Session ended.")


if __name__ == "__main__":
    main()
