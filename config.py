import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

FRAME_WIDTH = 700
EAR_THRESH = 0.25
FRAME_CHECK = 20

BEEP_FREQ = 1000
BEEP_DURATION = 400
BEEP_INTERVAL = 0.5

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14, 78, 308]
HEAD_TILT_POINTS = [33, 263]

CAM_LABELS = {
    0: "Laptop Webcam",
    1: "DroidCam USB (Phone)",
    2: "DroidCam USB (alt)",
    3: "External Cam",
    4: "External Cam 2",
}

BOOT_STEPS = [
    ("INITIALIZING DRIVER MONITOR", 0.45),
    ("CALIBRATING FACE LANDMARK MODEL", 0.45),
    ("CALIBRATING RADARS", 0.45),
    ("TRAFFIC AI ONLINE", 0.45),
    ("ADAS DECISION ENGINE READY", 0.45),
    ("SYSTEM ACTIVE", 0.55),
]

LANES = [-0.48, 0.48]
SHOULDER_U = -1.35
CRUISE_LANE_U = 0.48

