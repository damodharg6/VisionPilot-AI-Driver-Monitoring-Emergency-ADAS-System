import numpy as np
from scipy.spatial import distance


def eye_aspect_ratio(pts):
    a = distance.euclidean(pts[1], pts[5])
    b = distance.euclidean(pts[2], pts[4])
    c = distance.euclidean(pts[0], pts[3])
    return (a + b) / (2.0 * c) if c else 0.0


def landmark_points(landmarks, indices, width, height):
    return np.array([(landmarks[i].x * width, landmarks[i].y * height) for i in indices], dtype=np.float32)


def face_bbox(landmarks, width, height, pad=12):
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    return (
        max(0, int(min(xs)) - pad),
        max(0, int(min(ys)) - pad),
        min(width, int(max(xs)) + pad),
        min(height, int(max(ys)) + pad),
    )

