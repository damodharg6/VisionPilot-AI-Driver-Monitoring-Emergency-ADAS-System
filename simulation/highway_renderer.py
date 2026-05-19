import cv2
import numpy as np

from simulation.cinematic_effects import draw_rounded_rect


class HighwayRenderer:
    def __init__(self):
        self.sim_distance = 0.0
        self.sweep = 0.0

    def draw(self, frame, vehicles, adas):
        h, w = frame.shape[:2]
        self.sim_distance += adas.speed_kmh * 0.0135
        self.sweep += 0.045

        self._background(frame)
        view = self._view_bounds(w, h)

        def project(u, z):
            return self._project(view, u, z)

        self._draw_world(frame, project)
        self._draw_shoulder_parking_zone(frame, project, adas)
        self._draw_radar_cone(frame, project, adas)
        self._draw_planned_path(frame, project, adas)
        self._draw_lane_intelligence(frame, project, adas)

        for vehicle in sorted(vehicles, key=lambda v: v.z, reverse=True):
            self._draw_vehicle(frame, project, vehicle, adas)

        self._draw_ego_vehicle(frame, project, adas)
        self._draw_sensor_inset(frame, vehicles, adas)
        self._draw_emergency_protocol_panel(frame, adas)
        self._draw_top_status(frame, adas)
        self._draw_telemetry_bar(frame, adas)

    @staticmethod
    def _view_bounds(w, h):
        return {
            "left": int(w * 0.035),
            "right": int(w * 0.965),
            "top": int(h * 0.085),
            "bottom": int(h * 0.825),
            "horizon": int(h * 0.115),
            "cx": int(w * 0.50),
        }

    def _project(self, view, u, z):
        z = max(1.0, min(175.0, z))
        depth = np.power(1.0 - z / 175.0, 2.10)
        y = int(view["top"] + depth * (view["bottom"] - view["top"]))
        road_width = 44.0 + depth * ((view["right"] - view["left"]) * 0.74)
        bend = 0.11 * np.sin((z * 0.020) + self.sim_distance * 0.010)
        x = int(view["cx"] + (u + bend) * road_width * 0.38)
        return x, y

    def _background(self, frame):
        h, w = frame.shape[:2]
        for y in range(h):
            t = y / max(1, h - 1)
            if t < 0.50:
                k = t / 0.50
                color = (int(24 + 22 * k), int(15 + 16 * k), int(8 + 8 * k))
            else:
                k = (t - 0.50) / 0.50
                color = (int(30 + 20 * k), int(24 + 18 * k), int(18 + 12 * k))
            cv2.line(frame, (0, y), (w, y), color, 1)

        glow = frame.copy()
        cv2.ellipse(glow, (w // 2, int(h * 0.18)), (int(w * 0.44), int(h * 0.16)), 0, 0, 360, (70, 48, 22), -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.23, frame, 0.77, 0, frame)
        for x in range(0, w, 72):
            cv2.line(frame, (x, int(h * 0.52)), (x + int(w * 0.18), h), (20, 28, 34), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (16, 10, 5), 8, cv2.LINE_AA)

    def _draw_world(self, frame, project):
        road = np.array([project(-1.42, 175), project(1.42, 175), project(1.42, 2), project(-1.42, 2)], np.int32)
        left_shoulder = np.array([project(-2.22, 175), project(-1.42, 175), project(-1.42, 2), project(-2.22, 2)], np.int32)
        right_shoulder = np.array([project(1.42, 175), project(2.16, 175), project(2.16, 2), project(1.42, 2)], np.int32)

        cv2.fillPoly(frame, [left_shoulder], (42, 48, 38), cv2.LINE_AA)
        cv2.fillPoly(frame, [right_shoulder], (38, 35, 30), cv2.LINE_AA)
        cv2.fillPoly(frame, [road], (46, 43, 38), cv2.LINE_AA)

        overlay = frame.copy()
        road_inner = np.array([project(-0.95, 175), project(0.95, 175), project(0.95, 2), project(-0.95, 2)], np.int32)
        cv2.fillPoly(overlay, [road_inner], (58, 53, 45), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.32, frame, 0.68, 0, frame)

        for z in np.linspace(10, 174, 15):
            shade = int(42 + 20 * (1.0 - z / 175.0))
            cv2.line(frame, project(-1.42, z), project(1.42, z), (shade + 8, shade + 3, shade), 1, cv2.LINE_AA)

        z_vals = np.linspace(175, 2, 34)
        lane_specs = [
            (-1.42, (138, 138, 112), 2),
            (-0.95, (235, 244, 242), 3),
            (0.95, (235, 244, 242), 3),
            (1.42, (138, 138, 112), 2),
        ]
        for u, color, thickness in lane_specs:
            points = np.array([project(u, z) for z in z_vals], np.int32)
            cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)

        offset = self.sim_distance % 24.0
        for i in range(14):
            z1 = i * 24.0 - offset
            z2 = z1 + 10.0
            if z2 > 2:
                z1 = max(2.0, min(175.0, z1))
                z2 = max(2.0, min(175.0, z2))
                thickness = max(2, int(7 * np.power(1.0 - z1 / 175.0, 1.35)))
                cv2.line(frame, project(0.0, z1), project(0.0, z2), (245, 248, 246), thickness, cv2.LINE_AA)

        self._draw_guardrails(frame, project)

    def _draw_guardrails(self, frame, project):
        for side in [-1, 1]:
            rail_u = side * 1.72
            rail_points = np.array([project(rail_u, z) for z in np.linspace(175, 4, 28)], np.int32)
            cv2.polylines(frame, [rail_points], False, (125, 116, 86), 2, cv2.LINE_AA)
            for z in np.arange(12, 170, 18):
                p = project(rail_u, z)
                q = project(rail_u + side * 0.08, z + 1.8)
                cv2.line(frame, p, q, (112, 105, 80), 2, cv2.LINE_AA)

    def _draw_planned_path(self, frame, project, adas):
        points = []
        for z in np.linspace(7, 136, 34):
            blend = min(1.0, z / 92.0)
            smooth = blend * blend * (3.0 - 2.0 * blend)
            lane = adas.lane_u + (adas.target_lane_u - adas.lane_u) * smooth
            points.append(project(lane, z))

        pts = np.array(points, np.int32)
        color = (160, 220, 0) if adas.stage < 3 else (0, 170, 255)
        glow = frame.copy()
        cv2.polylines(glow, [pts], False, color, 24, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.12, frame, 0.88, 0, frame)
        cv2.polylines(frame, [pts], False, color, 4, cv2.LINE_AA)
        for i, p in enumerate(points[::4]):
            pulse = 3 + (i % 3)
            cv2.circle(frame, p, pulse, color, -1, cv2.LINE_AA)

    def _draw_shoulder_parking_zone(self, frame, project, adas):
        if adas.stage < 3:
            return
        overlay = frame.copy()
        left = [project(-1.62, z) for z in np.linspace(14, 72, 14)]
        right = [project(-1.10, z) for z in np.linspace(72, 14, 14)]
        bay = np.array(left + right, np.int32)
        color = (0, 170, 255)
        cv2.fillPoly(overlay, [bay], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)

        for z in [16, 30, 44, 58, 72]:
            p1 = project(-1.60, z)
            p2 = project(-1.12, z)
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)

        label_pt = project(-1.42, 78)
        cv2.putText(frame, "SAFE SHOULDER PARKING ZONE", (label_pt[0] - 82, label_pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    def _draw_radar_cone(self, frame, project, adas):
        overlay = frame.copy()
        sweep = 0.34 * np.sin(self.sweep)
        color = (0, 120, 255) if adas.stage >= 3 else (235, 220, 0)
        
        angle = adas.steering * 0.38
        if adas.stage >= 4 and adas.target_lane_u == -1.35:
            progress = max(0.0, min(1.0, (-0.8 - adas.lane_u) / 0.55))
            if progress > 0:
                angle = adas.steering * 0.38 * (1.0 - progress) - 0.22 * progress
                
        cx, cy = project(adas.lane_u, 6.3)
        c_a, s_a = np.cos(angle), np.sin(angle)
        def rot(pts):
            if angle == 0.0: return pts
            return [(int(cx + (x-cx)*c_a - (y-cy)*s_a), int(cy + (x-cx)*s_a + (y-cy)*c_a)) for x, y in pts]

        cone_pts = [
            project(adas.lane_u - 0.18, 8),
            project(adas.lane_u + 0.18, 8),
            project(adas.lane_u + 1.05 + sweep, 128),
            project(adas.lane_u - 1.05 + sweep, 128),
        ]
        cone = np.array(rot(cone_pts), np.int32)
        cv2.fillPoly(overlay, [cone], color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.13 if adas.stage < 2 else 0.22, frame, 0.87 if adas.stage < 2 else 0.78, 0, frame)
        for z in [28, 54, 82, 112, 140]:
            p1, p2 = rot([project(adas.lane_u - 0.48 + sweep * 0.3, z), project(adas.lane_u + 0.48 + sweep * 0.3, z)])
            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

    def _draw_lane_intelligence(self, frame, project, adas):
        if adas.stage < 2 and adas.collision_risk < 0.25:
            return
        overlay = frame.copy()
        lanes = [(-0.48, "L1"), (0.48, "L2"), (-1.35, "SHOULDER")]
        for lane_u, label in lanes:
            status = "TARGET" if abs(lane_u - adas.target_lane_u) < 0.08 else "SCAN"
            color = (110, 220, 0) if status == "TARGET" else (120, 95, 40)
            points_l = [project(lane_u - 0.22, z) for z in np.linspace(16, 116, 16)]
            points_r = [project(lane_u + 0.22, z) for z in np.linspace(116, 16, 16)]
            poly = np.array(points_l + points_r, np.int32)
            cv2.fillPoly(overlay, [poly], color, cv2.LINE_AA)
            p = project(lane_u, 118)
            cv2.putText(frame, f"{label} {status}", (p[0] - 38, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    def _draw_vehicle(self, frame, project, vehicle, adas):
        if vehicle.z < 3 or vehicle.z > 168:
            return

        cx, cy = project(vehicle.lane_u, vehicle.z)
        scale = np.power(1.0 - vehicle.z / 175.0, 1.58)
        base_w = {"motorcycle": 20, "truck": 54, "suv": 48, "sedan": 42}.get(vehicle.vehicle_type, 42)
        base_h = {"motorcycle": 34, "truck": 74, "suv": 50, "sedan": 44}.get(vehicle.vehicle_type, 44)
        w = max(10, int(base_w * scale))
        h = max(14, int(base_h * scale))

        angle = 0.0
        if getattr(vehicle, 'target_lane_u', None) is not None:
            angle = max(-0.28, min(0.28, (vehicle.target_lane_u - vehicle.lane_u) * 0.65))
        angle_deg = int(np.degrees(angle))

        def rot(pts):
            if angle == 0.0: return np.array(pts, np.int32)
            c, s = np.cos(angle), np.sin(angle)
            return np.array([(int(cx + (x-cx)*c - (y-cy)*s), int(cy + (x-cx)*s + (y-cy)*c)) for x, y in pts], np.int32)
            
        def pRot(x, y):
            if angle == 0.0: return (int(x), int(y))
            c, s = np.cos(angle), np.sin(angle)
            return (int(cx + (x-cx)*c - (y-cy)*s), int(cy + (x-cx)*s + (y-cy)*c))

        cv2.ellipse(frame, (cx, cy + max(3, h // 9)), (max(8, int(w * 0.76)), max(3, h // 8)), angle_deg, 0, 360, (2, 4, 7), -1, cv2.LINE_AA)
        
        if vehicle.vehicle_type == "motorcycle":
            self._draw_motorcycle(frame, cx, cy, w, h, vehicle.color, angle)
            self._draw_track_box(frame, cx, cy, w, h, vehicle, adas)
            return
        if vehicle.vehicle_type == "truck":
            self._draw_truck(frame, cx, cy, w, h, vehicle.color, angle)
            self._draw_track_box(frame, cx, cy, w, h, vehicle, adas)
            return

        body = rot([
            (cx - w // 2, cy),
            (cx - int(w * 0.45), cy - int(h * 0.66)),
            (cx - int(w * 0.24), cy - h),
            (cx + int(w * 0.24), cy - h),
            (cx + int(w * 0.45), cy - int(h * 0.66)),
            (cx + w // 2, cy),
        ])
        highlight = tuple(min(255, int(c * 1.22 + 18)) for c in vehicle.color)
        cv2.fillPoly(frame, [body], vehicle.color, cv2.LINE_AA)
        roof = rot([
            (cx - int(w * 0.26), cy - int(h * 0.82)),
            (cx + int(w * 0.26), cy - int(h * 0.82)),
            (cx + int(w * 0.34), cy - int(h * 0.48)),
            (cx - int(w * 0.34), cy - int(h * 0.48)),
        ])
        cv2.fillPoly(frame, [roof], highlight, cv2.LINE_AA)
        glass = rot([
            (cx - int(w * 0.20), cy - int(h * 0.76)),
            (cx + int(w * 0.20), cy - int(h * 0.76)),
            (cx + int(w * 0.27), cy - int(h * 0.52)),
            (cx - int(w * 0.27), cy - int(h * 0.52)),
        ])
        cv2.fillPoly(frame, [glass], (14, 27, 34), cv2.LINE_AA)
        cv2.polylines(frame, [body], True, (10, 15, 20), 1, cv2.LINE_AA)
        
        l1, l2 = pRot(cx, cy - int(h * 0.76)), pRot(cx, cy - int(h * 0.52))
        cv2.line(frame, l1, l2, (58, 84, 94), 1, cv2.LINE_AA)
        
        light_w = max(3, w // 5)
        tl_l = rot([(cx - w // 2, cy - max(4, h // 9)), (cx - w // 2 + light_w, cy - max(4, h // 9)), 
                    (cx - w // 2 + light_w, cy), (cx - w // 2, cy)])
        tl_r = rot([(cx + w // 2 - light_w, cy - max(4, h // 9)), (cx + w // 2, cy - max(4, h // 9)), 
                    (cx + w // 2, cy), (cx + w // 2 - light_w, cy)])
        cv2.fillPoly(frame, [tl_l], (0, 0, 245), cv2.LINE_AA)
        cv2.fillPoly(frame, [tl_r], (0, 0, 245), cv2.LINE_AA)
        self._draw_track_box(frame, cx, cy, w, h, vehicle, adas)

    @staticmethod
    def _draw_truck(frame, cx, cy, w, h, color, angle=0.0):
        def rot(pts):
            if angle == 0.0: return np.array(pts, np.int32)
            c, s = np.cos(angle), np.sin(angle)
            return np.array([(int(cx + (x-cx)*c - (y-cy)*s), int(cy + (x-cx)*s + (y-cy)*c)) for x, y in pts], np.int32)
        def pRot(x, y):
            if angle == 0.0: return (int(x), int(y))
            c, s = np.cos(angle), np.sin(angle)
            return (int(cx + (x-cx)*c - (y-cy)*s), int(cy + (x-cx)*s + (y-cy)*c))

        trailer_h = int(h * 0.74)
        cab_h = int(h * 0.48)
        trailer_w = max(12, int(w * 0.78))
        cab_w = max(10, int(w * 0.56))
        
        trailer = rot([
            (cx - trailer_w // 2, cy - int(h * 0.22)),
            (cx - trailer_w // 2, cy - trailer_h),
            (cx + trailer_w // 2, cy - trailer_h),
            (cx + trailer_w // 2, cy - int(h * 0.22)),
        ])
        cab = rot([
            (cx - cab_w // 2, cy),
            (cx - int(cab_w * 0.42), cy - cab_h),
            (cx + int(cab_w * 0.42), cy - cab_h),
            (cx + cab_w // 2, cy),
        ])
        cv2.fillPoly(frame, [trailer], tuple(max(0, int(c * 0.85)) for c in color), cv2.LINE_AA)
        cv2.fillPoly(frame, [cab], color, cv2.LINE_AA)
        cv2.polylines(frame, [trailer], True, (10, 15, 20), 1, cv2.LINE_AA)
        cv2.polylines(frame, [cab], True, (10, 15, 20), 1, cv2.LINE_AA)
        
        cab_glass = rot([
            (cx - int(cab_w * 0.25), cy - int(cab_h * 0.75)),
            (cx + int(cab_w * 0.25), cy - int(cab_h * 0.75)),
            (cx + int(cab_w * 0.25), cy - int(cab_h * 0.36)),
            (cx - int(cab_w * 0.25), cy - int(cab_h * 0.36)),
        ])
        cv2.fillPoly(frame, [cab_glass], (16, 30, 36), cv2.LINE_AA)
        
        for side in [-1, 1]:
            w1 = pRot(cx + side * int(w * 0.36), cy - 2)
            cv2.circle(frame, w1, max(2, w // 11), (5, 6, 8), -1, cv2.LINE_AA)
            w2 = pRot(cx + side * int(w * 0.25), cy - int(h * 0.25))
            cv2.circle(frame, w2, max(2, w // 13), (5, 6, 8), -1, cv2.LINE_AA)
            
        light_w = max(3, w // 6)
        tl_l = rot([(cx - cab_w // 2, cy - max(4, h // 11)), (cx - cab_w // 2 + light_w, cy - max(4, h // 11)), 
                    (cx - cab_w // 2 + light_w, cy), (cx - cab_w // 2, cy)])
        tl_r = rot([(cx + cab_w // 2 - light_w, cy - max(4, h // 11)), (cx + cab_w // 2, cy - max(4, h // 11)), 
                    (cx + cab_w // 2, cy), (cx + cab_w // 2 - light_w, cy)])
        cv2.fillPoly(frame, [tl_l], (0, 0, 245), cv2.LINE_AA)
        cv2.fillPoly(frame, [tl_r], (0, 0, 245), cv2.LINE_AA)

    def _draw_track_box(self, frame, cx, cy, w, h, vehicle, adas):
        threat = abs(vehicle.lane_u - adas.lane_u) < 0.42 and 0 < vehicle.z < 80
        color = (0, 95, 255) if threat else (180, 220, 40)
        x1, y1, x2, y2 = cx - w // 2 - 6, cy - h - 7, cx + w // 2 + 6, cy + 7
        bl = max(5, min(14, w // 3))
        for sx, sy, dx, dy in [
            (x1, y1, bl, bl), (x2, y1, -bl, bl),
            (x1, y2, bl, -bl), (x2, y2, -bl, -bl),
        ]:
            cv2.line(frame, (sx, sy), (sx + dx, sy), color, 1, cv2.LINE_AA)
            cv2.line(frame, (sx, sy), (sx, sy + dy), color, 1, cv2.LINE_AA)
        if vehicle.z < 95:
            tag = f"{vehicle.vehicle_type.upper()} {int(vehicle.z)}m"
            cv2.putText(frame, tag, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_motorcycle(frame, cx, cy, w, h, color, angle=0.0):
        def pRot(x, y):
            if angle == 0.0: return (int(x), int(y))
            c, s = np.cos(angle), np.sin(angle)
            return (int(cx + (x-cx)*c - (y-cy)*s), int(cy + (x-cx)*s + (y-cy)*c))
            
        wheel_r = max(3, w // 5)
        rear = pRot(cx - int(w * 0.42), cy)
        front = pRot(cx + int(w * 0.42), cy)
        body_mid = pRot(cx, cy - int(h * 0.44))
        handle = pRot(cx + int(w * 0.34), cy - int(h * 0.70))
        rider = pRot(cx - int(w * 0.06), cy - int(h * 0.86))
        
        cv2.circle(frame, rear, wheel_r, (5, 6, 8), -1, cv2.LINE_AA)
        cv2.circle(frame, front, wheel_r, (5, 6, 8), -1, cv2.LINE_AA)
        cv2.line(frame, rear, body_mid, color, max(2, w // 7), cv2.LINE_AA)
        cv2.line(frame, body_mid, front, color, max(2, w // 7), cv2.LINE_AA)
        cv2.line(frame, body_mid, handle, (210, 220, 220), max(1, w // 10), cv2.LINE_AA)
        cv2.ellipse(frame, body_mid, (max(4, w // 3), max(2, h // 8)), int(np.degrees(angle)), 0, 360, color, -1, cv2.LINE_AA)
        cv2.circle(frame, rider, max(3, w // 4), (35, 42, 52), -1, cv2.LINE_AA)
        cv2.line(frame, rider, body_mid, (42, 52, 62), max(2, w // 8), cv2.LINE_AA)
        
        tl = np.array([
            pRot(cx - w // 2, cy - 4), pRot(cx + w // 2, cy - 4),
            pRot(cx + w // 2, cy), pRot(cx - w // 2, cy)
        ], np.int32)
        cv2.fillPoly(frame, [tl], (0, 0, 230), cv2.LINE_AA)

    def _draw_ego_vehicle(self, frame, project, adas):
        cx, cy = project(adas.lane_u, 6.3)
        body_color = (245, 120, 0)
        if adas.stage >= 3:
            body_color = (0, 112, 255)
        if adas.secured:
            body_color = (0, 220, 130)
            
        angle = adas.steering * 0.28
        if adas.stage >= 4 and adas.target_lane_u == -1.35:
            progress = max(0.0, min(1.0, (-0.8 - adas.lane_u) / 0.55))
            if progress > 0:
                angle = adas.steering * 0.28 * (1.0 - progress) - 0.22 * progress
                
        angle_deg = int(np.degrees(angle))

        def rot(pts):
            if angle == 0.0: return np.array(pts, np.int32)
            c, s = np.cos(angle), np.sin(angle)
            return np.array([(int(cx + (x-cx)*c - (y-cy)*s), int(cy + (x-cx)*s + (y-cy)*c)) for x, y in pts], np.int32)
            
        def pRot(x, y):
            if angle == 0.0: return (int(x), int(y))
            c, s = np.cos(angle), np.sin(angle)
            return (int(cx + (x-cx)*c - (y-cy)*s), int(cy + (x-cx)*s + (y-cy)*c))

        w, h = 44, 48

        glow = frame.copy()
        g_cx, g_cy = pRot(cx, cy + 6)
        cv2.ellipse(glow, (g_cx, g_cy), (w + 14, int(h * 0.3)), angle_deg, 0, 360, (255, 210, 0), -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.10 if adas.stage >= 2 else 0.05, frame, 0.90 if adas.stage >= 2 else 0.95, 0, frame)
        
        s_cx, s_cy = pRot(cx, cy + max(3, h // 9))
        cv2.ellipse(frame, (s_cx, s_cy), (max(8, int(w * 0.76)), max(3, h // 8)), angle_deg, 0, 360, (2, 4, 8), -1, cv2.LINE_AA)

        body = rot([
            (cx - w // 2, cy),
            (cx - int(w * 0.45), cy - int(h * 0.66)),
            (cx - int(w * 0.24), cy - h),
            (cx + int(w * 0.24), cy - h),
            (cx + int(w * 0.45), cy - int(h * 0.66)),
            (cx + w // 2, cy),
        ])
        highlight = tuple(min(255, int(c * 1.22 + 18)) for c in body_color)
        cv2.fillPoly(frame, [body], body_color, cv2.LINE_AA)
        
        roof = rot([
            (cx - int(w * 0.26), cy - int(h * 0.82)),
            (cx + int(w * 0.26), cy - int(h * 0.82)),
            (cx + int(w * 0.34), cy - int(h * 0.48)),
            (cx - int(w * 0.34), cy - int(h * 0.48)),
        ])
        cv2.fillPoly(frame, [roof], highlight, cv2.LINE_AA)
        
        glass = rot([
            (cx - int(w * 0.20), cy - int(h * 0.76)),
            (cx + int(w * 0.20), cy - int(h * 0.76)),
            (cx + int(w * 0.27), cy - int(h * 0.52)),
            (cx - int(w * 0.27), cy - int(h * 0.52)),
        ])
        cv2.fillPoly(frame, [glass], (14, 27, 34), cv2.LINE_AA)
        cv2.polylines(frame, [body], True, (10, 15, 20), 1, cv2.LINE_AA)
        
        l1, l2 = pRot(cx, cy - int(h * 0.76)), pRot(cx, cy - int(h * 0.52))
        cv2.line(frame, l1, l2, (58, 84, 94), 1, cv2.LINE_AA)
        
        light_w = max(3, w // 5)
        tl_l = rot([(cx - w // 2, cy - max(4, h // 9)), (cx - w // 2 + light_w, cy - max(4, h // 9)), 
                    (cx - w // 2 + light_w, cy), (cx - w // 2, cy)])
        tl_r = rot([(cx + w // 2 - light_w, cy - max(4, h // 9)), (cx + w // 2, cy - max(4, h // 9)), 
                    (cx + w // 2, cy), (cx + w // 2 - light_w, cy)])
        cv2.fillPoly(frame, [tl_l], (250, 255, 255), cv2.LINE_AA)
        cv2.fillPoly(frame, [tl_r], (250, 255, 255), cv2.LINE_AA)
        
        if adas.hazard_lights and int(self.sim_distance * 0.48) % 2 == 0:
            h1 = pRot(cx - int(w * 0.55), cy - int(h * 0.25))
            h2 = pRot(cx + int(w * 0.55), cy - int(h * 0.25))
            cv2.circle(frame, h1, 5, (0, 185, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, h2, 5, (0, 185, 255), -1, cv2.LINE_AA)

    def _draw_sensor_inset(self, frame, vehicles, adas):
        h, w = frame.shape[:2]
        size = 184
        x1, y1 = w - size - 34, int(h * 0.17)
        x2, y2 = x1 + size, y1 + size
        draw_rounded_rect(frame, (x1, y1), (x2, y2), (8, 13, 18), radius=8, alpha=0.78)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 + 18
        for r in [32, 58, 84]:
            cv2.circle(frame, (cx, cy), r, (38, 72, 78), 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 88), (cx, cy + 88), (38, 72, 78), 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 88, cy), (cx + 88, cy), (38, 72, 78), 1, cv2.LINE_AA)
        cv2.putText(frame, "SENSOR FUSION", (x1 + 16, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 225, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, (245, 188, 58), -1, cv2.LINE_AA)
        for vehicle in vehicles:
            if -36 < vehicle.z < 104:
                px = int(cx + vehicle.lane_u * 38)
                py = int(cy - vehicle.z * 0.72)
                if x1 + 10 < px < x2 - 10 and y1 + 42 < py < y2 - 10:
                    clr = (0, 90, 255) if abs(vehicle.lane_u - adas.lane_u) < 0.42 and vehicle.z > 0 else (0, 220, 170)
                    cv2.circle(frame, (px, py), 4, clr, -1, cv2.LINE_AA)

    def _draw_emergency_protocol_panel(self, frame, adas):
        if adas.stage < 2:
            return
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = 38, int(h * 0.17), 318, int(h * 0.17) + 214
        draw_rounded_rect(frame, (x1, y1), (x2, y2), (6, 10, 16), radius=8, alpha=0.82)
        border = (0, 125, 255) if adas.stage >= 3 else (0, 190, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
        cv2.putText(frame, "EMERGENCY SAFE-STOP", (x1 + 16, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 248, 248), 1, cv2.LINE_AA)
        cv2.putText(frame, adas.driver_condition, (x1 + 16, y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 125, 255), 1, cv2.LINE_AA)

        steps = [
            ("Driver asleep confirmed", adas.stage >= 4),
            ("Hazard lights activated", adas.hazard_lights),
            ("360 radar scan active", adas.stage >= 3),
            ("Safe shoulder selected", adas.target_lane_u < -1.0),
            ("Autonomous steering", adas.stage >= 4 and adas.shoulder_distance > 0.10),
            ("Controlled braking", adas.stage >= 4 and adas.speed_kmh < 55.0),
            ("Vehicle secured", adas.secured),
        ]
        y = y1 + 86
        for text, done in steps:
            color = (0, 220, 140) if done else (88, 110, 122)
            marker = "OK" if done else "--"
            cv2.putText(frame, marker, (x1 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            cv2.putText(frame, text, (x1 + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
            y += 20

        cv2.putText(frame, adas.emergency_phase, (x1 + 16, y2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 210, 255), 1, cv2.LINE_AA)

    def _draw_top_status(self, frame, adas):
        h, w = frame.shape[:2]
        cv2.putText(frame, "ADAS VIRTUAL SIMULATION", (38, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (235, 248, 248), 2, cv2.LINE_AA)
        mode_color = (0, 220, 150) if adas.stage < 2 else ((0, 190, 255) if adas.stage < 3 else (0, 110, 255))
        cv2.putText(frame, adas.mode, (38, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 1, cv2.LINE_AA)
        cv2.putText(frame, adas.maneuver, (w - 520, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 224, 0), 1, cv2.LINE_AA)
        risk_w = 330
        x, y = w - risk_w - 42, 58
        cv2.rectangle(frame, (x, y), (x + risk_w, y + 8), (40, 48, 58), -1, cv2.LINE_AA)
        risk_color = (0, 210, 145) if adas.collision_risk < 0.35 else ((0, 185, 255) if adas.collision_risk < 0.65 else (0, 80, 255))
        cv2.rectangle(frame, (x, y), (x + int(risk_w * min(1.0, adas.collision_risk)), y + 8), risk_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"COLLISION RISK {adas.collision_risk * 100:03.0f}%", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (142, 158, 168), 1, cv2.LINE_AA)

    def _draw_telemetry_bar(self, frame, adas):
        h, w = frame.shape[:2]
        y1, y2 = h - 104, h - 24
        draw_rounded_rect(frame, (34, y1), (w - 34, y2), (7, 11, 16), radius=8, alpha=0.91)
        cv2.rectangle(frame, (34, y1), (w - 34, y2), (44, 62, 72), 1, cv2.LINE_AA)
        phase = adas.emergency_phase
        phase_short = {
            "AUTONOMOUS STEERING TO SHOULDER": "STEERING TO SHOULDER",
            "CONTROLLED DECELERATION": "BRAKING TO STOP",
            "VEHICLE SECURED ON SHOULDER": "SECURED",
            "FINAL PARKING ALIGNMENT": "FINAL ALIGNMENT",
            "RADAR THREAT SCAN ACTIVE": "RADAR SCAN",
            "SAFE SHOULDER ROUTE SELECTED": "ROUTE SELECTED",
            "AUDIBLE WARNING - DRIVER RESPONSE REQUIRED": "WARNING DRIVER",
            "SOFT FATIGUE WARNING": "FATIGUE WARNING",
        }.get(phase, phase[:18])
        items = [
            ("SPEED", f"{adas.speed_kmh:05.1f} km/h"),
            ("MODE", adas.mode),
            ("TTC", f"{adas.ttc:04.1f}s"),
            ("PATH", adas.maneuver),
            ("PHASE", phase_short),
            ("CONF", f"{adas.plan_confidence * 100:03.0f}%"),
            ("STAGE", str(adas.stage)),
        ]
        slots = [168, 156, 136, 282, 230, 104, 76]
        x = 58
        for (label, value), slot in zip(items, slots):
            cv2.putText(frame, label, (x, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 138, 148), 1, cv2.LINE_AA)
            cv2.putText(frame, value, (x, y1 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (232, 242, 242), 1, cv2.LINE_AA)
            x += slot
