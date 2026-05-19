from dataclasses import dataclass

from config import CRUISE_LANE_U, SHOULDER_U


@dataclass
class PathPlan:
    target_lane_u: float = CRUISE_LANE_U
    maneuver: str = "LANE KEEP"
    confidence: float = 1.0


class PathPlanner:
    def choose_path(self, fusion, fatigue_stage, ego_u):
        def path_is_safe(target_u):
            if abs(target_u - ego_u) < 0.1:
                return True
            step = -0.1 if target_u < ego_u else 0.1
            u = ego_u + step
            while (step > 0 and u <= target_u) or (step < 0 and u >= target_u):
                if fusion.lane_assessments:
                    closest = min(fusion.lane_assessments.keys(), key=lambda l: abs(l - u))
                    if abs(closest - u) < 0.35 and fusion.lane_assessments[closest].safety < 0.3:
                        return False
                u += step
            return True

        if fatigue_stage >= 4:
            shoulder = fusion.lane_assessments.get(SHOULDER_U)
            if shoulder and shoulder.safety > 0.48:
                if path_is_safe(SHOULDER_U):
                    return PathPlan(SHOULDER_U, "SAFE SHOULDER ENTRY", shoulder.safety)
                else:
                    safe_lanes = [l for l, a in fusion.lane_assessments.items() if a.safety > 0.4]
                    if not safe_lanes:
                        safe_lanes = [CRUISE_LANE_U]
                    closest = min(safe_lanes, key=lambda l: abs(l - ego_u))
                    return PathPlan(closest, "WAITING - PATH BLOCKED", 0.5)

        if fusion.collision_risk > 0.42 or fatigue_stage >= 3:
            candidates = sorted(fusion.lane_assessments.values(), key=lambda lane: lane.safety, reverse=True)
            for lane in candidates:
                if lane.lane_u == ego_u:
                    continue
                if lane.safety > 0.52 and path_is_safe(lane.lane_u):
                    label = "PLANNED LEFT ESCAPE" if lane.lane_u < ego_u else "PLANNED RIGHT ESCAPE"
                    return PathPlan(lane.lane_u, label, lane.safety)
            
            safe_lanes = [l for l, a in fusion.lane_assessments.items() if a.safety > 0.4]
            if not safe_lanes:
                safe_lanes = [CRUISE_LANE_U]
            closest = min(safe_lanes, key=lambda l: abs(l - ego_u))
            return PathPlan(closest, "CONTROLLED BRAKING", max(0.2, 1.0 - fusion.collision_risk))

        return PathPlan(CRUISE_LANE_U, "LANE KEEP", 1.0 - fusion.collision_risk)

