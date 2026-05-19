from dataclasses import dataclass, field
import math

from adas.collision_prediction import assess_collision


@dataclass
class LaneAssessment:
    lane_u: float
    front_gap: float = 999.0
    rear_gap: float = 999.0
    nearest_speed: float = 0.0
    density: int = 0
    risk: float = 0.0
    safety: float = 1.0


@dataclass
class FusionSnapshot:
    lane_assessments: dict = field(default_factory=dict)
    front_threat: object = None
    front_collision: object = None
    collision_risk: float = 0.0
    blind_spot_left: bool = False
    blind_spot_right: bool = False
    obstacle_density: float = 0.0


class SensorFusion:
    def __init__(self, lanes):
        self.lanes = lanes

    def evaluate(self, ego_u, ego_speed_mps, vehicles):
        lane_assessments = {}
        front_threat = None
        front_collision = None
        max_collision_risk = 0.0

        for lane_u in self.lanes + [-1.35]:
            assessment = LaneAssessment(lane_u=lane_u)
            for vehicle in vehicles:
                if abs(vehicle.lane_u - lane_u) > 0.34:
                    continue
                if vehicle.z >= 0:
                    assessment.front_gap = min(assessment.front_gap, vehicle.z)
                else:
                    assessment.rear_gap = min(assessment.rear_gap, abs(vehicle.z))
                if abs(vehicle.z) < 45:
                    assessment.density += 1
                    assessment.nearest_speed = vehicle.speed_mps

            if assessment.front_gap < 18.0 or assessment.rear_gap < 14.0:
                assessment.risk = 1.0
                assessment.safety = 0.0
                lane_assessments[lane_u] = assessment
                continue

            front_risk = max(0.0, min(1.0, (34.0 - assessment.front_gap) / 34.0))
            rear_risk = max(0.0, min(1.0, (16.0 - assessment.rear_gap) / 16.0))
            density_risk = min(1.0, assessment.density / 4.0)
            shoulder_bonus = 0.18 if lane_u < -1.0 and assessment.front_gap > 35 and assessment.rear_gap > 18 else 0.0
            assessment.risk = max(0.0, min(1.0, front_risk * 0.48 + rear_risk * 0.34 + density_risk * 0.18 - shoulder_bonus))
            assessment.safety = 1.0 - assessment.risk
            lane_assessments[lane_u] = assessment

        for vehicle in vehicles:
            if vehicle.z <= 0:
                continue
            lateral = abs(vehicle.lane_u - ego_u)
            if lateral < 0.42:
                collision = assess_collision(ego_speed_mps, vehicle, lateral)
                if front_collision is None or collision.danger > front_collision.danger:
                    front_collision = collision
                    front_threat = vehicle
                max_collision_risk = max(max_collision_risk, collision.danger)

        blind_left = any(-12 < v.z < 12 and v.lane_u < ego_u and abs(v.lane_u - ego_u) < 1.15 for v in vehicles)
        blind_right = any(-12 < v.z < 12 and v.lane_u > ego_u and abs(v.lane_u - ego_u) < 1.15 for v in vehicles)
        density = min(1.0, sum(1 for v in vehicles if -20 < v.z < 70) / 8.0)

        return FusionSnapshot(
            lane_assessments=lane_assessments,
            front_threat=front_threat,
            front_collision=front_collision,
            collision_risk=max_collision_risk,
            blind_spot_left=blind_left,
            blind_spot_right=blind_right,
            obstacle_density=density,
        )
