from dataclasses import dataclass
import math


@dataclass
class CollisionAssessment:
    ttc: float
    braking_distance: float
    distance: float
    relative_speed: float
    danger: float


def assess_collision(ego_speed_mps, vehicle, lateral_offset=0.0):
    distance = max(0.1, vehicle.z)
    relative_speed = max(0.0, ego_speed_mps - vehicle.speed_mps)
    ttc = distance / relative_speed if relative_speed > 0.2 else math.inf
    braking_distance = ego_speed_mps * 1.15 + (ego_speed_mps * ego_speed_mps) / (2.0 * 5.7)

    ttc_risk = 0.0 if math.isinf(ttc) else max(0.0, min(1.0, (5.5 - ttc) / 5.5))
    distance_risk = max(0.0, min(1.0, (braking_distance + 8.0 - distance) / (braking_distance + 8.0)))
    lane_risk = max(0.0, 1.0 - lateral_offset / 0.42)
    danger = max(ttc_risk * 0.62 + distance_risk * 0.38, distance_risk * 0.5) * lane_risk

    return CollisionAssessment(
        ttc=ttc,
        braking_distance=braking_distance,
        distance=distance,
        relative_speed=relative_speed,
        danger=max(0.0, min(1.0, danger)),
    )
