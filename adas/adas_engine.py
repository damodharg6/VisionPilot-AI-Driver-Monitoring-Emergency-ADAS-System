from dataclasses import dataclass

from adas.autonomous_parking import AutonomousParking
from adas.path_planner import PathPlanner
from adas.sensor_fusion import SensorFusion
from adas.vehicle_physics import VehiclePhysics
from config import CRUISE_LANE_U, LANES, SHOULDER_U


@dataclass
class AdasTelemetry:
    mode: str = "CRUISING"
    stage: int = 0
    maneuver: str = "LANE KEEP"
    target_lane_u: float = CRUISE_LANE_U
    speed_kmh: float = 100.0
    lane_u: float = CRUISE_LANE_U
    steering: float = 0.0
    collision_risk: float = 0.0
    ttc: float = 99.0
    hazard_lights: bool = False
    secured: bool = False
    plan_confidence: float = 1.0
    sensor_status: str = "FUSED"
    emergency_phase: str = "NORMAL MONITORING"
    driver_condition: str = "ATTENTIVE"
    shoulder_distance: float = 0.0


class AdasDecisionEngine:
    def __init__(self):
        self.physics = VehiclePhysics(speed_kmh=100.0, lane_u=CRUISE_LANE_U)
        self.fusion = SensorFusion(LANES)
        self.planner = PathPlanner()
        self.parking = AutonomousParking()
        self.takeover_timer = 0.0

    def update(self, dt, driver_metrics, vehicles):
        stage = self._escalation_stage(driver_metrics)
        if stage >= 3:
            self.takeover_timer += dt
        else:
            self.takeover_timer = 0.0

        fusion = self.fusion.evaluate(self.physics.lane_u, self.physics.speed_mps, vehicles)
        plan = self.planner.choose_path(fusion, stage, self.physics.lane_u)
        near_shoulder = abs(self.physics.lane_u - SHOULDER_U) < 0.08
        target_speed = self.parking.target_speed(stage, fusion.collision_risk, near_shoulder)

        if fusion.front_threat is not None and stage < 4:
            gap = fusion.front_threat.z
            if gap < 65.0:
                threat_speed_kmh = fusion.front_threat.speed_mps * 3.6
                if gap < 25.0:
                    target_speed = min(target_speed, threat_speed_kmh - 20.0)
                elif gap < 50.0:
                    target_speed = min(target_speed, threat_speed_kmh)

        if plan.maneuver == "CONTROLLED BRAKING" and fusion.front_threat is not None:
            target_speed = min(target_speed, fusion.front_threat.speed_mps * 3.6 - 15.0)

        secured = self.parking.update_secured(self.physics.speed_kmh, near_shoulder, stage)
        if secured:
            target_speed = 0.0
            plan.target_lane_u = SHOULDER_U
            plan.maneuver = "VEHICLE SECURED"

        speed, lane_u, steering = self.physics.update(dt, target_speed, plan.target_lane_u)
        mode = self._mode(stage, secured, fusion.collision_risk)
        front_ttc = fusion.front_collision.ttc if fusion.front_collision else 99.0
        shoulder_distance = abs(lane_u - SHOULDER_U)
        emergency_phase = self._emergency_phase(stage, secured, shoulder_distance, speed, fusion.collision_risk)

        return AdasTelemetry(
            mode=mode,
            stage=stage,
            maneuver=plan.maneuver,
            target_lane_u=plan.target_lane_u,
            speed_kmh=speed,
            lane_u=lane_u,
            steering=steering,
            collision_risk=fusion.collision_risk,
            ttc=front_ttc if front_ttc != float("inf") else 99.0,
            hazard_lights=stage >= 3,
            secured=secured,
            plan_confidence=plan.confidence,
            sensor_status="FUSED",
            emergency_phase=emergency_phase,
            driver_condition=self._driver_condition(stage),
            shoulder_distance=shoulder_distance,
        )

    @staticmethod
    def _escalation_stage(metrics):
        if not metrics.face_detected:
            return 2 if metrics.fatigue_score > 55 else 1
        if metrics.closure_duration > 3.0 or metrics.fatigue_score > 82:
            return 4
        if metrics.closure_duration > 1.8 or metrics.fatigue_score > 66:
            return 3
        if metrics.closure_duration > 0.9 or metrics.fatigue_score > 42:
            return 2
        if metrics.fatigue_score > 24 or metrics.attention_score < 76:
            return 1
        return 0

    @staticmethod
    def _mode(stage, secured, collision_risk):
        if secured:
            return "SECURED"
        if stage >= 4:
            return "SAFE STOP"
        if stage >= 3:
            return "ADAS TAKEOVER"
        if collision_risk > 0.62:
            return "COLLISION AVOID"
        if stage == 2:
            return "WARNING"
        if stage == 1:
            return "CAUTION"
        return "CRUISING"

    @staticmethod
    def _driver_condition(stage):
        if stage >= 4:
            return "DRIVER ASLEEP"
        if stage == 3:
            return "NO RESPONSE"
        if stage == 2:
            return "EYES CLOSED"
        if stage == 1:
            return "FATIGUE RISK"
        return "ATTENTIVE"

    @staticmethod
    def _emergency_phase(stage, secured, shoulder_distance, speed_kmh, collision_risk):
        if secured:
            return "VEHICLE SECURED ON SHOULDER"
        if stage >= 4:
            if shoulder_distance > 0.55:
                return "AUTONOMOUS STEERING TO SHOULDER"
            if speed_kmh > 6.0:
                return "CONTROLLED DECELERATION"
            return "FINAL PARKING ALIGNMENT"
        if stage == 3:
            if collision_risk > 0.45:
                return "RADAR THREAT SCAN ACTIVE"
            return "SAFE SHOULDER ROUTE SELECTED"
        if stage == 2:
            return "AUDIBLE WARNING - DRIVER RESPONSE REQUIRED"
        if stage == 1:
            return "SOFT FATIGUE WARNING"
        return "NORMAL MONITORING"
