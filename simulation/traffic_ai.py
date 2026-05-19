from dataclasses import dataclass
import random

from config import LANES


@dataclass
class TrafficVehicle:
    lane_u: float
    z: float
    speed_mps: float
    vehicle_type: str
    color: tuple
    length: float
    desired_speed_mps: float
    lane_change_timer: float = 0.0
    target_lane_u: float = None


class TrafficAI:
    TYPES = {
        "sedan": ((120, 150, 210), 4.4, (21.0, 29.0)),
        "suv": ((145, 160, 130), 4.9, (19.0, 27.0)),
        "truck": ((95, 115, 135), 10.8, (15.0, 21.0)),
        "motorcycle": ((210, 190, 90), 2.5, (24.0, 33.0)),
    }
    MIN_GAP = 20.0

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.vehicles = []
        self.spawn_timer = 0.0
        self._seed_initial_traffic()

    def _seed_initial_traffic(self):
        for z in [34, 68, 104, 142, 184]:
            self._spawn(z=z)

    def _spawn(self, z=120.0):
        kind = self.rng.choices(
            list(self.TYPES.keys()),
            weights=[0.46, 0.27, 0.18, 0.09],
            k=1,
        )[0]
        color, length, speed_range = self.TYPES[kind]
        lane = self._choose_safe_spawn_lane(z, length)
        if lane is None:
            return False
        desired_speed = self.rng.uniform(*speed_range)
        self.vehicles.append(
            TrafficVehicle(
                lane_u=lane,
                z=z + self.rng.uniform(-4.0, 5.0),
                speed_mps=desired_speed,
                vehicle_type=kind,
                color=color,
                length=length,
                desired_speed_mps=desired_speed,
                target_lane_u=None,
            )
        )
        self.vehicles[-1].blocked_timer = 0.0
        return True

    def update(self, dt, ego_speed_mps):
        dt = max(0.001, min(0.08, dt))
        self.spawn_timer += dt
        density_interval = 0.15 + self.rng.random() * 0.35
        if self.spawn_timer > density_interval and len(self.vehicles) < 45:
            self._spawn(z=190 + self.rng.random() * 120)
            self.spawn_timer = 0.0

        self._apply_following_model(dt)
        for vehicle in self.vehicles:
            vehicle.z += (vehicle.speed_mps - ego_speed_mps) * dt
            vehicle.lane_change_timer += dt
            self._maybe_lane_change(vehicle)
            if vehicle.target_lane_u is not None:
                lane_diff = vehicle.target_lane_u - vehicle.lane_u
                direction = 1.0 if lane_diff > 0 else -1.0
                move = direction * min(abs(lane_diff), dt * 0.45 + abs(lane_diff) * dt * 1.8)
                vehicle.lane_u += move
                if abs(vehicle.target_lane_u - vehicle.lane_u) < 0.01:
                    vehicle.lane_u = vehicle.target_lane_u
                    vehicle.target_lane_u = None

        self._enforce_lane_separation()
        self.vehicles = [v for v in self.vehicles if -45.0 < v.z < 320.0]
        attempts = 0
        while len(self.vehicles) < 25 and attempts < 25:
            self._spawn(z=160 + self.rng.random() * 150)
            attempts += 1
        return self.vehicles

    def _maybe_lane_change(self, vehicle):
        if vehicle.target_lane_u is not None:
            return
            
        desire_change = False
        if hasattr(vehicle, 'blocked_timer') and vehicle.blocked_timer > 1.5:
            desire_change = True
        elif vehicle.lane_change_timer > 4.0 and self.rng.random() < 0.02:
            desire_change = True
            
        if not desire_change:
            return
            
        vehicle.lane_change_timer = 0.0
        other_lane = LANES[0] if abs(vehicle.lane_u - LANES[1]) < 0.2 else LANES[1]
        safe = self._lane_has_gap(other_lane, vehicle.z, vehicle.length + self.MIN_GAP + 16.0, ignore=vehicle)
        
        if safe:
            vehicle.target_lane_u = other_lane
            if hasattr(vehicle, 'blocked_timer'):
                vehicle.blocked_timer = 0.0

    def _choose_safe_spawn_lane(self, z, length):
        lanes = LANES[:]
        self.rng.shuffle(lanes)
        for lane in lanes:
            if self._lane_has_gap(lane, z, length + self.MIN_GAP + 18.0):
                return lane
        return None

    def _lane_has_gap(self, lane_u, z, gap, ignore=None):
        for vehicle in self.vehicles:
            if vehicle is ignore:
                continue
            effective_lane = vehicle.target_lane_u if vehicle.target_lane_u is not None else vehicle.lane_u
            if abs(effective_lane - lane_u) < 0.34 and abs(vehicle.z - z) < gap:
                return False
        return True

    def _apply_following_model(self, dt):
        for lane in LANES:
            lane_vehicles = sorted(
                [v for v in self.vehicles if abs(v.lane_u - lane) < 0.36 and v.target_lane_u is None],
                key=lambda v: v.z,
                reverse=True,
            )
            leader = None
            for vehicle in lane_vehicles:
                if not hasattr(vehicle, 'blocked_timer'):
                    vehicle.blocked_timer = 0.0
                target_speed = vehicle.desired_speed_mps
                blocked = False
                if leader is not None:
                    gap = leader.z - vehicle.z - (leader.length + vehicle.length) * 0.5
                    desired_gap = self.MIN_GAP + vehicle.speed_mps * 0.85 + vehicle.length
                    if gap < desired_gap:
                        target_speed = min(target_speed, leader.speed_mps - min(4.0, (desired_gap - gap) * 0.12))
                        if leader.speed_mps < vehicle.desired_speed_mps - 2.0:
                            blocked = True
                            
                if blocked:
                    vehicle.blocked_timer += dt
                else:
                    vehicle.blocked_timer = max(0.0, vehicle.blocked_timer - dt)
                    
                vehicle.speed_mps += (max(9.0, target_speed) - vehicle.speed_mps) * min(1.0, dt * 1.8)
                leader = vehicle

    def _enforce_lane_separation(self):
        for lane in LANES:
            lane_vehicles = sorted([v for v in self.vehicles if abs(v.lane_u - lane) < 0.40], key=lambda v: v.z, reverse=True)
            leader = None
            for vehicle in lane_vehicles:
                if leader is not None:
                    min_gap = self.MIN_GAP + (leader.length + vehicle.length) * 0.5
                    if leader.z - vehicle.z < min_gap:
                        vehicle.z = leader.z - min_gap
                        vehicle.speed_mps = min(vehicle.speed_mps, leader.speed_mps)
                leader = vehicle
