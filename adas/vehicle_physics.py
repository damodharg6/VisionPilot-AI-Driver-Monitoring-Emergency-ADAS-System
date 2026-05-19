import math


class VehiclePhysics:
    def __init__(self, speed_kmh=100.0, lane_u=0.48):
        self.speed_kmh = speed_kmh
        self.lane_u = lane_u
        self.steer_velocity = 0.0
        self.accel = 0.0

    @property
    def speed_mps(self):
        return self.speed_kmh / 3.6

    def update(self, dt, target_speed_kmh, target_lane_u):
        dt = max(0.001, min(0.08, dt))
        desired_accel = max(-8.2, min(3.2, (target_speed_kmh - self.speed_kmh) * 0.45))
        self.accel += (desired_accel - self.accel) * min(1.0, dt * 3.2)
        self.speed_kmh = max(0.0, self.speed_kmh + self.accel * dt * 3.6)

        lane_error = target_lane_u - self.lane_u
        desired_steer = max(-1.0, min(1.0, lane_error * 1.55))
        self.steer_velocity += (desired_steer - self.steer_velocity) * min(1.0, dt * 2.4)
        
        speed_factor = max(0.0, min(1.2, self.speed_kmh / 45.0))
        self.lane_u += self.steer_velocity * dt * 0.72 * speed_factor

        if abs(lane_error) < 0.012:
            self.lane_u = target_lane_u
            self.steer_velocity *= 0.55

        return self.speed_kmh, self.lane_u, self.steer_velocity

