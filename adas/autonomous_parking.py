class AutonomousParking:
    def __init__(self):
        self.secured = False

    def target_speed(self, stage, collision_risk, near_shoulder):
        if self.secured:
            return 0.0
        if stage >= 4 and near_shoulder:
            return 0.0
        if stage >= 4:
            return 72.0
        if stage >= 3:
            return 82.0 if collision_risk < 0.55 else 60.0
        if collision_risk > 0.65:
            return 75.0
        return 100.0

    def update_secured(self, speed_kmh, near_shoulder, stage):
        if stage >= 4 and near_shoulder and speed_kmh < 1.2:
            self.secured = True
        if stage < 3:
            self.secured = False
        return self.secured

