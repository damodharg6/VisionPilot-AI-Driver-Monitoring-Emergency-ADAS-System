class AttentionSmoother:
    def __init__(self, alpha=0.12):
        self.alpha = alpha
        self.value = 100.0

    def update(self, target):
        self.value += (target - self.value) * self.alpha
        return self.value

