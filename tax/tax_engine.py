class TaxEngine:
    def __init__(self, config):
        self.config = config

    def realized_tax(self, gain):
        if gain <= 0:
            return 0.0
        if gain <= self.config.threshold:
            return gain * self.config.lower_rate
        return (
            self.config.threshold * self.config.lower_rate +
            (gain - self.config.threshold) * self.config.upper_rate
        )
