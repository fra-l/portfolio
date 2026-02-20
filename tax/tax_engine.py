from config import TaxConfig

class TaxEngine:
    def __init__(self, config=None):
        if config is None:
            config = TaxConfig()
        self.low_rate = config.lower_rate
        self.high_rate = config.upper_rate
        self.threshold = config.threshold

    def tax_due(self, realized_gain):
        if realized_gain <= self.threshold:
            return realized_gain * self.low_rate
        return self.threshold * self.low_rate + (realized_gain - self.threshold) * self.high_rate
