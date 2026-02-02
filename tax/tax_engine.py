class TaxEngine:
    def __init__(self):
        self.low_rate = 0.27
        self.high_rate = 0.42
        self.threshold = 10000

    def tax_due(self, realized_gain):
        if realized_gain <= self.threshold:
            return realized_gain * self.low_rate
        return self.threshold * self.low_rate + (realized_gain - self.threshold) * self.high_rate