class FactorTarget:
    def __init__(self, target_weights):
        self.target_weights = target_weights

    def vector(self, factor_names):
        return [self.target_weights.get(f, 0.0) for f in factor_names]