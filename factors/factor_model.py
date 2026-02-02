class FactorModel:
    def __init__(self, factor_names, exposure_map):
        self.factor_names = factor_names
        self.exposure_map = exposure_map

    def exposure(self, symbol):
        return self.exposure_map[symbol]
