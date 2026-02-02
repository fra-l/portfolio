class UniverseSelector:
    def __init__(self, min_r2=0.3):
        self.min_r2 = min_r2

    def select(self, exposures, r2):
        return exposures[r2 >= self.min_r2].index.tolist()