class Position:
    def __init__(self, ticker):
        self.ticker = ticker
        self.lots = []

    def total_shares(self):
        return sum(l.shares for l in self.lots)

    def sell_shares(self, shares, current_price, method="HIFO"):
        if method == "HIFO":
            sorted_lots = sorted(self.lots, key=lambda l: l.cost_basis, reverse=True)
        elif method == "FIFO":
            sorted_lots = sorted(self.lots, key=lambda l: l.purchase_date or 0)
        else:
            raise ValueError(f"Unknown lot selection method: {method}")

        remaining = shares
        proceeds = 0.0
        realized_gain = 0.0
        lots_consumed = 0

        for lot in sorted_lots:
            if remaining <= 0:
                break

            sell_from_lot = min(lot.shares, remaining)
            lot_proceeds = sell_from_lot * current_price
            lot_gain = sell_from_lot * (current_price - lot.cost_basis)

            proceeds += lot_proceeds
            realized_gain += lot_gain
            lot.shares -= sell_from_lot
            remaining -= sell_from_lot
            lots_consumed += 1

        self.lots = [l for l in self.lots if l.shares > 1e-12]

        return {
            "shares_sold": shares - remaining,
            "proceeds": proceeds,
            "realized_gain": realized_gain,
            "lots_consumed": lots_consumed,
        }
