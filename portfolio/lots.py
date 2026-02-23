from __future__ import annotations

import datetime
from typing import Optional


class Lot:
    def __init__(
        self,
        shares: float,
        cost_basis: float,
        purchase_date: Optional[datetime.date] = None,
    ) -> None:
        self.shares = shares
        self.cost_basis = cost_basis
        self.purchase_date = purchase_date
