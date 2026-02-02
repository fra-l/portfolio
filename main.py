from models.factor_etf import FactorETF
from factors.factor_model import FactorModel
from optimizer.factor_replication_optimizer import FactorReplicationOptimizer

def main():
    prices = {"AAPL": 180, "MSFT": 420, "JPM": 170, "KO": 60}

    factor_exposures = {
        "AAPL": {"Value": -0.3, "Momentum": 0.8, "Quality": 0.6},
        "MSFT": {"Value": -0.2, "Momentum": 0.6, "Quality": 0.7},
        "JPM": {"Value": 0.9, "Momentum": 0.1, "Quality": 0.4},
        "KO": {"Value": 0.7, "Momentum": 0.2, "Quality": 0.8},
    }

    factor_model = FactorModel(
        factor_names=["Value", "Momentum", "Quality"],
        exposure_map=factor_exposures
    )

    value_etf = FactorETF(
        name="Value ETF",
        target_exposures={"Value": 1.0, "Momentum": 0.0, "Quality": 0.2}
    )

    optimizer = FactorReplicationOptimizer()
    portfolio = optimizer.optimize(
        symbols=list(prices.keys()),
        prices=prices,
        factor_model=factor_model,
        target_exposures=value_etf.target_exposures,
        budget=20_000
    )

    print("Factor replicated portfolio:")
    for sym, qty in portfolio.items():
        print(sym, round(qty, 4))

if __name__ == "__main__":
    main()