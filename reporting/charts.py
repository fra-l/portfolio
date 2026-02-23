from __future__ import annotations

import os
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def _build_history_df(history: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"date": h["date"], "value": h["value"]} for h in history])
    df = df.set_index("date")
    return df


def _build_trades_df(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["date", "type", "ticker", "shares", "price", "amount"])
    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _build_alloc_df(history: list[dict]) -> pd.DataFrame:
    rows = []
    for h in history:
        row = {"date": h["date"]}
        row.update(h["allocations"])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date").fillna(0.0)
    total = df.sum(axis=1)
    pct = df.div(total, axis=0) * 100
    return pct


_BENCHMARK_COLORS = ["#FF9800", "#E91E63", "#9C27B0", "#00BCD4", "#8BC34A", "#FF5722"]


def _normalize_benchmark(prices: pd.Series, initial_value: float) -> pd.Series:
    s = prices.dropna()
    if s.empty:
        return s
    return s / s.iloc[0] * initial_value


def _plot_portfolio_vs_benchmarks(
    history_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    benchmark_prices: dict[str, pd.Series],
    initial_value: float,
    output_path: str,
    leverage_series: Optional[pd.Series] = None,
) -> plt.Figure:
    """
    benchmark_prices : dict[str, pd.Series] — raw (un-normalised) benchmark prices
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history_df.index, history_df["value"],
            label="Portfolio (€)", color="#2196F3", linewidth=1.5)

    for i, (ticker, prices) in enumerate(benchmark_prices.items()):
        color = _BENCHMARK_COLORS[i % len(_BENCHMARK_COLORS)]
        normalized = _normalize_benchmark(prices, initial_value)
        ax.plot(normalized.index, normalized.values,
                label=f"{ticker} (normalized €)",
                color=color, linewidth=1.5, linestyle="--")

    # Buy markers
    buys = trades_df[trades_df["type"] == "buy"] if not trades_df.empty else pd.DataFrame()
    if not buys.empty:
        for d in buys["date"].unique():
            if d in history_df.index:
                ax.scatter(d, history_df.loc[d, "value"], marker="^", color="#4CAF50", s=40, zorder=5)

    # Sell markers
    sells = trades_df[trades_df["type"] == "sell"] if not trades_df.empty else pd.DataFrame()
    if not sells.empty:
        for d in sells["date"].unique():
            if d in history_df.index:
                ax.scatter(d, history_df.loc[d, "value"], marker="v", color="#F44336", s=40, zorder=5)

    # Legend: portfolio + benchmarks + trade markers
    legend_elements = [
        Line2D([0], [0], color="#2196F3", linewidth=1.5, label="Portfolio (€)"),
    ]
    for i, ticker in enumerate(benchmark_prices):
        color = _BENCHMARK_COLORS[i % len(_BENCHMARK_COLORS)]
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=1.5, linestyle="--",
                   label=f"{ticker} (normalized €)")
        )
    legend_elements += [
        Line2D([0], [0], marker="^", color="#4CAF50", linestyle="None", markersize=7, label="Buy"),
        Line2D([0], [0], marker="v", color="#F44336", linestyle="None", markersize=7, label="Sell"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Leverage ratio overlay (secondary y-axis)
    if leverage_series is not None and not leverage_series.empty \
            and leverage_series.max() > 1.001:
        ax2 = ax.twinx()
        ax2.plot(leverage_series.index, leverage_series.values,
                 color="#9C27B0", linewidth=1.0, linestyle=":", alpha=0.8, label="Leverage")
        ax2.axhline(1.0, color="#9C27B0", linewidth=0.5, linestyle=":", alpha=0.3)
        ax2.set_ylabel("Leverage", color="#9C27B0", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#9C27B0")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}×"))
        ax2.set_ylim(bottom=0.9)

    benchmark_label = " / ".join(benchmark_prices.keys()) if benchmark_prices else "No benchmark"
    ax.set_title(f"Portfolio Value vs {benchmark_label}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


def _plot_trade_activity(trades_df: pd.DataFrame, output_path: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    if trades_df.empty:
        ax.text(0.5, 0.5, "No trades recorded", ha="center", va="center", transform=ax.transAxes)
    else:
        monthly = trades_df.copy()
        monthly["month"] = monthly["date"].dt.to_period("M")
        buys = monthly[monthly["type"] == "buy"].groupby("month")["amount"].sum()
        sells = monthly[monthly["type"] == "sell"].groupby("month")["amount"].sum()

        all_months = buys.index.union(sells.index)
        buys = buys.reindex(all_months, fill_value=0)
        sells = sells.reindex(all_months, fill_value=0)

        x = range(len(all_months))
        width = 0.4
        ax.bar([i - width / 2 for i in x], buys.values, width=width, label="Buys", color="#4CAF50")
        ax.bar([i + width / 2 for i in x], -sells.values, width=width, label="Sells", color="#F44336")

        ax.set_xticks(list(x))
        ax.set_xticklabels([str(m) for m in all_months], rotation=45, ha="right", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"€{abs(v):,.0f}")
        )
        ax.legend()

    ax.set_title("Monthly Trade Activity")
    ax.set_xlabel("Month")
    ax.set_ylabel("Volume (€)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


def _plot_allocation_over_time(alloc_pct: pd.DataFrame, output_path: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    # Separate cash column
    cols = [c for c in alloc_pct.columns if c != "cash"]
    cash_col = alloc_pct["cash"] if "cash" in alloc_pct.columns else pd.Series(0, index=alloc_pct.index)

    # Top-8 tickers by mean allocation
    mean_alloc = alloc_pct[cols].mean().sort_values(ascending=False)
    top8 = list(mean_alloc.head(8).index)
    other_cols = [c for c in cols if c not in top8]

    plot_df = alloc_pct[top8].copy()
    if other_cols:
        plot_df["Other"] = alloc_pct[other_cols].sum(axis=1)
    plot_df["Cash"] = cash_col

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(plot_df.columns), 1)) for i in range(len(plot_df.columns))]

    ax.stackplot(plot_df.index, plot_df.T.values, labels=plot_df.columns, colors=colors, alpha=0.85)
    ax.set_ylim(0, 100)
    ax.set_title("Portfolio Allocation Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Allocation (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


def _plot_rolling_factor_exposures(
    exposure_history: list[dict], output_path: str
) -> Optional[plt.Figure]:
    if not exposure_history:
        return None

    rows = []
    for snap in exposure_history:
        row = {"date": snap["date"]}
        for factor, val in zip(snap["factors"], snap["exposure"]):
            row[factor] = val
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date")

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    for i, col in enumerate(df.columns):
        ax.plot(df.index, df[col], label=col,
                linewidth=1.5, color=colors[i % len(colors)],
                marker="o", markersize=3)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_title("Rolling Portfolio Factor Exposures (at Each Rebalance)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Factor Exposure")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


def plot_results(
    history: list[dict],
    trades: list[dict],
    benchmark_prices: Union[dict[str, pd.Series], pd.Series, None],
    initial_value: float,
    exposure_history: Optional[list[dict]] = None,
    show: bool = True,
) -> None:
    """
    benchmark_prices : dict[str, pd.Series] or pd.Series (legacy SPY-only path)
    show             : if True, call plt.show() after saving charts (default True).
                       Set to False for headless/CI environments.
    """
    os.makedirs("reports", exist_ok=True)

    # Normalise legacy bare Series to dict
    if isinstance(benchmark_prices, pd.Series):
        benchmark_prices = {"SPY": benchmark_prices}
    if benchmark_prices is None:
        benchmark_prices = {}

    history_df = _build_history_df(history)
    trades_df = _build_trades_df(trades)
    alloc_pct = _build_alloc_df(history)

    leverage_series = None
    if history and "leverage" in history[0]:
        lev_rows = [{"date": h["date"], "leverage": h["leverage"]} for h in history]
        leverage_series = pd.DataFrame(lev_rows).set_index("date")["leverage"]

    _plot_portfolio_vs_benchmarks(
        history_df, trades_df, benchmark_prices, initial_value,
        "reports/portfolio_vs_benchmarks.png",
        leverage_series=leverage_series,
    )
    _plot_trade_activity(trades_df, "reports/trade_activity.png")
    _plot_allocation_over_time(alloc_pct, "reports/allocation_over_time.png")
    if exposure_history:
        _plot_rolling_factor_exposures(
            exposure_history, "reports/rolling_factor_exposures.png"
        )

    print("\nCharts saved to reports/")
    if show:
        plt.show()
