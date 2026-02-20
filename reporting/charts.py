import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def _build_history_df(history):
    df = pd.DataFrame([{"date": h["date"], "value": h["value"]} for h in history])
    df = df.set_index("date")
    return df


def _build_trades_df(trades):
    if not trades:
        return pd.DataFrame(columns=["date", "type", "ticker", "shares", "price", "amount"])
    df = pd.DataFrame(trades)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _build_alloc_df(history):
    rows = []
    for h in history:
        row = {"date": h["date"]}
        row.update(h["allocations"])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("date").fillna(0.0)
    total = df.sum(axis=1)
    pct = df.div(total, axis=0) * 100
    return pct


def _normalize_spy(spy_prices, initial_value):
    spy = spy_prices.dropna()
    if spy.empty:
        return spy
    return spy / spy.iloc[0] * initial_value


def _plot_portfolio_vs_spy(history_df, trades_df, spy_normalized, output_path,
                           leverage_series=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(history_df.index, history_df["value"], label="Portfolio (€)", color="#2196F3", linewidth=1.5)
    ax.plot(spy_normalized.index, spy_normalized.values, label="SPY (normalized €)",
            color="#FF9800", linewidth=1.5, linestyle="--")

    # Buy markers
    buys = trades_df[trades_df["type"] == "buy"] if not trades_df.empty else pd.DataFrame()
    if not buys.empty:
        buy_dates = buys["date"].unique()
        for d in buy_dates:
            if d in history_df.index:
                ax.scatter(d, history_df.loc[d, "value"], marker="^", color="#4CAF50", s=40, zorder=5)

    # Sell markers
    sells = trades_df[trades_df["type"] == "sell"] if not trades_df.empty else pd.DataFrame()
    if not sells.empty:
        sell_dates = sells["date"].unique()
        for d in sell_dates:
            if d in history_df.index:
                ax.scatter(d, history_df.loc[d, "value"], marker="v", color="#F44336", s=40, zorder=5)

    # Legend proxies for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2196F3", linewidth=1.5, label="Portfolio (€)"),
        Line2D([0], [0], color="#FF9800", linewidth=1.5, linestyle="--", label="SPY (normalized €)"),
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

    ax.set_title("Portfolio Value vs SPY Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    return fig


def _plot_trade_activity(trades_df, output_path):
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


def _plot_allocation_over_time(alloc_pct, output_path):
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


def plot_results(history, trades, spy_prices, initial_value):
    os.makedirs("reports", exist_ok=True)

    history_df = _build_history_df(history)
    trades_df = _build_trades_df(trades)
    alloc_pct = _build_alloc_df(history)
    spy_norm = _normalize_spy(spy_prices, initial_value)

    leverage_series = None
    if history and "leverage" in history[0]:
        lev_rows = [{"date": h["date"], "leverage": h["leverage"]} for h in history]
        leverage_series = pd.DataFrame(lev_rows).set_index("date")["leverage"]

    _plot_portfolio_vs_spy(history_df, trades_df, spy_norm, "reports/portfolio_vs_spy.png",
                           leverage_series=leverage_series)
    _plot_trade_activity(trades_df, "reports/trade_activity.png")
    _plot_allocation_over_time(alloc_pct, "reports/allocation_over_time.png")

    print("\nCharts saved to reports/")
    plt.show()
