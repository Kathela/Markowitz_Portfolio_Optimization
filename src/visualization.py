"""
Professional Financial Visualisations
========================================
All charts are saved to disk as PNG files.  ``plt.show()`` is never called,
making this module safe for headless server environments.

Charts produced
---------------
1.  Efficient Frontier   — MC scatter + frontier curve + CML + optimal portfolios + asset labels
2.  Portfolio Weights    — grouped bar chart for all strategies
3.  Correlation Heatmap  — annotated Seaborn heatmap
4.  Portfolio Growth     — cumulative value comparison vs benchmark
5.  Drawdown             — underwater equity curves
6.  Rolling Metrics      — 1-year rolling Sharpe, volatility, return
7.  Risk-Return Scatter  — individual assets + all optimal portfolios
8.  Monthly Returns      — calendar heatmap (returns by year × month)
9.  Stress Test          — normalised performance during crisis periods
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .portfolio_optimizer import OptimizationResult
from .risk_metrics import RiskMetrics

logger = logging.getLogger(__name__)

# Marker styles for up to 6 optimal portfolios
_STRATEGY_STYLES: Dict[str, Dict] = {
    "Max Sharpe":    {"color": "gold",        "marker": "*", "zorder": 8, "s": 500},
    "Min Variance":  {"color": "deepskyblue", "marker": "*", "zorder": 8, "s": 500},
    "Equal Weight":  {"color": "limegreen",   "marker": "D", "zorder": 7, "s": 120},
    "Constrained":   {"color": "tomato",      "marker": "^", "zorder": 7, "s": 160},
    "Target Return": {"color": "orchid",      "marker": "s", "zorder": 7, "s": 120},
}


def _pct_formatter(x, _):
    return f"{x:.0%}"


def _pct2_formatter(x, _):
    return f"{x:.1%}"


class Visualizer:
    """
    Publication-quality financial chart generator.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from ``config.yaml``.

    Examples
    --------
    >>> viz = Visualizer(config)
    >>> viz.plot_efficient_frontier(mc_data, strategies, frontier, cml)
    >>> viz.plot_portfolio_growth(backtest_result)
    """

    def __init__(self, config: dict) -> None:
        self.output_dir = Path(config["visualization"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi: int = config["visualization"].get("dpi", 150)
        style = config["visualization"].get("figure_style", "seaborn-v0_8-whitegrid")
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("seaborn-v0_8-whitegrid")

    # ── 1. Efficient Frontier ──────────────────────────────────────────────────

    def plot_efficient_frontier(
        self,
        mc_rets: np.ndarray,
        mc_vols: np.ndarray,
        mc_sharpes: np.ndarray,
        frontier_rets: np.ndarray,
        frontier_vols: np.ndarray,
        cml_vols: np.ndarray,
        cml_rets: np.ndarray,
        strategies: Dict[str, OptimizationResult],
        asset_mu: pd.Series,
        asset_sigma_diag: np.ndarray,
        risk_free_rate: float = 0.05,
    ) -> str:
        """
        Master efficient frontier chart.

        Includes:
        - 10,000 Monte Carlo portfolios (scatter, coloured by Sharpe)
        - Efficient frontier curve (crimson)
        - Capital Market Line (dashed grey)
        - Risk-free rate marker
        - All optimal portfolios (stars/diamonds with labels)
        - Individual assets as labelled points
        """
        fig, ax = plt.subplots(figsize=(13, 8))

        # Monte Carlo scatter
        sc = ax.scatter(
            mc_vols, mc_rets,
            c=mc_sharpes, cmap="viridis", alpha=0.35, s=6,
            label="_nolegend_",
        )
        cbar = plt.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label("Sharpe Ratio", fontsize=10)

        # Efficient frontier
        if len(frontier_rets) > 0:
            ax.plot(frontier_vols, frontier_rets, "r-", lw=2.5, label="Efficient Frontier", zorder=5)

        # Capital Market Line
        ax.plot(cml_vols, cml_rets, "--", color="grey", lw=1.5, alpha=0.8, label="Capital Market Line")

        # Risk-free rate point
        ax.scatter(0, risk_free_rate, marker="o", color="grey", s=80, zorder=6)
        ax.annotate(f"Risk-free\n{risk_free_rate:.0%}", xy=(0, risk_free_rate),
                    xytext=(6, -18), textcoords="offset points", fontsize=7.5, color="grey")

        # Individual assets
        for ticker, mu_val, vol_val in zip(
            asset_mu.index, asset_mu.values, asset_sigma_diag
        ):
            ax.scatter(vol_val, mu_val, marker="o", color="white", edgecolors="black",
                       s=70, zorder=7, linewidths=1)
            ax.annotate(ticker, xy=(vol_val, mu_val),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)

        # Optimal portfolios
        for name, res in strategies.items():
            style = _STRATEGY_STYLES.get(name, {"color": "orange", "marker": "P", "s": 150, "zorder": 7})
            ax.scatter(res.volatility, res.expected_return,
                       color=style["color"], marker=style["marker"],
                       s=style["s"], zorder=style["zorder"],
                       edgecolors="black", linewidths=0.6,
                       label=f"{name}  (S={res.sharpe_ratio:.2f})")

        ax.set_xlabel("Annual Volatility (Risk)", fontsize=12)
        ax.set_ylabel("Annual Expected Return", fontsize=12)
        ax.set_title(
            "Mean-Variance Efficient Frontier\n"
            "10,000 Monte Carlo Portfolios · Capital Market Line · Optimal Allocations",
            fontsize=13, fontweight="bold",
        )
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.3)

        return self._save(fig, "01_efficient_frontier.png")

    # ── 2. Portfolio Weights Comparison ───────────────────────────────────────

    def plot_weights_comparison(
        self,
        strategies: Dict[str, OptimizationResult],
    ) -> str:
        """Grouped bar chart comparing asset allocations across all strategies."""
        tickers = list(next(iter(strategies.values())).weights.index)
        x = np.arange(len(tickers))
        n_strats = len(strategies)
        total_width = 0.8
        bar_width = total_width / n_strats

        fig, ax = plt.subplots(figsize=(max(12, n_strats * 2), 6))

        colors = plt.cm.tab10(np.linspace(0, 1, n_strats))
        for idx, (name, res) in enumerate(strategies.items()):
            offset = (idx - n_strats / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                res.weights.values * 100,
                bar_width,
                label=f"{name} (S={res.sharpe_ratio:.2f})",
                color=colors[idx],
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=11)
        ax.set_ylabel("Portfolio Allocation (%)", fontsize=12)
        ax.set_title("Portfolio Weights by Strategy", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)

        return self._save(fig, "02_portfolio_weights.png")

    # ── 3. Correlation Heatmap ─────────────────────────────────────────────────

    def plot_correlation_heatmap(self, returns: pd.DataFrame) -> str:
        """
        Annotated heatmap of pairwise asset return correlations.

        Correlations below 1 are the mechanism of diversification in MPT.
        The colour scale (red-yellow-green) makes diversification opportunities
        immediately visible.
        """
        corr = returns.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # upper triangle mask

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True, fmt=".2f",
            cmap="RdYlGn",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            ax=ax,
            cbar_kws={"label": "Pearson Correlation"},
        )
        ax.set_title("Asset Return Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return self._save(fig, "03_correlation_heatmap.png")

    # ── 4. Portfolio Growth ────────────────────────────────────────────────────

    def plot_portfolio_growth(
        self,
        portfolio_values: pd.DataFrame,
        log_scale: bool = False,
    ) -> str:
        """
        Cumulative portfolio value comparison over the backtest period.

        All strategies start at $1.00 on the first day of the out-of-sample
        period, making relative performance directly comparable.

        Parameters
        ----------
        log_scale : bool
            Use logarithmic y-axis (useful for long multi-decade series).
        """
        fig, ax = plt.subplots(figsize=(13, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(portfolio_values.columns)))
        lss = ["-", "--", "-.", ":", "-", "--"]

        for idx, col in enumerate(portfolio_values.columns):
            ls = "--" if "Benchmark" in col else lss[idx % len(lss)]
            lw = 2.0 if "Benchmark" in col else 1.8
            ax.plot(portfolio_values[col], label=col, lw=lw, ls=ls, color=colors[idx])

        ax.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Portfolio Value ($1 initial)", fontsize=11)
        ax.set_title("Portfolio Growth: Walk-Forward Backtest", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.3)
        if log_scale:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:.2f}"))

        return self._save(fig, "04_portfolio_growth.png")

    # ── 5. Drawdown ────────────────────────────────────────────────────────────

    def plot_drawdown(self, portfolio_values: pd.DataFrame) -> str:
        """
        Underwater equity curves showing drawdown for each strategy.

        The chart shows how far each strategy is below its all-time high
        at every point in time — a critical risk visualisation for clients.
        """
        fig, ax = plt.subplots(figsize=(13, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(portfolio_values.columns)))

        for idx, col in enumerate(portfolio_values.columns):
            dd_series, _ = RiskMetrics.max_drawdown(portfolio_values[col].dropna())
            ls = "--" if "Benchmark" in col else "-"
            ax.fill_between(dd_series.index, dd_series.values * 100, 0,
                            alpha=0.25, color=colors[idx])
            ax.plot(dd_series.index, dd_series.values * 100, label=col,
                    lw=1.5, ls=ls, color=colors[idx])

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Drawdown (%)", fontsize=11)
        ax.set_title("Portfolio Drawdown (Underwater Equity Curves)", fontsize=13, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
        ax.grid(alpha=0.3)

        return self._save(fig, "05_drawdown.png")

    # ── 6. Rolling Metrics ─────────────────────────────────────────────────────

    def plot_rolling_metrics(
        self,
        rolling_metrics_dict: Dict[str, pd.DataFrame],
    ) -> str:
        """
        Three-panel chart of 1-year rolling Sharpe, volatility, and return.

        Shows how performance characteristics evolve over time and helps
        identify regime changes (e.g. pre/post COVID, rate hike cycle).
        """
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        colors = plt.cm.tab10(np.linspace(0, 1, len(rolling_metrics_dict)))

        metric_labels = {
            "Sharpe":     "Rolling 1Y Sharpe Ratio",
            "Volatility": "Rolling 1Y Annualised Volatility",
            "Return":     "Rolling 1Y Annualised Return",
        }
        formatters = {
            "Sharpe":     lambda y, _: f"{y:.1f}",
            "Volatility": _pct_formatter,
            "Return":     _pct_formatter,
        }

        for idx, (name, df) in enumerate(rolling_metrics_dict.items()):
            ls = "--" if "Benchmark" in name else "-"
            for ax, metric in zip(axes, ["Sharpe", "Volatility", "Return"]):
                if metric in df.columns:
                    ax.plot(df.index, df[metric], label=name, lw=1.5,
                            ls=ls, color=colors[idx])

        for ax, metric in zip(axes, ["Sharpe", "Volatility", "Return"]):
            ax.set_title(metric_labels[metric], fontsize=11)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatters[metric]))
            ax.grid(alpha=0.3)
            if metric == "Sharpe":
                ax.axhline(1.0, color="grey", lw=0.8, ls=":", alpha=0.7)
                ax.axhline(0.0, color="black", lw=0.6, ls="-", alpha=0.3)

        axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
        axes[-1].set_xlabel("Date", fontsize=11)
        fig.suptitle("Rolling 1-Year Performance Metrics", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        return self._save(fig, "06_rolling_metrics.png")

    # ── 7. Risk-Return Scatter ─────────────────────────────────────────────────

    def plot_risk_return_scatter(
        self,
        asset_mu: pd.Series,
        asset_vols: pd.Series,
        strategies: Dict[str, OptimizationResult],
    ) -> str:
        """
        Risk-return scatter plot with individual assets and optimal portfolios.

        Allows visual comparison of individual asset positions versus
        diversified portfolio allocations on the same risk-return axes.
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Individual assets
        for ticker in asset_mu.index:
            ax.scatter(asset_vols[ticker], asset_mu[ticker],
                       marker="o", color="#4c72b0", s=80, zorder=5,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(ticker, xy=(asset_vols[ticker], asset_mu[ticker]),
                        xytext=(6, 4), textcoords="offset points", fontsize=9, color="#333333")

        # Optimal portfolios
        for name, res in strategies.items():
            style = _STRATEGY_STYLES.get(name, {"color": "orange", "marker": "P", "s": 160, "zorder": 7})
            ax.scatter(res.volatility, res.expected_return,
                       color=style["color"], marker=style["marker"],
                       s=style["s"], zorder=style["zorder"],
                       edgecolors="black", linewidths=0.7,
                       label=f"{name}")
            ax.annotate(name, xy=(res.volatility, res.expected_return),
                        xytext=(8, 4), textcoords="offset points", fontsize=8, fontweight="bold")

        ax.set_xlabel("Annual Volatility (Risk)", fontsize=12)
        ax.set_ylabel("Annual Expected Return", fontsize=12)
        ax.set_title("Risk-Return Space: Individual Assets vs Optimal Portfolios",
                     fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.3)

        return self._save(fig, "07_risk_return_scatter.png")

    # ── 8. Monthly Returns Heatmap ─────────────────────────────────────────────

    def plot_monthly_returns_heatmap(
        self,
        returns: pd.Series,
        title: str = "Monthly Returns (%)",
    ) -> str:
        """
        Calendar heatmap of monthly returns (months as columns, years as rows).

        Provides a compact overview of seasonality patterns and lets the
        reader quickly identify the worst and best months at a glance.
        """
        monthly = (1 + returns).resample("ME").prod() - 1
        monthly_df = monthly.to_frame("ret")
        monthly_df["year"] = monthly_df.index.year
        monthly_df["month"] = monthly_df.index.month

        pivot = monthly_df.pivot_table(index="year", columns="month", values="ret")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

        fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6 + 2)))
        sns.heatmap(
            pivot * 100,
            annot=True, fmt=".1f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Monthly Return (%)"},
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Year")
        plt.tight_layout()

        return self._save(fig, "08_monthly_returns_heatmap.png")

    # ── 9. Stress Test ─────────────────────────────────────────────────────────

    def plot_stress_test(
        self,
        stress_returns: Dict[str, pd.DataFrame],
        strategies: Dict[str, OptimizationResult],
        asset_returns_full: pd.DataFrame,
    ) -> str:
        """
        Normalised performance of each strategy during historical crisis periods.

        For each stress period, applies the Max Sharpe and Min Variance weights
        to the crisis-period returns to simulate what the optimised portfolio
        would have experienced.
        """
        n_periods = len(stress_returns)
        if n_periods == 0:
            logger.warning("No stress test periods available.")
            return ""

        fig, axes = plt.subplots(1, n_periods, figsize=(6 * n_periods, 5), sharey=False)
        if n_periods == 1:
            axes = [axes]

        for ax, (period_name, period_ret) in zip(axes, stress_returns.items()):
            asset_cols = [c for c in period_ret.columns if c in period_ret.columns]

            for strat_name, res in strategies.items():
                style = _STRATEGY_STYLES.get(strat_name, {"color": "orange"})
                common = [t for t in res.weights.index if t in period_ret.columns]
                if not common:
                    continue
                w = res.weights[common].values
                w = w / w.sum()  # renormalise to available assets
                port_rets = period_ret[common] @ w
                cumulative = (1 + port_rets).cumprod()
                cumulative = cumulative / cumulative.iloc[0]
                ax.plot(cumulative.index, (cumulative - 1) * 100,
                        label=strat_name, lw=2, color=style["color"])

            # Benchmark (equal weight across available assets)
            ew_rets = period_ret.mean(axis=1)
            ew_cum = (1 + ew_rets).cumprod()
            ew_cum = ew_cum / ew_cum.iloc[0]
            ax.plot(ew_cum.index, (ew_cum - 1) * 100,
                    label="Equal Weight", lw=1.5, ls="--", color="grey")

            ax.axhline(0, color="black", lw=0.7, ls="-", alpha=0.4)
            ax.set_title(period_name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Date", fontsize=9)
            ax.set_ylabel("Cumulative Return (%)" if ax == axes[0] else "", fontsize=9)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax.legend(fontsize=8, loc="lower left")
            ax.grid(alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        fig.suptitle("Stress Test: Performance During Market Crises",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        return self._save(fig, "09_stress_test.png")

    # ── Helper ─────────────────────────────────────────────────────────────────

    def _save(self, fig: plt.Figure, filename: str) -> str:
        """Save figure to output directory and close it."""
        path = str(self.output_dir / filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved: %s", path)
        return path
