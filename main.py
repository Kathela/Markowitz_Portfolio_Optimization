#!/usr/bin/env python3
"""
Quant Portfolio Optimization System — Main Entry Point
========================================================
Runs the complete analysis pipeline:
  1. Data download & preprocessing
  2. Portfolio optimisation (5 strategies)
  3. Monte Carlo simulation
  4. Efficient frontier + Capital Market Line
  5. Risk metric suite (Sharpe, Sortino, VaR, CVaR, Max Drawdown, Beta, Alpha …)
  6. Walk-forward backtesting with rolling rebalancing
  7. Nine professional visualisations saved to ./plots/
  8. Stress testing under historical crisis periods

Usage
-----
    python main.py
    python main.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _banner(text: str) -> None:
    """Print a section banner to the console."""
    line = "═" * 62
    print(f"\n{line}")
    print(f"  {text}")
    print(line)


def _section(text: str) -> None:
    print(f"\n── {text} {'─' * (58 - len(text))}")


def main(config_path: str = "config.yaml") -> None:
    t0 = time.perf_counter()
    config = load_config(config_path)

    # Lazy imports to keep startup fast
    from src.data_loader import DataLoader
    from src.portfolio_optimizer import PortfolioOptimizer
    from src.risk_metrics import RiskMetrics
    from src.backtester import Backtester
    from src.visualization import Visualizer

    _banner("QUANT PORTFOLIO OPTIMIZATION SYSTEM")
    print(f"  Assets    : {', '.join(config['data']['tickers'])}")
    print(f"  Benchmark : {config['data']['benchmark']}")
    print(f"  Start     : {config['data']['start_date']}")
    print(f"  Risk-free : {config['optimization']['risk_free_rate']:.0%}")

    # ── 1. Data ────────────────────────────────────────────────────────────────
    _section("1 / 7   Data Acquisition")
    loader = DataLoader(config)
    prices = loader.download()

    returns = loader.get_returns()
    asset_returns = loader.get_asset_returns()
    benchmark_returns = loader.get_benchmark_returns()
    mu = loader.get_expected_returns()
    sigma = loader.get_covariance(method="sample")

    asset_mu = mu.drop(labels=[loader._benchmark_ticker.replace("^", "")], errors="ignore")
    asset_sigma = sigma.drop(
        index=[loader._benchmark_ticker.replace("^", "")],
        columns=[loader._benchmark_ticker.replace("^", "")],
        errors="ignore",
    )
    asset_vols = pd.Series(
        np.sqrt(np.diag(asset_sigma.values)), index=asset_sigma.index
    )

    print(loader.summary().to_string())

    # ── 2. Portfolio Optimisation ──────────────────────────────────────────────
    _section("2 / 7   Portfolio Optimisation")
    target_ret = config["optimization"]["target_return"]
    optimizer = PortfolioOptimizer(
        asset_mu, asset_sigma,
        risk_free_rate=config["optimization"]["risk_free_rate"],
        trading_days=config["optimization"]["trading_days"],
    )

    strategies = optimizer.compare_all(
        target_return=target_ret,
        constrained_min=config["optimization"]["constrained"]["min_weight"],
        constrained_max=config["optimization"]["constrained"]["max_weight"],
    )

    print("\n" + optimizer.comparison_table(strategies).to_string())

    # ── 3. Monte Carlo + Frontier + CML ───────────────────────────────────────
    _section("3 / 7   Monte Carlo & Efficient Frontier")
    n_sim = config["optimization"]["n_simulations"]
    print(f"  Simulating {n_sim:,} random portfolios …")
    mc_rets, mc_vols, mc_sharpes, _ = optimizer.monte_carlo(n=n_sim)

    print("  Tracing efficient frontier (200 points) …")
    eff_rets, eff_vols, _ = optimizer.efficient_frontier()

    cml_vols, cml_rets = optimizer.capital_market_line()
    print(f"  Done.  Frontier: {len(eff_rets)} points  ·  MC Sharpe range [{mc_sharpes.min():.2f}, {mc_sharpes.max():.2f}]")

    # ── 4. Risk Metrics ────────────────────────────────────────────────────────
    _section("4 / 7   Risk & Performance Metrics")
    returns_dict: dict[str, pd.Series] = {}
    for name, res in strategies.items():
        common = [t for t in res.weights.index if t in asset_returns.columns]
        w = res.weights[common].values
        w /= w.sum()
        returns_dict[name] = (asset_returns[common] @ w).rename(name)

    metrics_df = RiskMetrics.compare_portfolios(
        returns_dict,
        benchmark_returns=benchmark_returns,
        risk_free_rate=config["optimization"]["risk_free_rate"],
    )

    # Format for display
    pct_rows = ["Annualised Return", "Annualised Volatility", "Max Drawdown",
                "VaR 95% (daily)", "CVaR 95% (daily)", "Tracking Error", "Alpha (annualised)"]
    fmt_df = metrics_df.copy().astype(object)
    for row in fmt_df.index:
        for col in fmt_df.columns:
            val = metrics_df.loc[row, col]
            if isinstance(val, float) and any(kw in row for kw in ["Return", "Volatility", "Drawdown", "VaR", "CVaR", "Error", "Alpha"]):
                fmt_df.loc[row, col] = f"{val:.2%}"
            elif isinstance(val, float):
                fmt_df.loc[row, col] = f"{val:.2f}"
    print("\n" + fmt_df.to_string())

    # ── 5. Backtesting ─────────────────────────────────────────────────────────
    _section("5 / 7   Walk-Forward Backtesting")
    print(
        f"  In-sample: {config['backtesting']['in_sample_years']} years  "
        f"| Rebalancing: {config['backtesting']['rebalancing_frequency']}  "
        f"| TC: {config['backtesting']['transaction_cost']*100:.0f} bps"
    )

    backtester = Backtester(prices, config)
    backtest_result = backtester.run_all_strategies()

    bt_metrics = backtest_result.metrics(
        benchmark_returns=benchmark_returns,
        risk_free_rate=config["optimization"]["risk_free_rate"],
    )
    final_values = backtest_result.portfolio_values.iloc[-1].rename("Final Value ($1 invested)")
    print("\n" + bt_metrics.loc[["Annualised Return", "Annualised Volatility",
                                  "Sharpe Ratio", "Max Drawdown"]].round(4).to_string())
    print("\nFinal Portfolio Values (started at $1.00):")
    print(final_values.round(4).to_string())

    rolling = backtester.rolling_metrics(backtest_result, window=config["risk"]["rolling_window"])

    # ── 6. Stress Testing ──────────────────────────────────────────────────────
    _section("6 / 7   Stress Testing")
    stress_periods = config.get("stress_tests", {}).get("periods", [])
    stress_returns: dict[str, pd.DataFrame] = {}
    for period in stress_periods:
        try:
            slice_ret = loader.stress_test_slice(period["start"], period["end"])
            if not slice_ret.empty:
                # Keep only asset columns
                available = [c for c in slice_ret.columns if c != loader._benchmark_ticker.replace("^", "")]
                stress_returns[period["name"]] = slice_ret[available]
                worst = slice_ret[available].mean(axis=1).sum()
                print(f"  {period['name']:<25} ({period['start']} → {period['end']})  "
                      f"Equal-wt cumulative: {worst:.2%}")
        except Exception as e:
            logger.warning("Stress test '%s' failed: %s", period.get("name"), e)

    # ── 7. Visualisations ──────────────────────────────────────────────────────
    _section("7 / 7   Generating Visualisations")
    viz = Visualizer(config)

    saved_paths: list[str] = []

    p = viz.plot_efficient_frontier(
        mc_rets, mc_vols, mc_sharpes,
        eff_rets, eff_vols,
        cml_vols, cml_rets,
        strategies, asset_mu,
        asset_sigma_diag=asset_vols.values,
        risk_free_rate=config["optimization"]["risk_free_rate"],
    )
    saved_paths.append(p)

    saved_paths.append(viz.plot_weights_comparison(strategies))
    saved_paths.append(viz.plot_correlation_heatmap(asset_returns))
    saved_paths.append(viz.plot_portfolio_growth(backtest_result.portfolio_values))
    saved_paths.append(viz.plot_drawdown(backtest_result.portfolio_values))
    saved_paths.append(viz.plot_rolling_metrics(rolling))
    saved_paths.append(viz.plot_risk_return_scatter(asset_mu, asset_vols, strategies))

    # Monthly returns heatmap for the Max Sharpe strategy
    ms_returns = returns_dict.get("Max Sharpe")
    if ms_returns is not None:
        saved_paths.append(viz.plot_monthly_returns_heatmap(ms_returns, "Max Sharpe — Monthly Returns"))

    if stress_returns:
        saved_paths.append(viz.plot_stress_test(stress_returns, strategies, asset_returns))

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    _banner(f"ANALYSIS COMPLETE  ({elapsed:.0f}s)")
    print(f"  {'Charts saved':15}: {len(saved_paths)} files in ./{config['visualization']['output_dir']}/")
    for path in saved_paths:
        if path:
            print(f"    ✓ {path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quant Portfolio Optimization System")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args.config)
