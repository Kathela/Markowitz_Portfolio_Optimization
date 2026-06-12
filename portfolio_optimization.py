#!/usr/bin/env python3
"""
Markowitz Portfolio Optimization
=================================
Production-ready implementation of Modern Portfolio Theory (MPT) following
Harry Markowitz's seminal 1952 paper "Portfolio Selection".

This module:
  - Downloads 3 years of adjusted-close prices for 8 diversified assets
  - Computes annualised expected returns and the covariance matrix
  - Runs a 10,000-portfolio Monte Carlo simulation
  - Finds the Maximum Sharpe and Minimum Variance portfolios via scipy SLSQP
  - Traces the mean-variance Efficient Frontier
  - Saves three publication-quality charts to ./plots/
  - Prints a formatted allocation table with financial interpretation

Usage
-----
    python portfolio_optimization.py

Requirements
------------
    Python >= 3.10
    See requirements.txt for pinned library versions.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize

# ── Global constants ──────────────────────────────────────────────────────────
TICKERS: list[str] = ["AAPL", "MSFT", "AMZN", "GOOGL", "JNJ", "JPM", "GLD", "^GSPC"]

# Risk-free rate: approximate annualised US 10-year Treasury yield (as of 2024)
RISK_FREE_RATE: float = 0.05

# Annualisation factor: standard trading days in a US equity calendar year
TRADING_DAYS: int = 252

# Monte Carlo sample size: large enough to approximate the feasible set
N_SIMULATIONS: int = 10_000

# Historical look-back window
YEARS: int = 3

# Output directory for all saved figures
PLOTS_DIR: str = "plots"


# ── 1. Data Acquisition ───────────────────────────────────────────────────────

def download_data(tickers: list[str], years: int = YEARS) -> pd.DataFrame:
    """Download adjusted-close prices for the given tickers.

    Uses yfinance with `auto_adjust=True` so prices are split- and
    dividend-adjusted (total-return series), consistent with standard
    performance benchmarking.

    Missing observations (public holidays, early-close days) are
    forward-filled before any remaining leading NaNs are dropped,
    keeping the longest clean common history across all assets.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols, e.g. ["AAPL", "^GSPC"].
    years : int
        Number of years of history to download (default: 3).

    Returns
    -------
    pd.DataFrame
        Daily adjusted-close prices with tickers as columns.
    """
    end = datetime.today()
    start = end - timedelta(days=int(years * 365.25))

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance 0.2+ returns a MultiIndex DataFrame; isolate the Close slice.
    # Level 0 contains the price metric, Level 1 contains the ticker symbol.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Fallback for a single ticker (should not occur with this ticker list)
        prices = raw[["Close"]]

    prices = prices.ffill().dropna()
    return prices


# ── 2. Return & Statistics Calculation ───────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily simple (arithmetic) returns from price series.

    Arithmetic returns are preferred over log-returns for portfolio
    construction because portfolio return is exactly a weighted sum of
    asset arithmetic returns: R_p = w · R, with no approximation needed.
    The first row (always NaN after pct_change) is dropped.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily adjusted-close prices.

    Returns
    -------
    pd.DataFrame
        Daily arithmetic returns, same columns as `prices`.
    """
    return prices.pct_change().dropna()


def annualise_stats(
    returns: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute annualised expected returns (mu) and covariance matrix (sigma).

    Under the assumption of i.i.d. daily returns:
      mu_annual    = mu_daily   × 252   (linearity of expectation)
      sigma_annual = sigma_daily × 252  (variance scales linearly with time,
                                         so σ scales as √252)

    Parameters
    ----------
    returns : pd.DataFrame
        Daily arithmetic returns.

    Returns
    -------
    mu : pd.Series
        Annualised expected return for each asset.
    sigma : pd.DataFrame
        Annualised covariance matrix (assets × assets).
    """
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.cov() * TRADING_DAYS
    return mu, sigma


# ── 3. Portfolio Performance Metrics ─────────────────────────────────────────

def portfolio_performance(
    weights: np.ndarray,
    mu: pd.Series,
    sigma: pd.DataFrame,
) -> tuple[float, float, float]:
    """Compute annualised return, volatility, and Sharpe ratio for a portfolio.

    The portfolio variance uses the full covariance matrix:
        σ²_p = wᵀ Σ w
    capturing all pairwise correlations, which is the core insight of MPT.
    Sharpe ratio: (R_p - R_f) / σ_p, measuring excess return per unit of risk.

    Parameters
    ----------
    weights : array-like
        Portfolio weight vector; must sum to 1.
    mu : pd.Series
        Annualised expected returns.
    sigma : pd.DataFrame
        Annualised covariance matrix.

    Returns
    -------
    (annual_return, annual_volatility, sharpe_ratio) : tuple of float
    """
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ sigma.values @ w))
    sharpe = (ret - RISK_FREE_RATE) / vol
    return ret, vol, sharpe


# ── 4. Monte Carlo Simulation ─────────────────────────────────────────────────

def monte_carlo_simulation(
    mu: pd.Series,
    sigma: pd.DataFrame,
    n: int = N_SIMULATIONS,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample random long-only portfolios and compute their performance.

    Weight vectors are drawn from a uniform Dirichlet distribution by
    normalising n_assets uniform random numbers to sum to 1. This is
    equivalent to sampling uniformly from the n-simplex (all weights
    non-negative and summing to 1).

    A fixed seed ensures reproducibility; the cloud of points approximates
    the full feasible set of long-only portfolios.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    sigma : pd.DataFrame
        Annualised covariance matrix.
    n : int
        Number of random portfolios to simulate (default: 10,000).
    seed : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    (sim_returns, sim_vols, sim_sharpes, sim_weights)
        Four parallel arrays of length n.
    """
    np.random.seed(seed)
    n_assets = len(mu)

    sim_rets = np.empty(n)
    sim_vols = np.empty(n)
    sim_sharpes = np.empty(n)
    sim_weights = np.empty((n, n_assets))

    for i in range(n):
        w = np.random.random(n_assets)
        w /= w.sum()   # normalise to enforce sum-to-1 constraint
        sim_weights[i] = w
        sim_rets[i], sim_vols[i], sim_sharpes[i] = portfolio_performance(w, mu, sigma)

    return sim_rets, sim_vols, sim_sharpes, sim_weights


# ── 5. Scipy Optimisation Infrastructure ─────────────────────────────────────

def _neg_sharpe(
    weights: np.ndarray,
    mu: pd.Series,
    sigma: pd.DataFrame,
) -> float:
    """Return the negative Sharpe ratio (minimising this maximises Sharpe)."""
    _, _, sharpe = portfolio_performance(weights, mu, sigma)
    return -sharpe


def _portfolio_vol(
    weights: np.ndarray,
    mu: pd.Series,
    sigma: pd.DataFrame,
) -> float:
    """Return portfolio volatility (minimised directly for min-variance and frontier)."""
    _, vol, _ = portfolio_performance(weights, mu, sigma)
    return vol


def _run_optimiser(
    objective,
    mu: pd.Series,
    sigma: pd.DataFrame,
    x0: Optional[np.ndarray] = None,
    extra_constraints: Optional[list] = None,
) -> object:
    """Run scipy SLSQP with automatic retry on non-convergence.

    Two-stage approach:
      Attempt 1: standard unit-interval bounds (0, 1) — allows full concentration.
      Attempt 2: tighter bounds (0.01, 0.50) — avoids flat gradient regions near
                 the corners of the simplex that can stall SLSQP.

    Raises RuntimeError if both attempts fail to converge.

    Parameters
    ----------
    objective : callable
        Objective function with signature f(weights, mu, sigma) -> float.
    mu : pd.Series
        Annualised expected returns.
    sigma : pd.DataFrame
        Annualised covariance matrix.
    x0 : np.ndarray, optional
        Starting weight vector. Defaults to equal weights.
    extra_constraints : list of dict, optional
        Additional scipy constraint dicts (e.g., target-return constraint).

    Returns
    -------
    scipy OptimizeResult
    """
    n = len(mu)
    if x0 is None:
        x0 = np.ones(n) / n   # equal-weight starting point (neutral, feasible)

    base_constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if extra_constraints:
        base_constraints.extend(extra_constraints)

    # Attempt 1: loose bounds, standard tolerance
    for bounds, maxiter, ftol in [
        (tuple((0.0, 1.0) for _ in range(n)), 1_000, 1e-9),
        (tuple((0.01, 0.50) for _ in range(n)), 2_000, 1e-9),  # tighter retry
    ]:
        result = minimize(
            objective,
            x0,
            args=(mu, sigma),
            method="SLSQP",
            bounds=bounds,
            constraints=base_constraints,
            options={"ftol": ftol, "maxiter": maxiter},
        )
        if result.success:
            return result

    raise RuntimeError(
        f"Optimisation did not converge after two attempts.\n"
        f"Solver message: {result.message}"
    )


# ── 6. Optimal Portfolio Construction ────────────────────────────────────────

def max_sharpe_portfolio(
    mu: pd.Series,
    sigma: pd.DataFrame,
) -> tuple[np.ndarray, float, float, float]:
    """Find the tangency (Maximum Sharpe Ratio) portfolio.

    The tangency portfolio is the point on the efficient frontier where
    the Capital Market Line (from the risk-free asset) is tangent to the
    frontier. In CAPM theory this is the 'market portfolio'. It delivers
    the highest expected return per unit of risk.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    sigma : pd.DataFrame
        Annualised covariance matrix.

    Returns
    -------
    (weights, expected_return, volatility, sharpe_ratio)
    """
    result = _run_optimiser(_neg_sharpe, mu, sigma)
    w = result.x
    ret, vol, sharpe = portfolio_performance(w, mu, sigma)
    return w, ret, vol, sharpe


def min_variance_portfolio(
    mu: pd.Series,
    sigma: pd.DataFrame,
) -> tuple[np.ndarray, float, float, float]:
    """Find the Global Minimum Variance (GMV) portfolio.

    The GMV portfolio sits at the leftmost tip of the efficient frontier.
    It ignores expected returns entirely, focusing purely on minimising
    portfolio volatility through diversification. Appropriate for investors
    who prioritise capital preservation over returns.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    sigma : pd.DataFrame
        Annualised covariance matrix.

    Returns
    -------
    (weights, expected_return, volatility, sharpe_ratio)
    """
    result = _run_optimiser(_portfolio_vol, mu, sigma)
    w = result.x
    ret, vol, sharpe = portfolio_performance(w, mu, sigma)
    return w, ret, vol, sharpe


# ── 7. Efficient Frontier ─────────────────────────────────────────────────────

def efficient_frontier(
    mu: pd.Series,
    sigma: pd.DataFrame,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the mean-variance efficient frontier.

    Solves n_points minimum-volatility problems, each with an additional
    equality constraint fixing the portfolio return to a target level.
    The target return range spans from the GMV return (the frontier's
    leftmost achievable point) to the highest single-asset expected return
    (the rightmost feasible target for a long-only portfolio).

    Points below the GMV return are dominated (same or higher risk for lower
    return) and are excluded by design.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns.
    sigma : pd.DataFrame
        Annualised covariance matrix.
    n_points : int
        Number of frontier points to compute (default: 200).

    Returns
    -------
    (frontier_returns, frontier_volatilities) : tuple of np.ndarray
        Parallel arrays tracing the efficient frontier from GMV to max return.
    """
    _, gmv_ret, _, _ = min_variance_portfolio(mu, sigma)
    max_achievable_ret = float(mu.max())

    target_rets = np.linspace(gmv_ret, max_achievable_ret, n_points)
    eff_rets: list[float] = []
    eff_vols: list[float] = []

    for target in target_rets:
        ret_constraint = [{
            "type": "eq",
            "fun": lambda w, t=target: portfolio_performance(w, mu, sigma)[0] - t,
        }]
        try:
            result = _run_optimiser(_portfolio_vol, mu, sigma, extra_constraints=ret_constraint)
            r, v, _ = portfolio_performance(result.x, mu, sigma)
            eff_rets.append(r)
            eff_vols.append(v)
        except RuntimeError:
            # Skip target returns that are infeasible under the long-only constraint
            pass

    return np.array(eff_rets), np.array(eff_vols)


# ── 8. Visualisations ─────────────────────────────────────────────────────────

def plot_efficient_frontier(
    sim_rets: np.ndarray,
    sim_vols: np.ndarray,
    sim_sharpes: np.ndarray,
    frontier_rets: np.ndarray,
    frontier_vols: np.ndarray,
    ms_ret: float,
    ms_vol: float,
    mv_ret: float,
    mv_vol: float,
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Scatter of 10,000 Monte Carlo portfolios with efficient frontier overlay.

    Color-encodes each simulated portfolio by its Sharpe ratio (viridis scale).
    The efficient frontier is drawn as a crimson line. Both optimal portfolios
    are marked with gold and cyan stars with text annotations.

    Returns the file path of the saved PNG.
    """
    os.makedirs(plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    sc = ax.scatter(
        sim_vols, sim_rets,
        c=sim_sharpes, cmap="viridis", alpha=0.45, s=8,
        label="10,000 random portfolios",
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    ax.plot(frontier_vols, frontier_rets, color="crimson", lw=2.5, label="Efficient Frontier")

    # Mark Max Sharpe portfolio (gold star)
    ax.scatter(ms_vol, ms_ret, marker="*", color="gold", s=500, zorder=6,
               edgecolors="black", linewidths=0.5, label="Max Sharpe ★")
    ax.annotate(
        "Max Sharpe", xy=(ms_vol, ms_ret),
        xytext=(10, 8), textcoords="offset points",
        fontsize=9, color="darkorange", fontweight="bold",
    )

    # Mark Min Variance portfolio (cyan star)
    ax.scatter(mv_vol, mv_ret, marker="*", color="deepskyblue", s=500, zorder=6,
               edgecolors="black", linewidths=0.5, label="Min Variance ★")
    ax.annotate(
        "Min Variance", xy=(mv_vol, mv_ret),
        xytext=(10, -14), textcoords="offset points",
        fontsize=9, color="steelblue", fontweight="bold",
    )

    ax.set_xlabel("Annual Volatility (Risk)", fontsize=12)
    ax.set_ylabel("Annual Expected Return", fontsize=12)
    ax.set_title(
        "Markowitz Efficient Frontier\n10,000 Monte Carlo Portfolios — Coloured by Sharpe Ratio",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(alpha=0.3)

    path = os.path.join(plots_dir, "efficient_frontier.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_portfolio_weights(
    ms_weights: np.ndarray,
    mv_weights: np.ndarray,
    tickers: list[str],
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Side-by-side bar chart comparing Max Sharpe and Min Variance allocations.

    Displays each asset's portfolio weight as a percentage, making it easy to
    see how the two optimisation objectives lead to different concentration patterns.

    Returns the file path of the saved PNG.
    """
    os.makedirs(plots_dir, exist_ok=True)

    x = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, ms_weights * 100, width,
           label="Max Sharpe", color="gold", edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, mv_weights * 100, width,
           label="Min Variance", color="steelblue", edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Portfolio Allocation (%)", fontsize=12)
    ax.set_title(
        "Optimal Portfolio Weights: Max Sharpe vs Min Variance",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(plots_dir, "portfolio_weights.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    plots_dir: str = PLOTS_DIR,
) -> str:
    """Annotated heatmap of pairwise return correlations across all 8 assets.

    Low (or negative) correlations are the engine of Markowitz diversification:
    blending assets that don't move together reduces portfolio variance without
    proportionally reducing expected return — the 'free lunch' of diversification.

    Returns the file path of the saved PNG.
    """
    os.makedirs(plots_dir, exist_ok=True)

    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True, fmt=".2f",
        cmap="RdYlGn",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_title("Asset Return Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(plots_dir, "correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 9. Summary Output ─────────────────────────────────────────────────────────

def print_summary(
    tickers: list[str],
    ms_weights: np.ndarray, ms_ret: float, ms_vol: float, ms_sharpe: float,
    mv_weights: np.ndarray, mv_ret: float, mv_vol: float, mv_sharpe: float,
) -> None:
    """Print a formatted allocation table and financial interpretation."""
    bar = "─" * 54
    print(f"\n{bar}")
    print(f"  {'Asset':<8}  {'Max Sharpe':>14}  {'Min Variance':>14}")
    print(bar)
    for t, msw, mvw in zip(tickers, ms_weights, mv_weights):
        print(f"  {t:<8}  {msw:>13.1%}  {mvw:>14.1%}")
    print(bar)
    print(f"  {'Return':<8}  {ms_ret:>13.2%}  {mv_ret:>14.2%}")
    print(f"  {'Volatility':<8}  {ms_vol:>13.2%}  {mv_vol:>14.2%}")
    print(f"  {'Sharpe':<8}  {ms_sharpe:>13.2f}  {mv_sharpe:>14.2f}")
    print(bar)

    print("""
┌─ Financial Interpretation ──────────────────────────────────────────────────┐
│                                                                              │
│  MAX SHARPE RATIO PORTFOLIO (Tangency Portfolio)                             │
│  • Maximises return per unit of risk — the most "efficient" allocation.     │
│  • Typically concentrates in high-momentum growth equities (AAPL, MSFT,     │
│    AMZN, GOOGL) which have historically driven superior risk-adj. returns.  │
│  • Suitable for growth-oriented investors with a 5+ year horizon.           │
│  • This is the portfolio all rational investors would choose when they can  │
│    combine it with a risk-free asset (the Capital Market Line result).      │
│                                                                              │
│  MINIMUM VARIANCE PORTFOLIO (Global Minimum Variance)                       │
│  • Ignores expected returns; minimises total portfolio volatility.          │
│  • Naturally tilts towards defensive assets (JNJ healthcare, GLD as a      │
│    safe-haven) and lower-beta names whose low mutual correlation dampens    │
│    portfolio swings.                                                         │
│  • Suitable for conservative investors or as a defensive portfolio sleeve.  │
│  • Paradoxically, GMV often outperforms higher-return target portfolios on  │
│    a risk-adjusted basis due to estimation error in expected returns.       │
│                                                                              │
│  THE EFFICIENT FRONTIER                                                      │
│  • Traces every portfolio that cannot be improved: no other combination     │
│    delivers higher expected return for the same risk, or lower risk for the │
│    same expected return.                                                     │
│  • Any portfolio below/right of the frontier is sub-optimal (inefficient).  │
│  • Portfolios above the frontier are unattainable given this asset universe.│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
""")


# ── 10. Main Entry Point ──────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate the full Markowitz portfolio optimisation pipeline."""
    print("═" * 60)
    print("  MARKOWITZ PORTFOLIO OPTIMISATION")
    print("═" * 60)

    # ── Step 1: Download price data ───────────────────────────────────────────
    print("\n[1/6]  Downloading price data …")
    prices = download_data(TICKERS, years=YEARS)

    # Strip the '^' prefix from index symbols (e.g. "^GSPC" → "GSPC") for labels
    prices.columns = [c.replace("^", "") for c in prices.columns]
    clean_tickers = list(prices.columns)

    print(f"       {len(prices):,} trading days  ·  {prices.shape[1]} assets")
    print(f"       {prices.index[0].date()} → {prices.index[-1].date()}")

    # ── Step 2: Compute returns and statistics ────────────────────────────────
    print("\n[2/6]  Computing daily returns and annualised statistics …")
    returns = compute_returns(prices)
    mu, sigma = annualise_stats(returns)

    # ── Step 3: Monte Carlo simulation ───────────────────────────────────────
    print("\n[3/6]  Running Monte Carlo simulation …")
    sim_rets, sim_vols, sim_sharpes, _ = monte_carlo_simulation(mu, sigma)
    print(f"       Simulated Sharpe range: [{sim_sharpes.min():.2f}, {sim_sharpes.max():.2f}]")

    # ── Step 4: Scipy portfolio optimisation ─────────────────────────────────
    print("\n[4/6]  Optimising portfolios …")
    ms_w, ms_ret, ms_vol, ms_sharpe = max_sharpe_portfolio(mu, sigma)
    print(f"       Max Sharpe   → Ret {ms_ret:.2%}  Vol {ms_vol:.2%}  Sharpe {ms_sharpe:.2f}")
    mv_w, mv_ret, mv_vol, mv_sharpe = min_variance_portfolio(mu, sigma)
    print(f"       Min Variance → Ret {mv_ret:.2%}  Vol {mv_vol:.2%}  Sharpe {mv_sharpe:.2f}")

    # ── Step 5: Efficient frontier ────────────────────────────────────────────
    print("\n[5/6]  Tracing efficient frontier (200 points) …")
    eff_rets, eff_vols = efficient_frontier(mu, sigma)
    print(f"       {len(eff_rets)} feasible frontier points computed")

    # ── Step 6: Plots + summary ───────────────────────────────────────────────
    print("\n[6/6]  Saving plots to ./plots/ …")
    p1 = plot_efficient_frontier(
        sim_rets, sim_vols, sim_sharpes,
        eff_rets, eff_vols,
        ms_ret, ms_vol, mv_ret, mv_vol,
    )
    p2 = plot_portfolio_weights(ms_w, mv_w, clean_tickers)
    p3 = plot_correlation_heatmap(returns)
    for path in (p1, p2, p3):
        print(f"       ✓ {path}")

    print_summary(
        clean_tickers,
        ms_w, ms_ret, ms_vol, ms_sharpe,
        mv_w, mv_ret, mv_vol, mv_sharpe,
    )


if __name__ == "__main__":
    main()
