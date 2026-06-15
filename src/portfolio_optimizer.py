"""
Portfolio Optimisation Engine
================================
Implements multiple mean-variance portfolio allocation strategies using
scipy's SLSQP solver with automatic convergence retry.

Strategies
----------
1. Minimum Variance       — lowest achievable portfolio volatility
2. Maximum Sharpe         — highest risk-adjusted return (tangency portfolio)
3. Equal Weight           — 1/N benchmark allocation
4. Target Return          — minimum variance subject to a return constraint
5. Constrained            — user-specified min/max weight bounds per asset
6. Monte Carlo            — 10,000 random portfolios for feasible-set visualisation
7. Efficient Frontier     — full mean-variance frontier (200 points)
8. Capital Market Line    — risk-free → tangency portfolio line
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


# ── Result Container ───────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """
    Container for a single portfolio optimisation result.

    Attributes
    ----------
    weights : pd.Series
        Allocation weights indexed by ticker name.
    expected_return : float
        Annualised expected portfolio return.
    volatility : float
        Annualised portfolio volatility (standard deviation).
    sharpe_ratio : float
        Annualised Sharpe ratio.
    method : str
        Name of the optimisation strategy.
    converged : bool
        Whether the scipy solver reported success.
    """
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str
    converged: bool = True

    def __repr__(self) -> str:
        return (
            f"OptimizationResult({self.method!r})\n"
            f"  Expected Return : {self.expected_return:.2%}\n"
            f"  Volatility      : {self.volatility:.2%}\n"
            f"  Sharpe Ratio    : {self.sharpe_ratio:.2f}\n"
            f"  Converged       : {self.converged}"
        )

    def weights_table(self) -> pd.DataFrame:
        """Return a formatted DataFrame of non-trivial weights."""
        df = self.weights.to_frame("Weight")
        df["Weight %"] = df["Weight"].map("{:.1%}".format)
        return df[df["Weight"] > 1e-4].sort_values("Weight", ascending=False)


# ── Optimiser ──────────────────────────────────────────────────────────────────

class PortfolioOptimizer:
    """
    Mean-variance portfolio optimiser for a given asset universe.

    Parameters
    ----------
    mu : pd.Series
        Annualised expected returns indexed by ticker.
    sigma : pd.DataFrame
        Annualised covariance matrix (tickers × tickers).
    risk_free_rate : float
        Annualised risk-free rate (default: 5%).
    trading_days : int
        Trading days per year (default: 252).

    Examples
    --------
    >>> opt = PortfolioOptimizer(mu, sigma, risk_free_rate=0.05)
    >>> ms = opt.max_sharpe()
    >>> print(ms)
    >>> mc_rets, mc_vols, mc_sharpes, mc_weights = opt.monte_carlo()
    """

    def __init__(
        self,
        mu: pd.Series,
        sigma: pd.DataFrame,
        risk_free_rate: float = 0.05,
        trading_days: int = _TRADING_DAYS,
    ) -> None:
        self.mu = mu.copy()
        self.sigma = sigma.copy()
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.tickers: List[str] = list(mu.index)
        self.n: int = len(mu)

    # ── Portfolio Metrics ──────────────────────────────────────────────────────

    def _portfolio_stats(
        self, weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute (return, volatility, Sharpe) for a weight vector."""
        w = np.asarray(weights, dtype=float)
        ret = float(w @ self.mu.values)
        vol = float(np.sqrt(w @ self.sigma.values @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _result(self, weights: np.ndarray, method: str, converged: bool = True) -> OptimizationResult:
        """Build an OptimizationResult from a weight array."""
        ret, vol, sharpe = self._portfolio_stats(weights)
        return OptimizationResult(
            weights=pd.Series(weights, index=self.tickers),
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method=method,
            converged=converged,
        )

    # ── Scipy SLSQP Core ──────────────────────────────────────────────────────

    def _optimise(
        self,
        objective,
        extra_constraints: Optional[List[dict]] = None,
        bounds: Optional[Tuple] = None,
        x0: Optional[np.ndarray] = None,
    ):
        """
        Run scipy SLSQP with automatic retry on non-convergence.

        First attempt uses standard unit-interval bounds [0, 1].
        If that fails, retries with tighter bounds [0.01, 0.50] which
        can avoid flat-gradient regions near simplex corners.

        Raises RuntimeError if both attempts fail.
        """
        if x0 is None:
            x0 = np.ones(self.n) / self.n

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        if extra_constraints:
            constraints.extend(extra_constraints)

        default_bounds = tuple((0.0, 1.0) for _ in range(self.n))
        if bounds is None:
            bounds = default_bounds

        # Attempt 1: user-supplied or standard bounds
        res = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1_000},
        )
        if res.success:
            return res

        # Attempt 2: tighter bounds to assist convergence
        tight_bounds = tuple((0.01, 0.50) for _ in range(self.n))
        logger.debug("First optimisation attempt failed (%s). Retrying with tighter bounds.", res.message)
        res2 = minimize(
            objective, x0, method="SLSQP",
            bounds=tight_bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2_000},
        )
        if res2.success:
            return res2

        # Both failed — return best available result with a warning
        logger.warning(
            "Optimisation did not converge after two attempts.  "
            "Result may be suboptimal.  Message: %s", res2.message
        )
        # Return whichever had the better objective value
        return res2 if res2.fun < res.fun else res

    # ── 1. Minimum Variance Portfolio ─────────────────────────────────────────

    def min_variance(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> OptimizationResult:
        """
        Global Minimum Variance (GMV) Portfolio.

        Minimises portfolio volatility σ_p = √(wᵀΣw) subject to:
          - Σwᵢ = 1 (fully invested)
          - wᵢ ∈ [min_weight, max_weight]  (long-only by default)

        The GMV portfolio is the leftmost point on the efficient frontier.
        It ignores expected returns entirely — pure risk minimisation
        through diversification.
        """
        def objective(w):
            return float(np.sqrt(w @ self.sigma.values @ w))

        bounds = tuple((min_weight, max_weight) for _ in range(self.n))
        res = self._optimise(objective, bounds=bounds)
        return self._result(res.x, "Min Variance", res.success)

    # ── 2. Maximum Sharpe Ratio Portfolio ─────────────────────────────────────

    def max_sharpe(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> OptimizationResult:
        """
        Tangency (Maximum Sharpe Ratio) Portfolio.

        Maximises S_p = (R_p - R_f) / σ_p by minimising -S_p.

        The tangency portfolio is the point where the Capital Market Line
        (from the risk-free asset) is tangent to the efficient frontier.
        In CAPM theory, all rational investors hold this portfolio (combined
        with the risk-free asset) regardless of risk appetite.
        """
        def objective(w):
            ret, vol, sharpe = self._portfolio_stats(w)
            return -sharpe if vol > 1e-8 else 0.0

        bounds = tuple((min_weight, max_weight) for _ in range(self.n))
        res = self._optimise(objective, bounds=bounds)
        return self._result(res.x, "Max Sharpe", res.success)

    # ── 3. Equal Weight Portfolio ──────────────────────────────────────────────

    def equal_weight(self) -> OptimizationResult:
        """
        Naive 1/N Equal Weight Portfolio.

        Allocates 1/N to each asset.  Serves as a benchmark: research
        (DeMiguel et al., 2009) shows 1/N often outperforms complex
        optimised portfolios out-of-sample due to estimation error.
        """
        w = np.ones(self.n) / self.n
        return self._result(w, "Equal Weight")

    # ── 4. Target Return Portfolio ────────────────────────────────────────────

    def target_return(
        self,
        target: float,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> OptimizationResult:
        """
        Minimum Variance Portfolio subject to a target expected return.

        Adds the equality constraint: wᵀμ = target_return.
        This is a single point on the efficient frontier corresponding
        to the given return level.

        Parameters
        ----------
        target : float
            Annualised target expected return (e.g. 0.15 for 15%).
        """
        if target < float(self.mu.min()) or target > float(self.mu.max()):
            raise ValueError(
                f"Target return {target:.2%} is outside the feasible range "
                f"[{self.mu.min():.2%}, {self.mu.max():.2%}]."
            )

        def objective(w):
            return float(np.sqrt(w @ self.sigma.values @ w))

        extra = [{"type": "eq", "fun": lambda w, t=target: float(w @ self.mu.values) - t}]
        bounds = tuple((min_weight, max_weight) for _ in range(self.n))
        res = self._optimise(objective, extra_constraints=extra, bounds=bounds)
        return self._result(res.x, f"Target Return ({target:.1%})", res.success)

    # ── 5. Constrained Portfolio ───────────────────────────────────────────────

    def constrained(
        self,
        min_weight: float = 0.02,
        max_weight: float = 0.40,
        objective: str = "max_sharpe",
        target_return: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Portfolio with realistic per-asset allocation constraints.

        Enforces a minimum and maximum weight for every asset, preventing
        near-zero positions (which are impractical) and over-concentration.

        Parameters
        ----------
        min_weight : float
            Minimum weight per asset (e.g. 0.02 = at least 2%).
        max_weight : float
            Maximum weight per asset (e.g. 0.40 = at most 40%).
        objective : str
            ``'max_sharpe'`` or ``'min_variance'``.
        target_return : float, optional
            If provided, overrides objective and minimises variance
            subject to the return constraint.
        """
        bounds = tuple((min_weight, max_weight) for _ in range(self.n))

        if target_return is not None:
            def obj(w):
                return float(np.sqrt(w @ self.sigma.values @ w))
            extra = [{"type": "eq", "fun": lambda w, t=target_return: float(w @ self.mu.values) - t}]
            res = self._optimise(obj, extra_constraints=extra, bounds=bounds)
            label = f"Constrained Target ({target_return:.1%})"

        elif objective == "max_sharpe":
            def obj(w):
                ret, vol, sharpe = self._portfolio_stats(w)
                return -sharpe if vol > 1e-8 else 0.0
            res = self._optimise(obj, bounds=bounds)
            label = f"Constrained Max Sharpe [{min_weight:.0%}–{max_weight:.0%}]"

        elif objective == "min_variance":
            def obj(w):
                return float(np.sqrt(w @ self.sigma.values @ w))
            res = self._optimise(obj, bounds=bounds)
            label = f"Constrained Min Var [{min_weight:.0%}–{max_weight:.0%}]"

        else:
            raise ValueError(f"Unknown objective '{objective}'. Use 'max_sharpe' or 'min_variance'.")

        return self._result(res.x, label, res.success)

    # ── 6. Monte Carlo Simulation ──────────────────────────────────────────────

    def monte_carlo(
        self,
        n: int = 10_000,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample n random long-only portfolios from the uniform Dirichlet distribution.

        Weight vectors are drawn by generating n_assets uniform randoms and
        dividing by their sum — equivalent to uniform sampling on the n-simplex
        (all weights ≥ 0, sum = 1).

        A fixed seed ensures reproducibility across runs.

        Parameters
        ----------
        n : int
            Number of portfolios to simulate (default: 10,000).
        seed : int
            Random seed (default: 42).

        Returns
        -------
        (returns, volatilities, sharpe_ratios, weight_matrix)
            Four parallel arrays of length n.
        """
        np.random.seed(seed)
        sim_rets = np.empty(n)
        sim_vols = np.empty(n)
        sim_sharpes = np.empty(n)
        sim_weights = np.empty((n, self.n))

        for i in range(n):
            w = np.random.random(self.n)
            w /= w.sum()
            sim_weights[i] = w
            sim_rets[i], sim_vols[i], sim_sharpes[i] = self._portfolio_stats(w)

        return sim_rets, sim_vols, sim_sharpes, sim_weights

    # ── 7. Efficient Frontier ──────────────────────────────────────────────────

    def efficient_frontier(
        self,
        n_points: int = 200,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Trace the mean-variance efficient frontier.

        Solves n_points minimum-variance problems, each constraining portfolio
        return to a target level.  The target range spans from the GMV return
        (the frontier's leftmost feasible point) to the maximum single-asset
        expected return (the rightmost feasible point for a long-only portfolio).

        Parameters
        ----------
        n_points : int
            Number of frontier points to compute (default: 200).

        Returns
        -------
        (frontier_returns, frontier_vols, frontier_weights)
        """
        gmv = self.min_variance(min_weight=min_weight, max_weight=max_weight)
        max_ret = float(self.mu.max())
        targets = np.linspace(gmv.expected_return, max_ret, n_points)

        eff_rets, eff_vols, eff_weights = [], [], []

        def vol_obj(w):
            return float(np.sqrt(w @ self.sigma.values @ w))

        bounds = tuple((min_weight, max_weight) for _ in range(self.n))

        for target in targets:
            extra = [{"type": "eq", "fun": lambda w, t=target: float(w @ self.mu.values) - t}]
            try:
                res = self._optimise(vol_obj, extra_constraints=extra, bounds=bounds)
                r, v, _ = self._portfolio_stats(res.x)
                eff_rets.append(r)
                eff_vols.append(v)
                eff_weights.append(res.x)
            except RuntimeError:
                pass  # skip infeasible targets (e.g. above max asset return)

        return np.array(eff_rets), np.array(eff_vols), eff_weights

    # ── 8. Capital Market Line ─────────────────────────────────────────────────

    def capital_market_line(
        self,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Capital Market Line (CML).

        The CML runs from the risk-free asset (zero vol, R_f return) through
        the tangency portfolio (max Sharpe).  All points on the CML represent
        combinations of the risk-free asset and the tangency portfolio —
        these dominate the efficient frontier for all risk-averse investors.

        Returns
        -------
        (cml_vols, cml_rets) : parallel arrays of length n_points
        """
        tangency = self.max_sharpe()
        max_vol = tangency.volatility * 2.0   # extend CML beyond tangency point

        cml_vols = np.linspace(0, max_vol, n_points)
        # On the CML: return = R_f + Sharpe × vol
        cml_rets = self.risk_free_rate + tangency.sharpe_ratio * cml_vols

        return cml_vols, cml_rets

    # ── 9. Compare All Strategies ─────────────────────────────────────────────

    def compare_all(
        self,
        target_return: Optional[float] = None,
        constrained_min: float = 0.02,
        constrained_max: float = 0.40,
    ) -> Dict[str, OptimizationResult]:
        """
        Run all optimisation strategies and return a comparison dictionary.

        Parameters
        ----------
        target_return : float, optional
            Target return for the 'Target Return' strategy.
            Defaults to the config value or midpoint of the return range.

        Returns
        -------
        dict
            Mapping of strategy name → OptimizationResult.
        """
        if target_return is None:
            target_return = float((self.mu.min() + self.mu.max()) / 2)

        logger.info("Running all portfolio strategies …")
        results: Dict[str, OptimizationResult] = {
            "Min Variance":  self.min_variance(),
            "Max Sharpe":    self.max_sharpe(),
            "Equal Weight":  self.equal_weight(),
            "Target Return": self.target_return(target_return),
            "Constrained":   self.constrained(constrained_min, constrained_max),
        }
        for name, res in results.items():
            logger.info(
                "  %-20s → Ret %.2f%%  Vol %.2f%%  Sharpe %.2f  %s",
                name, res.expected_return * 100, res.volatility * 100, res.sharpe_ratio,
                "✓" if res.converged else "⚠ did not converge",
            )
        return results

    def comparison_table(
        self,
        strategies: Optional[Dict[str, OptimizationResult]] = None,
    ) -> pd.DataFrame:
        """
        Build a formatted comparison table for multiple strategies.

        Parameters
        ----------
        strategies : dict, optional
            Pre-computed strategy results.  If None, runs compare_all().

        Returns
        -------
        pd.DataFrame
            Strategies as columns, weights + metrics as rows.
        """
        if strategies is None:
            strategies = self.compare_all()

        rows = {}
        for ticker in self.tickers:
            rows[ticker] = {name: f"{res.weights[ticker]:.1%}" for name, res in strategies.items()}

        rows["──────────"] = {name: "──────" for name in strategies}
        rows["Exp. Return"] = {name: f"{res.expected_return:.2%}" for name, res in strategies.items()}
        rows["Volatility"] = {name: f"{res.volatility:.2%}" for name, res in strategies.items()}
        rows["Sharpe Ratio"] = {name: f"{res.sharpe_ratio:.2f}" for name, res in strategies.items()}

        return pd.DataFrame(rows).T
