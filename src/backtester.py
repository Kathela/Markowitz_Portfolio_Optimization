"""
Portfolio Backtesting Framework
=================================
Walk-forward backtesting engine that evaluates how optimised portfolios
would have performed historically, with realistic rolling rebalancing and
transaction cost modelling.

Methodology
-----------
- In-sample window  : fit the optimisation model on historical data
- Out-of-sample     : apply weights to the immediately following period
- Roll forward      : shift both windows by the rebalancing frequency
- Transaction costs : deduct turnover × cost_per_dollar from portfolio value

Benchmark comparison is built in (S&P 500 by default).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .portfolio_optimizer import OptimizationResult, PortfolioOptimizer
from .risk_metrics import RiskMetrics

logger = logging.getLogger(__name__)


# ── Result Container ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """
    Container for backtest output.

    Attributes
    ----------
    portfolio_values : pd.DataFrame
        Cumulative portfolio value (starting at 1.0) for each strategy.
    returns : pd.DataFrame
        Daily portfolio returns for each strategy.
    weights_history : dict
        Rebalancing dates → weight vector for each strategy.
    rebalancing_dates : list
        Dates on which the portfolio was rebalanced.
    strategy_name : str
        Name of the backtested strategy.
    """
    portfolio_values: pd.DataFrame
    returns: pd.DataFrame
    weights_history: Dict[str, pd.DataFrame]
    rebalancing_dates: List[pd.Timestamp]
    strategy_name: str

    def metrics(
        self,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Compute the full metric table for all strategies in this backtest."""
        return RiskMetrics.compare_portfolios(
            {col: self.returns[col].dropna() for col in self.returns.columns},
            benchmark_returns=benchmark_returns,
            risk_free_rate=risk_free_rate,
        )


# ── Backtester ─────────────────────────────────────────────────────────────────

class Backtester:
    """
    Walk-forward portfolio backtesting engine.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price history (adjusted close) with assets as columns.
    config : dict
        Configuration dictionary from ``config.yaml``.

    Examples
    --------
    >>> bt = Backtester(prices, config)
    >>> result = bt.run_all_strategies()
    >>> print(result.metrics())
    """

    _FREQ_MAP: Dict[str, str] = {
        "monthly":   "MS",
        "quarterly": "QS",
        "annually":  "AS",
        "weekly":    "W-MON",
    }

    def __init__(self, prices: pd.DataFrame, config: dict) -> None:
        self.prices = prices
        self.config = config
        self.risk_free_rate: float = config["optimization"]["risk_free_rate"]
        self.trading_days: int = config["optimization"]["trading_days"]
        self.transaction_cost: float = config["backtesting"]["transaction_cost"]
        self.in_sample_years: int = config["backtesting"]["in_sample_years"]
        self.rebalancing_freq: str = config["backtesting"]["rebalancing_frequency"]

        # Identify benchmark column (strip ^ from config ticker)
        bench_raw = config["data"]["benchmark"]
        self.benchmark_col: str = bench_raw.replace("^", "")

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_all_strategies(
        self,
        strategies: Optional[List[str]] = None,
        benchmark: bool = True,
    ) -> BacktestResult:
        """
        Backtest multiple strategies over the full price history.

        Each strategy is re-optimised at every rebalancing date using
        the preceding ``in_sample_years`` of data.  The resulting weights
        are then applied to the next rebalancing period.

        Parameters
        ----------
        strategies : list of str, optional
            Strategies to include.  Options:
            ``'min_variance'``, ``'max_sharpe'``, ``'equal_weight'``,
            ``'constrained'``.
            Defaults to all four.
        benchmark : bool
            Whether to include the benchmark buy-and-hold in the comparison.

        Returns
        -------
        BacktestResult
        """
        strategies = strategies or ["min_variance", "max_sharpe", "equal_weight", "constrained"]

        in_sample_td = int(self.in_sample_years * self.trading_days)
        returns = self.prices.pct_change().dropna()

        # Exclude benchmark from asset universe
        asset_cols = [c for c in returns.columns if c != self.benchmark_col]
        rebalancing_dates = self._get_rebalancing_dates(
            returns.index[in_sample_td], returns.index[-1], self.rebalancing_freq
        )

        logger.info(
            "Backtesting %d strategies over %d rebalancing periods …",
            len(strategies), len(rebalancing_dates),
        )

        strategy_portfolio_values: Dict[str, pd.Series] = {}
        strategy_returns: Dict[str, pd.Series] = {}
        weights_history: Dict[str, List[Tuple[pd.Timestamp, pd.Series]]] = {s: [] for s in strategies}

        for strategy in strategies:
            port_vals, port_rets, wt_hist = self._run_strategy(
                strategy, returns, asset_cols, in_sample_td, rebalancing_dates
            )
            strategy_portfolio_values[self._strategy_label(strategy)] = port_vals
            strategy_returns[self._strategy_label(strategy)] = port_rets
            weights_history[strategy] = wt_hist

        if benchmark and self.benchmark_col in returns.columns:
            bench_rets = returns[self.benchmark_col].loc[list(strategy_returns.values())[0].index]
            bench_vals = (1 + bench_rets).cumprod()
            bench_vals = bench_vals / bench_vals.iloc[0]
            strategy_portfolio_values["Benchmark (S&P 500)"] = bench_vals
            strategy_returns["Benchmark (S&P 500)"] = bench_rets

        # Align all series to the same date index
        port_val_df = pd.DataFrame(strategy_portfolio_values).dropna(how="all")
        port_ret_df = pd.DataFrame(strategy_returns).dropna(how="all")

        # Convert weights history to DataFrames
        wh_dfs: Dict[str, pd.DataFrame] = {}
        for strat, hist in weights_history.items():
            if hist:
                dates, weights = zip(*hist)
                wh_dfs[strat] = pd.DataFrame(
                    list(weights), index=list(dates), columns=asset_cols
                )

        return BacktestResult(
            portfolio_values=port_val_df,
            returns=port_ret_df,
            weights_history=wh_dfs,
            rebalancing_dates=rebalancing_dates,
            strategy_name="Multi-Strategy Backtest",
        )

    def rolling_metrics(
        self,
        backtest_result: BacktestResult,
        window: int = 252,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute rolling performance metrics for each backtested strategy.

        Parameters
        ----------
        window : int
            Rolling window in trading days (default: 252 = 1 year).

        Returns
        -------
        dict
            Strategy name → DataFrame with columns ['Return', 'Volatility', 'Sharpe'].
        """
        return {
            col: RiskMetrics.rolling_metrics(
                backtest_result.returns[col].dropna(),
                window=window,
                risk_free_rate=self.risk_free_rate,
            )
            for col in backtest_result.returns.columns
        }

    # ── Private Methods ────────────────────────────────────────────────────────

    def _run_strategy(
        self,
        strategy: str,
        returns: pd.DataFrame,
        asset_cols: List[str],
        in_sample_td: int,
        rebalancing_dates: List[pd.Timestamp],
    ) -> Tuple[pd.Series, pd.Series, List[Tuple[pd.Timestamp, pd.Series]]]:
        """
        Walk-forward backtest for a single strategy.

        For each rebalancing date:
          1. Fit optimiser on the preceding in_sample_td trading days.
          2. Hold resulting weights until the next rebalancing date.
          3. Deduct transaction costs from portfolio value on rebalance.
        """
        asset_returns = returns[asset_cols]
        portfolio_values: List[float] = [1.0]
        portfolio_returns_list: List[float] = []
        dates_list: List[pd.Timestamp] = []
        weights_history: List[Tuple[pd.Timestamp, pd.Series]] = []

        current_weights: Optional[np.ndarray] = None
        portfolio_value = 1.0

        for i, reb_date in enumerate(rebalancing_dates):
            # --- In-sample window ---
            in_sample_end = reb_date
            in_sample_mask = asset_returns.index < in_sample_end
            in_sample_data = asset_returns.loc[in_sample_mask].tail(in_sample_td)

            if len(in_sample_data) < 60:  # need at least 60 days
                logger.debug("Skipping rebalance at %s: insufficient in-sample data.", reb_date.date())
                continue

            # --- Optimise ---
            mu = in_sample_data.mean() * self.trading_days
            sigma = in_sample_data.cov() * self.trading_days
            optimizer = PortfolioOptimizer(mu, sigma, self.risk_free_rate, self.trading_days)

            try:
                result = self._get_strategy_result(optimizer, strategy)
                new_weights = result.weights.values
            except Exception as e:
                logger.warning("Optimisation failed at %s (%s). Using equal weights.", reb_date.date(), e)
                new_weights = np.ones(len(asset_cols)) / len(asset_cols)

            # --- Transaction costs ---
            if current_weights is not None:
                turnover = np.abs(new_weights - current_weights).sum()
                tc = turnover * self.transaction_cost
                portfolio_value *= (1 - tc)
                logger.debug("  Rebalance %s: turnover=%.1f%%  TC=%.4f%%",
                             reb_date.date(), turnover * 100, tc * 100)

            weights_history.append((reb_date, pd.Series(new_weights, index=asset_cols)))
            current_weights = new_weights

            # --- Out-of-sample period ---
            next_reb = rebalancing_dates[i + 1] if i + 1 < len(rebalancing_dates) else asset_returns.index[-1]
            oos_mask = (asset_returns.index >= reb_date) & (asset_returns.index < next_reb)
            oos_data = asset_returns.loc[oos_mask]

            if oos_data.empty:
                continue

            # Apply constant weights during out-of-sample period
            period_rets = oos_data @ current_weights
            for date, r in period_rets.items():
                portfolio_value *= (1 + r)
                portfolio_values.append(portfolio_value)
                portfolio_returns_list.append(r)
                dates_list.append(date)

        if not dates_list:
            raise RuntimeError(f"Strategy '{strategy}' produced no output.  Check date range and in_sample_years.")

        port_vals = pd.Series(portfolio_values[1:], index=dates_list)
        port_vals = port_vals / port_vals.iloc[0]   # normalise to start at 1.0
        port_rets = pd.Series(portfolio_returns_list, index=dates_list)

        logger.info("  %-20s complete.  Final value: %.4f", strategy, port_vals.iloc[-1])
        return port_vals, port_rets, weights_history

    def _get_strategy_result(
        self, optimizer: PortfolioOptimizer, strategy: str
    ) -> OptimizationResult:
        """Dispatch to the correct optimizer method by strategy name."""
        cfg = self.config.get("optimization", {})
        constrained = cfg.get("constrained", {})

        dispatch = {
            "min_variance":  lambda: optimizer.min_variance(),
            "max_sharpe":    lambda: optimizer.max_sharpe(),
            "equal_weight":  lambda: optimizer.equal_weight(),
            "constrained":   lambda: optimizer.constrained(
                min_weight=constrained.get("min_weight", 0.02),
                max_weight=constrained.get("max_weight", 0.40),
            ),
        }
        if strategy not in dispatch:
            raise ValueError(
                f"Unknown strategy '{strategy}'.  "
                f"Choose from: {list(dispatch.keys())}."
            )
        return dispatch[strategy]()

    @staticmethod
    def _get_rebalancing_dates(
        start: pd.Timestamp,
        end: pd.Timestamp,
        freq: str,
    ) -> List[pd.Timestamp]:
        """Generate rebalancing dates between start and end at the given frequency."""
        freq_alias = Backtester._FREQ_MAP.get(freq, "QS")
        dates = pd.date_range(start=start, end=end, freq=freq_alias)
        return [d for d in dates if d >= start]

    @staticmethod
    def _strategy_label(strategy: str) -> str:
        """Convert strategy key to display label."""
        labels = {
            "min_variance": "Min Variance",
            "max_sharpe":   "Max Sharpe",
            "equal_weight": "Equal Weight",
            "constrained":  "Constrained",
        }
        return labels.get(strategy, strategy.replace("_", " ").title())
