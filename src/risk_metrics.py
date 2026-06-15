"""
Risk and Performance Metrics
==============================
Institutional-grade portfolio analytics.  All methods are pure functions
wrapped as static methods — no state, no instantiation required.

Metrics implemented
-------------------
Return metrics      : CAGR, annualised volatility
Risk-adjusted       : Sharpe, Sortino, Calmar
Drawdown            : Maximum Drawdown, drawdown series
Tail risk           : VaR (historical + parametric), CVaR / Expected Shortfall
Market sensitivity  : Beta, Jensen's Alpha, Information Ratio, Tracking Error
Summary             : portfolio_summary(), rolling_metrics()
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


class RiskMetrics:
    """
    Institutional-grade risk and performance metrics.

    Usage
    -----
    All methods are static — call directly on the class::

        sharpe = RiskMetrics.sharpe_ratio(daily_returns, risk_free_rate=0.05)
        metrics = RiskMetrics.portfolio_summary(returns, benchmark_returns)
    """

    # ── Return Metrics ─────────────────────────────────────────────────────────

    @staticmethod
    def annualised_return(
        returns: pd.Series,
        trading_days: int = _TRADING_DAYS,
    ) -> float:
        """
        Compound Annualised Growth Rate (CAGR).

        CAGR = (∏(1 + r_t))^(252 / T) - 1

        Preferred over simple average because it accounts for compounding
        and is unaffected by the choice of time period length.
        """
        if len(returns) == 0:
            return 0.0
        total_return = float((1 + returns).prod())
        n_years = len(returns) / trading_days
        return float(total_return ** (1.0 / n_years) - 1)

    @staticmethod
    def annualised_volatility(
        returns: pd.Series,
        trading_days: int = _TRADING_DAYS,
    ) -> float:
        """
        Annualised standard deviation of daily returns.

        σ_annual = σ_daily × √252

        Assumes i.i.d. returns (variance scales linearly with time).
        """
        return float(returns.std() * np.sqrt(trading_days))

    # ── Risk-Adjusted Return Metrics ───────────────────────────────────────────

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        trading_days: int = _TRADING_DAYS,
    ) -> float:
        """
        Sharpe Ratio (annualised): (R_p - R_f) / σ_p.

        Measures excess return per unit of total risk.
        Benchmarks: > 1.0 good, > 1.5 very good, > 2.0 exceptional.
        """
        rf_daily = risk_free_rate / trading_days
        excess = returns - rf_daily
        if excess.std() < 1e-10:
            return 0.0
        return float((excess.mean() / excess.std()) * np.sqrt(trading_days))

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        trading_days: int = _TRADING_DAYS,
    ) -> float:
        """
        Sortino Ratio (annualised): (R_p - R_f) / σ_downside.

        Only penalises downside (negative) volatility — a fairer measure
        for strategies with positively-skewed return distributions.
        """
        rf_daily = risk_free_rate / trading_days
        excess_ann = (returns.mean() - rf_daily) * trading_days
        downside = returns[returns < rf_daily] - rf_daily
        if len(downside) == 0 or downside.std() == 0:
            return np.inf
        downside_vol = float(downside.std() * np.sqrt(trading_days))
        return float(excess_ann / downside_vol)

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        trading_days: int = _TRADING_DAYS,
    ) -> float:
        """
        Calmar Ratio: CAGR / |Maximum Drawdown|.

        Widely used by CTA / managed-futures funds.  Higher is better.
        Ratios above 0.5 are generally considered acceptable.
        """
        ann_ret = RiskMetrics.annualised_return(returns, trading_days)
        portfolio_values = (1 + returns).cumprod()
        _, max_dd = RiskMetrics.max_drawdown(portfolio_values)
        if max_dd == 0:
            return np.inf
        return float(ann_ret / abs(max_dd))

    # ── Drawdown Analysis ──────────────────────────────────────────────────────

    @staticmethod
    def max_drawdown(
        portfolio_values: pd.Series,
    ) -> Tuple[pd.Series, float]:
        """
        Drawdown series and Maximum Drawdown (MDD).

        Drawdown_t = (Peak_t - Value_t) / Peak_t

        MDD is the largest peak-to-trough decline over the full period.
        Returned as a negative number (e.g. -0.35 = 35% loss from peak).

        Parameters
        ----------
        portfolio_values : pd.Series
            Cumulative portfolio value (not returns).

        Returns
        -------
        (drawdown_series, max_drawdown) : (pd.Series, float)
        """
        rolling_peak = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_peak) / rolling_peak
        return drawdown, float(drawdown.min())

    @staticmethod
    def drawdown_periods(
        portfolio_values: pd.Series,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Identify and rank the top-N drawdown periods by magnitude.

        Returns a DataFrame with start date, trough date, magnitude,
        and recovery date (NaT if not yet recovered) for each period.
        """
        rolling_peak = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_peak) / rolling_peak

        periods: list[dict] = []
        in_dd = False
        start = None

        for date, dd in drawdown.items():
            if dd < 0 and not in_dd:
                in_dd = True
                start = date
                trough_val = dd
                trough_date = date
            elif in_dd and dd < trough_val:
                trough_val = dd
                trough_date = date
            elif in_dd and dd == 0:
                in_dd = False
                periods.append({
                    "Start": start,
                    "Trough": trough_date,
                    "Recovery": date,
                    "Drawdown": trough_val,
                    "Duration (days)": (date - start).days,
                })

        # Ongoing drawdown
        if in_dd:
            periods.append({
                "Start": start,
                "Trough": trough_date,
                "Recovery": pd.NaT,
                "Drawdown": trough_val,
                "Duration (days)": (drawdown.index[-1] - start).days,
            })

        df = pd.DataFrame(periods).sort_values("Drawdown").head(top_n)
        df["Drawdown"] = df["Drawdown"].map("{:.2%}".format)
        return df.reset_index(drop=True)

    # ── Tail Risk ──────────────────────────────────────────────────────────────

    @staticmethod
    def value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Value at Risk (VaR).

        The loss threshold not exceeded with probability (1 - confidence).
        E.g. 5% daily VaR of -2%: only a 5% chance of losing >2% on any day.

        Parameters
        ----------
        method : str
            ``'historical'``  — empirical percentile (non-parametric).
            ``'parametric'``  — normal distribution assumption.

        Returns
        -------
        float
            Negative number (loss convention), e.g. -0.02 for -2% VaR.
        """
        if method == "historical":
            return float(np.percentile(returns, (1 - confidence) * 100))
        elif method == "parametric":
            mu, sigma = returns.mean(), returns.std()
            return float(stats.norm.ppf(1 - confidence, mu, sigma))
        else:
            raise ValueError(f"Unknown VaR method '{method}'. Use 'historical' or 'parametric'.")

    @staticmethod
    def conditional_value_at_risk(
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        Conditional Value at Risk (CVaR) / Expected Shortfall.

        Expected loss given that loss exceeds VaR.  CVaR is a *coherent*
        risk measure (unlike VaR) and better captures tail risk from
        fat-tailed or skewed return distributions.

        Returns the mean of all returns in the worst (1-confidence) tail.
        """
        var = RiskMetrics.value_at_risk(returns, confidence, method="historical")
        tail = returns[returns <= var]
        return float(tail.mean()) if len(tail) > 0 else var

    # ── Market Sensitivity ─────────────────────────────────────────────────────

    @staticmethod
    def beta_alpha(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.05,
        trading_days: int = _TRADING_DAYS,
    ) -> Tuple[float, float]:
        """
        CAPM Beta and Jensen's Alpha (annualised).

        Beta  = Cov(R_p, R_b) / Var(R_b)
              Measures systematic risk: β=1 moves with market, β<1 less volatile.

        Alpha = R_p - [R_f + β(R_b - R_f)]
              Annualised excess return unexplained by market exposure.
              Positive alpha indicates outperformance on a risk-adjusted basis.

        Returns
        -------
        (beta, alpha) : (float, float)
        """
        aligned = pd.concat(
            [portfolio_returns.rename("p"), benchmark_returns.rename("b")],
            axis=1, join="inner",
        ).dropna()

        cov = aligned.cov()
        beta = float(cov.loc["p", "b"] / cov.loc["b", "b"])

        ann_p = RiskMetrics.annualised_return(aligned["p"], trading_days)
        ann_b = RiskMetrics.annualised_return(aligned["b"], trading_days)
        alpha = float(ann_p - (risk_free_rate + beta * (ann_b - risk_free_rate)))

        return beta, alpha

    @staticmethod
    def information_ratio(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        trading_days: int = _TRADING_DAYS,
    ) -> float:
        """
        Information Ratio: active return / tracking error (annualised).

        Measures active management skill.  Benchmarks: > 0.5 good, > 1.0 excellent.
        """
        active = portfolio_returns - benchmark_returns
        te = float(active.std() * np.sqrt(trading_days))
        if te == 0:
            return 0.0
        ann_active = float(active.mean() * trading_days)
        return float(ann_active / te)

    @staticmethod
    def tracking_error(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        trading_days: int = _TRADING_DAYS,
    ) -> float:
        """Annualised standard deviation of active (portfolio minus benchmark) returns."""
        return float((portfolio_returns - benchmark_returns).std() * np.sqrt(trading_days))

    # ── Comprehensive Summary ──────────────────────────────────────────────────

    @staticmethod
    def portfolio_summary(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        portfolio_values: Optional[pd.Series] = None,
        risk_free_rate: float = 0.05,
        trading_days: int = _TRADING_DAYS,
        var_confidence: float = 0.95,
    ) -> pd.Series:
        """
        Full suite of institutional risk metrics in a single call.

        Parameters
        ----------
        returns : pd.Series
            Daily portfolio return series (arithmetic).
        benchmark_returns : pd.Series, optional
            Daily benchmark returns for Beta, Alpha, IR, and TE.
        portfolio_values : pd.Series, optional
            Cumulative portfolio value; derived from *returns* if None.
        risk_free_rate : float
            Annualised risk-free rate used in Sharpe, Sortino, Calmar, Alpha.
        var_confidence : float
            Confidence level for VaR and CVaR (e.g. 0.95 for 95%).

        Returns
        -------
        pd.Series
            Metric name → value, formatted for display.
        """
        if portfolio_values is None:
            portfolio_values = (1 + returns).cumprod()

        _, max_dd = RiskMetrics.max_drawdown(portfolio_values)

        metrics: Dict[str, float] = {
            "Annualised Return":      RiskMetrics.annualised_return(returns, trading_days),
            "Annualised Volatility":  RiskMetrics.annualised_volatility(returns, trading_days),
            "Sharpe Ratio":           RiskMetrics.sharpe_ratio(returns, risk_free_rate, trading_days),
            "Sortino Ratio":          RiskMetrics.sortino_ratio(returns, risk_free_rate, trading_days),
            "Calmar Ratio":           RiskMetrics.calmar_ratio(returns, trading_days),
            "Max Drawdown":           max_dd,
            f"VaR {var_confidence:.0%} (daily)":  RiskMetrics.value_at_risk(returns, var_confidence, "historical"),
            f"CVaR {var_confidence:.0%} (daily)": RiskMetrics.conditional_value_at_risk(returns, var_confidence),
            "Skewness":               float(returns.skew()),
            "Excess Kurtosis":        float(returns.kurt()),
        }

        if benchmark_returns is not None:
            b, a = RiskMetrics.beta_alpha(
                returns, benchmark_returns, risk_free_rate, trading_days
            )
            metrics.update({
                "Beta":              b,
                "Alpha (annualised)": a,
                "Information Ratio": RiskMetrics.information_ratio(returns, benchmark_returns, trading_days),
                "Tracking Error":    RiskMetrics.tracking_error(returns, benchmark_returns, trading_days),
            })

        return pd.Series(metrics)

    @staticmethod
    def rolling_metrics(
        returns: pd.Series,
        window: int = _TRADING_DAYS,
        risk_free_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compute rolling Sharpe ratio, volatility, and return over a moving window.

        Useful for plotting how risk-adjusted performance evolves over time
        and identifying regime changes.

        Parameters
        ----------
        window : int
            Rolling window in trading days (default: 252 = 1 year).

        Returns
        -------
        pd.DataFrame
            Columns: ['Return', 'Volatility', 'Sharpe']
        """
        rf_daily = risk_free_rate / _TRADING_DAYS
        excess = returns - rf_daily

        rolling_vol = returns.rolling(window).std() * np.sqrt(_TRADING_DAYS)
        rolling_mean = returns.rolling(window).mean() * _TRADING_DAYS
        rolling_sharpe = (
            excess.rolling(window).mean()
            / returns.rolling(window).std()
            * np.sqrt(_TRADING_DAYS)
        )

        return pd.DataFrame({
            "Return":     rolling_mean,
            "Volatility": rolling_vol,
            "Sharpe":     rolling_sharpe,
        })

    @staticmethod
    def compare_portfolios(
        returns_dict: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.05,
        trading_days: int = _TRADING_DAYS,
    ) -> pd.DataFrame:
        """
        Compute the full metric table for multiple portfolios side by side.

        Parameters
        ----------
        returns_dict : dict
            Mapping of strategy name → daily return series.

        Returns
        -------
        pd.DataFrame
            Metrics as rows, strategies as columns.
        """
        summaries = {
            name: RiskMetrics.portfolio_summary(
                ret, benchmark_returns,
                risk_free_rate=risk_free_rate,
                trading_days=trading_days,
            )
            for name, ret in returns_dict.items()
        }
        return pd.DataFrame(summaries)
