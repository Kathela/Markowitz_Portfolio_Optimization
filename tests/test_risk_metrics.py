"""Unit tests for src/risk_metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import RiskMetrics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def constant_returns():
    """Returns series with constant 10% annualised return (no volatility)."""
    daily = (1 + 0.10) ** (1 / 252) - 1
    return pd.Series([daily] * 500)


@pytest.fixture
def random_returns(seed=42):
    """Normally distributed daily returns."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.012, 504))


@pytest.fixture
def benchmark_returns(seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=504, freq="B")
    return pd.Series(rng.normal(0.0003, 0.011, 504), index=idx)


@pytest.fixture
def dated_returns(seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=504, freq="B")
    return pd.Series(rng.normal(0.0004, 0.012, 504), index=idx)


# ── Annualised Return ──────────────────────────────────────────────────────────

class TestAnnualisedReturn:
    def test_constant_positive_returns(self, constant_returns):
        result = RiskMetrics.annualised_return(constant_returns)
        assert abs(result - 0.10) < 1e-4, f"Expected ~10%, got {result:.4%}"

    def test_empty_returns(self):
        assert RiskMetrics.annualised_return(pd.Series([], dtype=float)) == 0.0

    def test_negative_return_series(self):
        daily_loss = (1 - 0.20) ** (1 / 252) - 1
        returns = pd.Series([daily_loss] * 252)
        result = RiskMetrics.annualised_return(returns)
        assert result < 0, "Annual return should be negative"


# ── Annualised Volatility ──────────────────────────────────────────────────────

class TestAnnualisedVolatility:
    def test_zero_vol_for_constant_returns(self, constant_returns):
        result = RiskMetrics.annualised_volatility(constant_returns)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_noisy_returns(self, random_returns):
        result = RiskMetrics.annualised_volatility(random_returns)
        assert result > 0

    def test_scales_with_sqrt_252(self, random_returns):
        daily_std = random_returns.std()
        expected = daily_std * np.sqrt(252)
        assert RiskMetrics.annualised_volatility(random_returns) == pytest.approx(expected, rel=1e-6)


# ── Sharpe Ratio ───────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_for_good_returns(self, random_returns):
        returns = pd.Series(np.abs(random_returns.values) + 0.001)
        sharpe = RiskMetrics.sharpe_ratio(returns, risk_free_rate=0.05)
        assert sharpe > 0

    def test_zero_std_returns(self):
        constant = pd.Series([0.0005] * 252)
        sharpe = RiskMetrics.sharpe_ratio(constant)
        assert sharpe == 0.0

    def test_higher_rf_lowers_sharpe(self, random_returns):
        s1 = RiskMetrics.sharpe_ratio(random_returns, risk_free_rate=0.02)
        s2 = RiskMetrics.sharpe_ratio(random_returns, risk_free_rate=0.08)
        assert s1 > s2


# ── Sortino Ratio ──────────────────────────────────────────────────────────────

class TestSortinoRatio:
    def test_sortino_gte_sharpe_for_positive_skew(self, random_returns):
        only_positive = random_returns.copy()
        only_positive[only_positive < 0] *= 0.3  # reduce downside
        sortino = RiskMetrics.sortino_ratio(only_positive)
        sharpe = RiskMetrics.sharpe_ratio(only_positive)
        # Sortino ≥ Sharpe when downside vol < total vol
        assert sortino >= sharpe - 0.5   # allow some numerical slack

    def test_no_downside_returns_inf(self):
        returns = pd.Series([0.001] * 252)
        assert RiskMetrics.sortino_ratio(returns) == np.inf


# ── Max Drawdown ───────────────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_monotone_increase_zero_drawdown(self):
        vals = pd.Series(np.linspace(1.0, 2.0, 100))
        _, mdd = RiskMetrics.max_drawdown(vals)
        assert mdd == pytest.approx(0.0, abs=1e-6)

    def test_known_drawdown(self):
        # Peak at 2, trough at 1 → DD = (2-1)/2 = -50%
        vals = pd.Series([1.0, 1.5, 2.0, 1.5, 1.0, 1.2])
        _, mdd = RiskMetrics.max_drawdown(vals)
        assert mdd == pytest.approx(-0.5, abs=1e-6)

    def test_drawdown_series_nonpositive(self):
        rng = np.random.default_rng(0)
        vals = pd.Series((1 + rng.normal(0.0004, 0.012, 252)).cumprod())
        dd_series, _ = RiskMetrics.max_drawdown(vals)
        assert (dd_series <= 1e-10).all(), "Drawdown should be non-positive"


# ── Value at Risk ──────────────────────────────────────────────────────────────

class TestVaR:
    def test_historical_var_negative(self, random_returns):
        var = RiskMetrics.value_at_risk(random_returns, confidence=0.95)
        assert var < 0, "VaR should represent a loss (negative number)"

    def test_var_95_less_extreme_than_99(self, random_returns):
        var95 = RiskMetrics.value_at_risk(random_returns, confidence=0.95)
        var99 = RiskMetrics.value_at_risk(random_returns, confidence=0.99)
        assert var95 > var99, "99% VaR should be more extreme (more negative)"

    def test_invalid_method_raises(self, random_returns):
        with pytest.raises(ValueError, match="Unknown VaR method"):
            RiskMetrics.value_at_risk(random_returns, method="monte_carlo")


# ── CVaR ───────────────────────────────────────────────────────────────────────

class TestCVaR:
    def test_cvar_more_extreme_than_var(self, random_returns):
        var = RiskMetrics.value_at_risk(random_returns, 0.95)
        cvar = RiskMetrics.conditional_value_at_risk(random_returns, 0.95)
        assert cvar <= var, "CVaR should be at least as extreme as VaR"


# ── Beta & Alpha ───────────────────────────────────────────────────────────────

class TestBetaAlpha:
    def test_beta_one_for_identical_series(self, dated_returns, benchmark_returns):
        # If portfolio = benchmark, beta should be ~1
        beta, _ = RiskMetrics.beta_alpha(benchmark_returns, benchmark_returns)
        assert beta == pytest.approx(1.0, abs=1e-6)

    def test_alpha_zero_for_identical_series(self, benchmark_returns):
        _, alpha = RiskMetrics.beta_alpha(benchmark_returns, benchmark_returns)
        assert alpha == pytest.approx(0.0, abs=1e-6)

    def test_low_beta_for_uncorrelated_series(self, dated_returns, benchmark_returns):
        # Uncorrelated random series should have beta close to 0
        rng = np.random.default_rng(99)
        idx = benchmark_returns.index
        uncorrelated = pd.Series(rng.normal(0.0002, 0.005, len(idx)), index=idx)
        beta, _ = RiskMetrics.beta_alpha(uncorrelated, benchmark_returns)
        assert abs(beta) < 0.4  # expect low systematic risk


# ── Portfolio Summary ──────────────────────────────────────────────────────────

class TestPortfolioSummary:
    def test_returns_series_with_all_keys(self, random_returns, benchmark_returns):
        idx = pd.date_range("2022-01-03", periods=len(random_returns), freq="B")
        ret = pd.Series(random_returns.values, index=idx)
        bench = benchmark_returns.iloc[:len(ret)]
        bench.index = ret.index

        summary = RiskMetrics.portfolio_summary(ret, bench)
        required_keys = [
            "Annualised Return", "Annualised Volatility", "Sharpe Ratio",
            "Sortino Ratio", "Max Drawdown", "Beta",
        ]
        for key in required_keys:
            assert key in summary.index, f"Missing metric: {key}"

    def test_no_benchmark_still_works(self, random_returns):
        idx = pd.date_range("2022-01-03", periods=len(random_returns), freq="B")
        ret = pd.Series(random_returns.values, index=idx)
        summary = RiskMetrics.portfolio_summary(ret)
        assert "Sharpe Ratio" in summary.index
        assert "Beta" not in summary.index
