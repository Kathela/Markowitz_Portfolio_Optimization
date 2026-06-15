"""Unit tests for src/portfolio_optimizer.py."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_optimizer import OptimizationResult, PortfolioOptimizer


# ── Fixtures ──────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "AMZN", "JNJ"]
N = len(TICKERS)


@pytest.fixture
def synthetic_mu():
    """Annualised expected returns for synthetic assets."""
    return pd.Series([0.20, 0.18, 0.22, 0.08], index=TICKERS)


@pytest.fixture
def synthetic_sigma():
    """Positive-definite annualised covariance matrix."""
    # Build via random correlation matrix to ensure PD
    rng = np.random.default_rng(42)
    A = rng.normal(0, 1, (N, N))
    cov_raw = A @ A.T
    # Scale to reasonable volatilities (15–30%)
    vols = np.array([0.25, 0.24, 0.30, 0.15])
    D = np.diag(vols)
    corr = cov_raw / np.sqrt(np.outer(np.diag(cov_raw), np.diag(cov_raw)))
    cov = D @ corr @ D
    return pd.DataFrame(cov, index=TICKERS, columns=TICKERS)


@pytest.fixture
def optimizer(synthetic_mu, synthetic_sigma):
    return PortfolioOptimizer(synthetic_mu, synthetic_sigma, risk_free_rate=0.05)


# ── OptimizationResult ─────────────────────────────────────────────────────────

class TestOptimizationResult:
    def test_repr_contains_method(self, optimizer):
        result = optimizer.equal_weight()
        assert "Equal Weight" in repr(result)

    def test_weights_table_filters_trivial(self, optimizer):
        result = optimizer.min_variance()
        table = result.weights_table()
        assert all(result.weights[idx] > 1e-4 for idx in table.index)


# ── Equal Weight ───────────────────────────────────────────────────────────────

class TestEqualWeight:
    def test_weights_sum_to_one(self, optimizer):
        result = optimizer.equal_weight()
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_all_weights_equal(self, optimizer):
        result = optimizer.equal_weight()
        expected = 1 / N
        assert all(abs(w - expected) < 1e-10 for w in result.weights)

    def test_method_label(self, optimizer):
        result = optimizer.equal_weight()
        assert result.method == "Equal Weight"


# ── Min Variance ───────────────────────────────────────────────────────────────

class TestMinVariance:
    def test_weights_sum_to_one(self, optimizer):
        result = optimizer.min_variance()
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_in_bounds(self, optimizer):
        result = optimizer.min_variance()
        assert (result.weights >= -1e-6).all()
        assert (result.weights <= 1 + 1e-6).all()

    def test_vol_less_than_equal_weight(self, optimizer):
        mv = optimizer.min_variance()
        ew = optimizer.equal_weight()
        # Min variance should have lower or equal volatility
        assert mv.volatility <= ew.volatility + 1e-4

    def test_custom_bounds(self, optimizer):
        result = optimizer.min_variance(min_weight=0.05, max_weight=0.50)
        assert (result.weights >= 0.05 - 1e-5).all()
        assert (result.weights <= 0.50 + 1e-5).all()


# ── Max Sharpe ─────────────────────────────────────────────────────────────────

class TestMaxSharpe:
    def test_weights_sum_to_one(self, optimizer):
        result = optimizer.max_sharpe()
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_sharpe_greater_than_equal_weight(self, optimizer):
        ms = optimizer.max_sharpe()
        ew = optimizer.equal_weight()
        assert ms.sharpe_ratio >= ew.sharpe_ratio - 0.05  # allow tolerance

    def test_return_and_vol_positive(self, optimizer):
        result = optimizer.max_sharpe()
        assert result.expected_return > 0
        assert result.volatility > 0


# ── Target Return ──────────────────────────────────────────────────────────────

class TestTargetReturn:
    def test_return_matches_target(self, optimizer, synthetic_mu):
        target = 0.15
        result = optimizer.target_return(target)
        assert result.expected_return == pytest.approx(target, abs=1e-4)

    def test_infeasible_target_raises(self, optimizer, synthetic_mu):
        with pytest.raises(ValueError, match="feasible range"):
            optimizer.target_return(0.50)  # above max single-asset return

    def test_weights_sum_to_one(self, optimizer):
        result = optimizer.target_return(0.15)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


# ── Constrained ────────────────────────────────────────────────────────────────

class TestConstrained:
    def test_weights_within_bounds(self, optimizer):
        result = optimizer.constrained(min_weight=0.10, max_weight=0.40)
        assert (result.weights >= 0.10 - 1e-5).all()
        assert (result.weights <= 0.40 + 1e-5).all()

    def test_weights_sum_to_one(self, optimizer):
        result = optimizer.constrained(0.05, 0.45)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-5)

    def test_invalid_objective_raises(self, optimizer):
        with pytest.raises(ValueError, match="Unknown objective"):
            optimizer.constrained(objective="risk_parity")


# ── Monte Carlo ────────────────────────────────────────────────────────────────

class TestMonteCarlo:
    def test_output_shapes(self, optimizer):
        n = 500
        rets, vols, sharpes, weights = optimizer.monte_carlo(n=n)
        assert rets.shape == (n,)
        assert vols.shape == (n,)
        assert sharpes.shape == (n,)
        assert weights.shape == (n, N)

    def test_weights_sum_to_one(self, optimizer):
        _, _, _, weights = optimizer.monte_carlo(n=100)
        row_sums = weights.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_all_weights_nonnegative(self, optimizer):
        _, _, _, weights = optimizer.monte_carlo(n=100)
        assert (weights >= 0).all()

    def test_reproducible_with_seed(self, optimizer):
        r1, v1, _, _ = optimizer.monte_carlo(n=200, seed=99)
        r2, v2, _, _ = optimizer.monte_carlo(n=200, seed=99)
        np.testing.assert_array_equal(r1, r2)


# ── Efficient Frontier ─────────────────────────────────────────────────────────

class TestEfficientFrontier:
    def test_returns_increasing(self, optimizer):
        eff_rets, eff_vols, _ = optimizer.efficient_frontier(n_points=20)
        # Frontier returns should be non-decreasing
        diffs = np.diff(eff_rets)
        assert (diffs >= -1e-6).all(), "Frontier returns should be non-decreasing"

    def test_minimum_points_returned(self, optimizer):
        eff_rets, eff_vols, _ = optimizer.efficient_frontier(n_points=30)
        assert len(eff_rets) > 5, "Expected at least 5 frontier points"

    def test_vols_nonnegative(self, optimizer):
        _, eff_vols, _ = optimizer.efficient_frontier(n_points=20)
        assert (np.array(eff_vols) >= 0).all()


# ── Compare All ────────────────────────────────────────────────────────────────

class TestCompareAll:
    def test_all_strategies_present(self, optimizer):
        results = optimizer.compare_all(target_return=0.14)
        expected_keys = {"Min Variance", "Max Sharpe", "Equal Weight", "Target Return", "Constrained"}
        assert set(results.keys()) == expected_keys

    def test_comparison_table_shape(self, optimizer):
        table = optimizer.comparison_table()
        assert "Max Sharpe" in table.columns
        assert "Min Variance" in table.columns
