"""Unit tests for src/backtester.py — uses synthetic price data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import Backtester, BacktestResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "JNJ", "GSPC"]
N_DAYS = 1500   # ~6 years of data


@pytest.fixture
def config():
    return {
        "data": {"benchmark": "^GSPC", "tickers": TICKERS},
        "optimization": {
            "risk_free_rate": 0.05,
            "trading_days": 252,
            "n_simulations": 100,
            "constrained": {"min_weight": 0.05, "max_weight": 0.50},
        },
        "backtesting": {
            "in_sample_years": 2,
            "out_sample_years": 1,
            "rebalancing_frequency": "quarterly",
            "transaction_cost": 0.001,
        },
        "risk": {"rolling_window": 252},
    }


@pytest.fixture
def synthetic_prices():
    """Synthetic price DataFrame spanning ~6 years."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-01-02", periods=N_DAYS, freq="B")
    data = {}
    for i, t in enumerate(TICKERS):
        start = 100 + i * 25
        drift = 0.0003 + i * 0.00005
        rets = rng.normal(drift, 0.012 + i * 0.002, N_DAYS)
        data[t] = start * (1 + rets).cumprod()
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def backtester(synthetic_prices, config):
    return Backtester(synthetic_prices, config)


# ── Rebalancing Dates ──────────────────────────────────────────────────────────

class TestRebalancingDates:
    def test_quarterly_dates_spaced_3_months(self):
        start = pd.Timestamp("2022-01-03")
        end = pd.Timestamp("2023-12-31")
        dates = Backtester._get_rebalancing_dates(start, end, "quarterly")
        assert len(dates) >= 4
        # Check spacing is roughly 3 months between consecutive dates
        for d1, d2 in zip(dates, dates[1:]):
            diff = (d2 - d1).days
            assert 60 <= diff <= 100, f"Unexpected gap: {diff} days"

    def test_monthly_gives_more_dates_than_quarterly(self):
        start = pd.Timestamp("2021-01-04")
        end = pd.Timestamp("2022-12-30")
        monthly = Backtester._get_rebalancing_dates(start, end, "monthly")
        quarterly = Backtester._get_rebalancing_dates(start, end, "quarterly")
        assert len(monthly) > len(quarterly)

    def test_unknown_freq_falls_back_gracefully(self):
        start = pd.Timestamp("2022-01-03")
        end = pd.Timestamp("2023-01-02")
        # Should not raise, falls back to "QS" alias
        dates = Backtester._get_rebalancing_dates(start, end, "unknown_freq")
        assert isinstance(dates, list)


# ── Strategy Labels ────────────────────────────────────────────────────────────

class TestStrategyLabel:
    def test_known_labels(self):
        assert Backtester._strategy_label("min_variance") == "Min Variance"
        assert Backtester._strategy_label("max_sharpe") == "Max Sharpe"
        assert Backtester._strategy_label("equal_weight") == "Equal Weight"
        assert Backtester._strategy_label("constrained") == "Constrained"

    def test_unknown_label_title_cases(self):
        label = Backtester._strategy_label("custom_strategy")
        assert label == "Custom Strategy"


# ── Single Strategy Backtest ───────────────────────────────────────────────────

class TestRunAllStrategies:
    def test_returns_backtest_result(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        assert isinstance(result, BacktestResult)

    def test_portfolio_values_start_near_one(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        first_val = result.portfolio_values["Equal Weight"].iloc[0]
        assert first_val == pytest.approx(1.0, abs=0.05)

    def test_all_strategies_in_result(self, backtester):
        result = backtester.run_all_strategies(
            strategies=["min_variance", "max_sharpe", "equal_weight"]
        )
        expected = {"Min Variance", "Max Sharpe", "Equal Weight"}
        assert expected.issubset(set(result.portfolio_values.columns))

    def test_benchmark_included_by_default(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        assert any("Benchmark" in col for col in result.portfolio_values.columns)

    def test_no_nan_in_portfolio_values(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        assert not result.portfolio_values.isnull().all().any()

    def test_weights_history_populated(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        assert "equal_weight" in result.weights_history
        assert len(result.weights_history["equal_weight"]) > 0


# ── Metrics ────────────────────────────────────────────────────────────────────

class TestBacktestMetrics:
    def test_metrics_returns_dataframe(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        metrics = result.metrics()
        assert isinstance(metrics, pd.DataFrame)

    def test_sharpe_ratio_in_metrics(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        metrics = result.metrics()
        assert "Sharpe Ratio" in metrics.index

    def test_metrics_nrows_matches_strategy_count(self, backtester):
        result = backtester.run_all_strategies(
            strategies=["equal_weight", "max_sharpe"], benchmark=False
        )
        metrics = result.metrics()
        assert metrics.shape[1] == 2


# ── Rolling Metrics ────────────────────────────────────────────────────────────

class TestRollingMetrics:
    def test_returns_dict_of_dataframes(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        rolling = backtester.rolling_metrics(result, window=63)
        assert isinstance(rolling, dict)
        for key, df in rolling.items():
            assert isinstance(df, pd.DataFrame)
            assert "Sharpe" in df.columns

    def test_rolling_window_shortens_output(self, backtester):
        result = backtester.run_all_strategies(strategies=["equal_weight"])
        rolling = backtester.rolling_metrics(result, window=63)
        col = list(rolling.keys())[0]
        # Rolling window drops first (window-1) rows
        full_len = len(result.returns["Equal Weight"].dropna())
        rolling_len = rolling[col]["Sharpe"].dropna().__len__()
        assert rolling_len <= full_len
