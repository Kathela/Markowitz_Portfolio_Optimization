"""Unit tests for src/data_loader.py — uses mock data to avoid API calls."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader


# ── Fixtures ──────────────────────────────────────────────────────────────────

TICKERS = ["AAPL", "MSFT", "JNJ", "GSPC"]  # ^ already stripped

@pytest.fixture
def config(tmp_path):
    return {
        "data": {
            "tickers": ["AAPL", "MSFT", "JNJ", "^GSPC"],
            "benchmark": "^GSPC",
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "data_dir": str(tmp_path),
            "cache_enabled": True,
        },
        "optimization": {
            "risk_free_rate": 0.05,
            "trading_days": 252,
        },
    }


@pytest.fixture
def sample_prices():
    """Synthetic price DataFrame with 504 trading days."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-03", periods=504, freq="B")
    data = {}
    for i, t in enumerate(TICKERS):
        start_price = 100 + i * 20
        returns = rng.normal(0.0004, 0.015, 504)
        prices = start_price * (1 + returns).cumprod()
        data[t] = prices
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def loader(config, sample_prices, monkeypatch):
    """DataLoader with patched yfinance download."""
    def mock_download(*args, **kwargs):
        # Simulate yfinance MultiIndex return
        cols = pd.MultiIndex.from_product(
            [["Close"], TICKERS],
            names=["Price", "Ticker"],
        )
        flat_prices = sample_prices.copy()
        flat_prices.columns = pd.MultiIndex.from_product(
            [["Close"], flat_prices.columns], names=["Price", "Ticker"]
        )
        return flat_prices

    monkeypatch.setattr("yfinance.download", mock_download)
    ld = DataLoader(config)
    ld.download()
    return ld


# ── DataLoader Initialisation ──────────────────────────────────────────────────

class TestDataLoaderInit:
    def test_creates_data_directory(self, config, tmp_path):
        DataLoader(config)
        assert tmp_path.exists()

    def test_prices_none_before_download(self, config):
        loader = DataLoader(config)
        with pytest.raises(RuntimeError, match="No data loaded"):
            _ = loader.prices


# ── Download & Caching ─────────────────────────────────────────────────────────

class TestDownload:
    def test_prices_dataframe_returned(self, loader):
        assert isinstance(loader.prices, pd.DataFrame)

    def test_correct_shape(self, loader, sample_prices):
        assert loader.prices.shape[0] == len(sample_prices)
        assert loader.prices.shape[1] == len(TICKERS)

    def test_cache_file_created(self, loader, config, tmp_path):
        cache_files = list(tmp_path.glob("prices_*.pkl"))
        assert len(cache_files) == 1

    def test_cache_loaded_on_second_call(self, config, sample_prices, monkeypatch, tmp_path):
        download_count = {"n": 0}

        def mock_download(*args, **kwargs):
            download_count["n"] += 1
            flat_prices = sample_prices.copy()
            flat_prices.columns = pd.MultiIndex.from_product(
                [["Close"], flat_prices.columns]
            )
            return flat_prices

        monkeypatch.setattr("yfinance.download", mock_download)
        ld = DataLoader(config)
        ld.download()   # first call — should download
        ld2 = DataLoader(config)
        ld2.download()  # second call — should use cache
        assert download_count["n"] == 1, "Expected only one actual download"

    def test_no_nan_after_download(self, loader):
        assert not loader.prices.isnull().any().any()


# ── Returns ────────────────────────────────────────────────────────────────────

class TestGetReturns:
    def test_shape(self, loader, sample_prices):
        returns = loader.get_returns()
        assert returns.shape[0] == len(sample_prices) - 1  # one row lost to pct_change

    def test_no_nan(self, loader):
        assert not loader.get_returns().isnull().any().any()

    def test_arithmetic_returns_mean_near_zero(self, loader):
        returns = loader.get_returns()
        for col in returns.columns:
            assert abs(returns[col].mean()) < 0.01, f"Daily mean too high for {col}"

    def test_log_returns_smaller_than_arithmetic(self, loader):
        arith = loader.get_returns(log_returns=False)
        log = loader.get_returns(log_returns=True)
        # For small returns, log ≈ arithmetic; log is slightly smaller
        assert (log.mean() <= arith.mean() + 0.001).all()


# ── Expected Returns ───────────────────────────────────────────────────────────

class TestExpectedReturns:
    def test_returns_series_with_correct_index(self, loader):
        mu = loader.get_expected_returns()
        assert list(mu.index) == TICKERS

    def test_annualised_magnitude(self, loader):
        mu = loader.get_expected_returns()
        # Annualised returns should be roughly in [-50%, +100%] for typical assets
        assert (mu.abs() < 2.0).all()

    def test_invalid_method_raises(self, loader):
        with pytest.raises(ValueError, match="Unknown method"):
            loader.get_expected_returns(method="black_litterman")


# ── Covariance Matrix ──────────────────────────────────────────────────────────

class TestCovariance:
    def test_shape(self, loader):
        sigma = loader.get_covariance()
        n = len(TICKERS)
        assert sigma.shape == (n, n)

    def test_symmetric(self, loader):
        sigma = loader.get_covariance()
        pd.testing.assert_frame_equal(sigma, sigma.T, check_names=False)

    def test_positive_definite(self, loader):
        sigma = loader.get_covariance()
        eigvals = np.linalg.eigvalsh(sigma.values)
        assert (eigvals > -1e-8).all(), "Covariance matrix is not PSD"

    def test_invalid_method_raises(self, loader):
        with pytest.raises(ValueError, match="Unknown covariance method"):
            loader.get_covariance(method="factor_model")


# ── Benchmark ──────────────────────────────────────────────────────────────────

class TestBenchmark:
    def test_benchmark_returns_series(self, loader):
        bench = loader.get_benchmark_returns()
        assert isinstance(bench, pd.Series)

    def test_benchmark_col_in_returns(self, loader):
        bench = loader.get_benchmark_returns()
        returns = loader.get_returns()
        assert bench.name in returns.columns


# ── Stress Test Slice ──────────────────────────────────────────────────────────

class TestStressTestSlice:
    def test_slice_within_range(self, loader):
        sliced = loader.stress_test_slice("2022-06-01", "2022-09-30")
        assert len(sliced) > 0
        assert sliced.index[0] >= pd.Timestamp("2022-06-01")
        assert sliced.index[-1] <= pd.Timestamp("2022-09-30")

    def test_empty_slice_outside_range(self, loader):
        sliced = loader.stress_test_slice("2010-01-01", "2010-12-31")
        assert len(sliced) == 0
