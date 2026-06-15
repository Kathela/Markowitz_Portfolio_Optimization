"""
Data Acquisition and Preprocessing
=====================================
Downloads, caches, and preprocesses historical financial price data
from Yahoo Finance.  Provides clean return series, expected returns,
and covariance matrix estimates for portfolio optimisation.

Features
--------
- Adjusted-close prices via yfinance (split + dividend adjusted)
- On-disk pickle caching to avoid repeated API calls
- Robust missing-data handling (forward-fill → back-fill → drop)
- Daily and annualised return calculation
- Covariance estimation: sample, Ledoit-Wolf, or EWMA
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252


class DataLoader:
    """
    Downloads and preprocesses historical financial price data.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from ``config.yaml``.

    Examples
    --------
    >>> loader = DataLoader(config)
    >>> prices = loader.download()
    >>> returns = loader.get_returns()
    >>> mu, sigma = loader.get_expected_returns(), loader.get_covariance()
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.data_dir = Path(config["data"]["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trading_days: int = config["optimization"]["trading_days"]
        self.cache_enabled: bool = config["data"].get("cache_enabled", True)

        # Populated by download()
        self._prices: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None
        self._tickers: List[str] = []
        self._benchmark_ticker: str = config["data"].get("benchmark", "^GSPC")

    # ── Public Interface ───────────────────────────────────────────────────────

    def download(
        self,
        tickers: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Download adjusted-close prices for the given tickers.

        Results are cached on disk and reused on subsequent calls unless
        ``force_download=True``.  Prices are split- and dividend-adjusted
        (total-return series) via ``auto_adjust=True``.

        Parameters
        ----------
        tickers : list of str, optional
            Ticker symbols.  Defaults to ``config.data.tickers``.
        start : str, optional
            Start date in ``'YYYY-MM-DD'`` format.  Defaults to config value.
        end : str, optional
            End date.  Defaults to today.
        force_download : bool
            If True, bypass the cache and re-download.

        Returns
        -------
        pd.DataFrame
            Daily adjusted-close prices; columns = tickers (cleaned names).
        """
        tickers = tickers or self.config["data"]["tickers"]
        start = start or self.config["data"]["start_date"]
        end = end or datetime.today().strftime("%Y-%m-%d")

        cache_key = self._cache_key(tickers, start, end)

        if self.cache_enabled and not force_download:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.info("Loaded price data from cache (%s).", cache_key[:8])
                self._prices = cached
                self._tickers = list(cached.columns)
                return self._prices

        logger.info("Downloading price data for %d tickers (%s → %s) …", len(tickers), start, end)

        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        prices = self._extract_close(raw, tickers)
        prices = self._clean_prices(prices)

        if prices.empty:
            raise ValueError("No price data downloaded.  Check tickers and date range.")

        if self.cache_enabled:
            self._save_cache(cache_key, prices)

        self._prices = prices
        self._tickers = list(prices.columns)
        self._returns = None  # invalidate cached returns
        logger.info(
            "Downloaded %d trading days for %d assets (%s → %s).",
            len(prices), prices.shape[1],
            prices.index[0].date(), prices.index[-1].date(),
        )
        return prices

    @property
    def prices(self) -> pd.DataFrame:
        """Return the loaded price DataFrame; raises if not yet downloaded."""
        if self._prices is None:
            raise RuntimeError("No data loaded.  Call download() first.")
        return self._prices

    def get_returns(self, log_returns: bool = False) -> pd.DataFrame:
        """
        Calculate daily return series from adjusted-close prices.

        Arithmetic returns are used by default because portfolio return is
        exactly a weighted sum of arithmetic asset returns.

        Parameters
        ----------
        log_returns : bool
            If True, return log returns ln(P_t / P_{t-1}) instead.
            Log returns have nicer statistical properties (time-additive)
            but are less appropriate for portfolio construction.

        Returns
        -------
        pd.DataFrame
            Daily returns, same shape and columns as prices minus first row.
        """
        if self._returns is None or log_returns:
            p = self.prices
            if log_returns:
                self._returns = np.log(p / p.shift(1)).dropna()
            else:
                self._returns = p.pct_change().dropna()
        return self._returns

    def get_expected_returns(
        self,
        method: str = "mean_historical",
        ewm_halflife: int = 126,
    ) -> pd.Series:
        """
        Estimate annualised expected returns for each asset.

        Parameters
        ----------
        method : str
            ``'mean_historical'``  — simple historical mean × 252.
            ``'ewm'``              — exponentially weighted mean (recent data
                                     weighted more heavily).
        ewm_halflife : int
            Half-life in trading days for the EWM estimator (default: 126 ≈ 6 months).

        Returns
        -------
        pd.Series
            Annualised expected return per asset.
        """
        returns = self.get_returns()
        if method == "mean_historical":
            return returns.mean() * self.trading_days
        elif method == "ewm":
            return returns.ewm(halflife=ewm_halflife).mean().iloc[-1] * self.trading_days
        else:
            raise ValueError(f"Unknown method '{method}'.  Use 'mean_historical' or 'ewm'.")

    def get_covariance(
        self,
        method: str = "sample",
        ewm_halflife: int = 126,
    ) -> pd.DataFrame:
        """
        Estimate the annualised covariance matrix.

        Parameters
        ----------
        method : str
            ``'sample'``        — standard sample covariance × 252.
            ``'ledoit_wolf'``   — Ledoit-Wolf shrinkage (reduces estimation
                                   error; recommended for small samples).
            ``'ewm'``           — Exponentially weighted covariance (recent
                                   data weighted more heavily; good for
                                   capturing changing correlations).
        ewm_halflife : int
            Half-life in days for EWM (default: 126 ≈ 6 months).

        Returns
        -------
        pd.DataFrame
            Annualised covariance matrix (assets × assets).
        """
        returns = self.get_returns()

        if method == "sample":
            return returns.cov() * self.trading_days

        elif method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(returns.values)
                cov_matrix = pd.DataFrame(
                    lw.covariance_ * self.trading_days,
                    index=returns.columns,
                    columns=returns.columns,
                )
                logger.info("Using Ledoit-Wolf shrinkage covariance (shrinkage=%.4f).", lw.shrinkage_)
                return cov_matrix
            except ImportError:
                logger.warning("scikit-learn not available.  Falling back to sample covariance.")
                return returns.cov() * self.trading_days

        elif method == "ewm":
            ewm_cov = returns.ewm(halflife=ewm_halflife).cov().iloc[-len(returns.columns):]
            return ewm_cov * self.trading_days

        else:
            raise ValueError(f"Unknown covariance method '{method}'.  Use 'sample', 'ledoit_wolf', or 'ewm'.")

    def get_benchmark_returns(self) -> pd.Series:
        """
        Return daily returns for the benchmark (default: S&P 500 = ^GSPC).

        The benchmark must be included in the downloaded tickers.

        Returns
        -------
        pd.Series
            Daily benchmark returns aligned to the asset return dates.
        """
        returns = self.get_returns()
        bench_clean = self._benchmark_ticker.replace("^", "")
        if bench_clean not in returns.columns:
            raise ValueError(
                f"Benchmark '{self._benchmark_ticker}' not found in downloaded data.  "
                f"Add it to config.data.tickers."
            )
        return returns[bench_clean]

    def get_asset_returns(self) -> pd.DataFrame:
        """Return returns excluding the benchmark column."""
        returns = self.get_returns()
        bench_clean = self._benchmark_ticker.replace("^", "")
        return returns.drop(columns=[bench_clean], errors="ignore")

    def stress_test_slice(self, start: str, end: str) -> pd.DataFrame:
        """
        Extract a date-range slice of returns for stress testing.

        Parameters
        ----------
        start, end : str
            Date strings in ``'YYYY-MM-DD'`` format.

        Returns
        -------
        pd.DataFrame
            Returns for the specified crisis period.
        """
        returns = self.get_returns()
        mask = (returns.index >= start) & (returns.index <= end)
        if not mask.any():
            logger.warning("No data found for stress period %s → %s.", start, end)
        return returns.loc[mask]

    def summary(self) -> pd.DataFrame:
        """Print a concise summary table of downloaded asset statistics."""
        returns = self.get_returns()
        mu = self.get_expected_returns()
        sigma_diag = np.sqrt(np.diag(self.get_covariance().values))

        from .risk_metrics import RiskMetrics
        sharpes = {
            col: RiskMetrics.sharpe_ratio(returns[col], trading_days=self.trading_days)
            for col in returns.columns
        }

        return pd.DataFrame({
            "Expected Return (ann.)": mu.map("{:.2%}".format),
            "Volatility (ann.)":      pd.Series(sigma_diag, index=returns.columns).map("{:.2%}".format),
            "Sharpe Ratio":           pd.Series(sharpes).map("{:.2f}".format),
        })

    # ── Private Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Extract the Close column from a potentially MultiIndex yfinance DataFrame."""
        if isinstance(raw.columns, pd.MultiIndex):
            # yfinance 0.2+ returns (Metric, Ticker) MultiIndex
            prices = raw["Close"].copy()
        else:
            prices = raw.copy()

        # Standardise column names: remove '^' prefix (e.g. '^GSPC' → 'GSPC')
        prices.columns = [str(c).replace("^", "") for c in prices.columns]
        return prices

    @staticmethod
    def _clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values and ensure price series are monotone.

        Strategy:
          1. Forward-fill: propagate last known price into gaps
             (e.g. when GLD trades on a day equities are closed).
          2. Back-fill: handle leading NaNs for assets with later start dates.
          3. Drop: remove any remaining NaN rows (very early dates).
        """
        cleaned = prices.ffill().bfill().dropna()
        n_dropped = len(prices) - len(cleaned)
        if n_dropped > 0:
            logger.info("Dropped %d rows with missing data.", n_dropped)
        return cleaned

    def _cache_key(self, tickers: List[str], start: str, end: str) -> str:
        """Deterministic cache key based on tickers and date range."""
        content = "|".join(sorted(tickers)) + f"|{start}|{end}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.data_dir / f"prices_{key[:12]}.pkl"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        path = self._cache_path(key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("Cache read failed (%s); re-downloading.", e)
        return None

    def _save_cache(self, key: str, data: pd.DataFrame) -> None:
        path = self._cache_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("Cached prices to %s.", path)
        except Exception as e:
            logger.warning("Cache write failed: %s", e)
