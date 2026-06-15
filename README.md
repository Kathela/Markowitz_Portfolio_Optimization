# Quant Portfolio Optimization System

> *"The portfolio problem is fundamentally one of hedging against uncertainty."*
> — Harry Markowitz, Nobel Laureate in Economics (1990)

A production-grade implementation of **Modern Portfolio Theory (MPT)** built for Quant Analyst, Investment Analyst, and Risk Analyst portfolio presentations. The system goes far beyond a basic Markowitz implementation — it includes a full backtesting framework, institutional risk metrics, stress testing, and nine professional visualisations.

---

## Business Objective

Asset managers face a fundamental challenge: given N risky assets, how should capital be allocated to maximise return for a given level of risk? This system answers that question systematically using:

- **Quantitative optimisation** — scipy SLSQP solver with analytical gradients
- **Monte Carlo simulation** — visual exploration of the full feasible set
- **Walk-forward backtesting** — realistic out-of-sample performance evaluation
- **Institutional risk metrics** — the same KPIs used at hedge funds and asset managers
- **Stress testing** — performance simulation under historical crisis scenarios

---

## Financial Theory

### Modern Portfolio Theory (Markowitz, 1952)

The core insight is that **portfolio risk depends not just on individual asset volatilities, but on how assets co-vary**. Two assets may each be risky individually, but if they are negatively correlated, combining them reduces portfolio variance more than proportionally to weight:

$$\sigma^2_p = \mathbf{w}^\top \Sigma \mathbf{w} = \sum_i \sum_j w_i w_j \sigma_{ij}$$

This **covariance term** ($\sigma_{ij}$) is the engine of diversification — the "free lunch" Markowitz identified.

### The Efficient Frontier

The efficient frontier is the set of all portfolios for which no other combination of assets delivers:
- Higher expected return **for the same level of risk**, or
- Lower risk **for the same expected return**

Any portfolio below the frontier is *inefficient* — you are accepting too much risk for your return.

### Optimal Portfolio Strategies

| Strategy | Objective | Math |
|----------|-----------|------|
| **Min Variance** | Lowest achievable risk | $\min_w \; \mathbf{w}^\top \Sigma \mathbf{w}$ |
| **Max Sharpe** | Best risk-adjusted return | $\max_w \; \frac{\mathbf{w}^\top \mu - R_f}{\sqrt{\mathbf{w}^\top \Sigma \mathbf{w}}}$ |
| **Equal Weight** | Naive 1/N benchmark | $w_i = 1/N \; \forall i$ |
| **Target Return** | Min risk at fixed return | $\min_w \sigma_p \;\text{s.t.}\; \mathbf{w}^\top \mu = r^*$ |
| **Constrained** | Max Sharpe with bounds | $w_i \in [\underline{w}, \bar{w}]$ |

### Capital Market Line (CML)

The CML extends from the risk-free asset through the tangency portfolio (Max Sharpe). All points on the CML represent combinations of the risk-free asset and the tangency portfolio. This is the set from which all rational risk-averse investors should choose — by mixing the tangency portfolio with the risk-free asset, they can achieve any risk level on the CML with maximum efficiency.

$$\text{CML}: \quad R = R_f + \frac{R_T - R_f}{\sigma_T} \cdot \sigma$$

---

## Key Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Sharpe Ratio** | $(R_p - R_f)/\sigma_p$ | Return per unit of total risk. >1 good, >2 exceptional |
| **Sortino Ratio** | $(R_p - R_f)/\sigma_{\downarrow}$ | Like Sharpe but penalises only downside volatility |
| **Calmar Ratio** | $\text{CAGR} / |\text{MDD}|$ | Return relative to worst peak-to-trough loss |
| **Max Drawdown** | $\max_t (P_t - P_{peak,t})/P_{peak,t}$ | Largest peak-to-trough decline |
| **VaR (95%)** | $Q_{5\%}$ of returns | Maximum expected 1-day loss 95% of the time |
| **CVaR (95%)** | $E[R \mid R \leq \text{VaR}]$ | Expected loss in the worst 5% of days |
| **Beta** | $\text{Cov}(R_p, R_b)/\text{Var}(R_b)$ | Systematic risk vs benchmark (1.0 = market-like) |
| **Jensen's Alpha** | $R_p - [R_f + \beta(R_b - R_f)]$ | Excess return unexplained by market exposure |
| **Information Ratio** | Active Return / Tracking Error | Active management skill (>0.5 good, >1.0 excellent) |

---

## System Architecture

```
portfolio_optimization_system/
│
├── main.py                      ← Full analysis pipeline entry point
├── config.yaml                  ← All parameters in one place
├── requirements.txt             ← Pinned dependencies
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           ← yfinance download, caching, preprocessing
│   ├── portfolio_optimizer.py   ← 5 strategies + MC + frontier + CML
│   ├── risk_metrics.py          ← 12+ institutional metrics (all static)
│   ├── backtester.py            ← Walk-forward engine + rolling rebalance
│   └── visualization.py         ← 9 professional charts (no plt.show)
│
├── tests/
│   ├── test_data_loader.py      ← 20+ tests with mock yfinance
│   ├── test_portfolio_optimizer.py  ← 30+ tests on all strategies
│   ├── test_risk_metrics.py     ← 25+ tests on all metrics
│   └── test_backtester.py       ← 20+ tests on backtest engine
│
├── notebooks/
│   └── portfolio_analysis.ipynb ← Interactive end-to-end analysis
│
├── data/                        ← Auto-populated price cache (pickle)
└── plots/                       ← Auto-generated PNG charts
```

### Module Dependency Flow

```
DataLoader → (external only)
RiskMetrics → (external only)
PortfolioOptimizer → RiskMetrics
Backtester → PortfolioOptimizer + RiskMetrics + DataLoader
Visualizer → RiskMetrics + PortfolioOptimizer
main.py → all modules via config
```

---

## Asset Universe

| Ticker | Name | Sector | Role in Portfolio |
|--------|------|--------|-------------------|
| AAPL | Apple Inc. | Technology | Growth |
| MSFT | Microsoft | Technology / Cloud | Growth |
| AMZN | Amazon | E-commerce / Cloud | Growth |
| GOOGL | Alphabet | Advertising / AI | Growth |
| JNJ | Johnson & Johnson | Healthcare | Defensive |
| JPM | JPMorgan Chase | Financials | Cyclical |
| GLD | SPDR Gold Trust | Commodity | Safe-haven |
| ^GSPC | S&P 500 Index | Benchmark | Reference |

**Rationale:** The universe spans the four major portfolio roles (growth, defensive, cyclical, safe-haven) to provide meaningful diversification opportunities and illustrate how correlation structure drives optimal allocation.

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/kathela/markowitz_portfolio_optimization.git
cd markowitz_portfolio_optimization

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Quick Start — Full Pipeline
```bash
python main.py
```

Expected runtime: **~3–5 minutes** (dominated by efficient frontier + backtesting).

### Custom Configuration
Edit `config.yaml` to change:
- **Tickers** — use any Yahoo Finance symbol
- **Date range** — adjust `start_date` and `end_date`
- **Risk-free rate** — update `risk_free_rate`
- **Weight constraints** — `constrained.min_weight` / `constrained.max_weight`
- **Rebalancing frequency** — `monthly` | `quarterly` | `annually`
- **Transaction costs** — `transaction_cost` (in decimal, e.g. 0.001 = 10 bps)

### Use Individual Modules
```python
import yaml
from src.data_loader import DataLoader
from src.portfolio_optimizer import PortfolioOptimizer
from src.risk_metrics import RiskMetrics

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Download data
loader = DataLoader(config)
prices = loader.download()
mu = loader.get_expected_returns()
sigma = loader.get_covariance(method="ledoit_wolf")  # shrinkage covariance

# Optimise
opt = PortfolioOptimizer(mu, sigma, risk_free_rate=0.05)
ms = opt.max_sharpe()
print(ms)                  # OptimizationResult with weights, return, vol, Sharpe

# Risk metrics
returns = loader.get_asset_returns() @ ms.weights
metrics = RiskMetrics.portfolio_summary(returns)
print(metrics.to_string())
```

### Run Tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Launch Notebook
```bash
jupyter notebook notebooks/portfolio_analysis.ipynb
```

---

## Visualisations

The system generates nine charts automatically:

| # | File | Description |
|---|------|-------------|
| 1 | `01_efficient_frontier.png` | MC scatter + frontier + CML + all optimal portfolios + individual assets |
| 2 | `02_portfolio_weights.png` | Grouped bar chart comparing all 5 strategy allocations |
| 3 | `03_correlation_heatmap.png` | Annotated pairwise return correlation matrix |
| 4 | `04_portfolio_growth.png` | Cumulative value comparison vs S&P 500 benchmark |
| 5 | `05_drawdown.png` | Underwater equity curves (peak-to-trough declines) |
| 6 | `06_rolling_metrics.png` | 1-year rolling Sharpe, volatility, and return |
| 7 | `07_risk_return_scatter.png` | Individual assets vs optimised portfolios in risk-return space |
| 8 | `08_monthly_returns_heatmap.png` | Calendar heatmap of monthly returns |
| 9 | `09_stress_test.png` | Performance during COVID crash, 2022 bear market, 2018 selloff |

---

## Interpreting the Results

### Max Sharpe Portfolio
The **tangency portfolio** — the point where the Capital Market Line touches the efficient frontier. This portfolio offers the highest return per unit of risk. In practice it typically concentrates in high-momentum growth equities (AAPL, MSFT, AMZN) because these have historically delivered superior risk-adjusted returns. Appropriate for **growth-oriented investors with a 5+ year horizon**.

### Min Variance Portfolio
The **Global Minimum Variance portfolio** — ignores expected returns entirely and minimises total portfolio volatility through diversification. Counterintuitively, research shows GMV often outperforms higher-target portfolios *out-of-sample* because it is less sensitive to estimation error in expected returns (Haugen & Baker, 1991). Appropriate for **conservative investors prioritising capital preservation**.

### Equal Weight Portfolio (1/N)
The naive benchmark. Research (DeMiguel et al., 2009) shows 1/N frequently outperforms sophisticated models out-of-sample — primarily because optimised models over-fit to the in-sample covariance structure. The 1/N rule serves as an important **sanity check** for all other strategies.

### Constrained Portfolio
Adds realistic limits (e.g. 2%–40% per asset) to the Max Sharpe optimisation. These bounds prevent near-zero and near-100% positions that would be impractical for an actual fund (compliance, liquidity, diversification mandates). **Most applicable to institutional portfolio management.**

---

## Covariance Estimation

The covariance matrix Σ is the most critical and uncertain input to MPT. Three methods are supported:

| Method | `get_covariance(method=...)` | Notes |
|--------|------------------------------|-------|
| Sample covariance | `'sample'` | Standard; suffers from estimation error with small T/N |
| Ledoit-Wolf | `'ledoit_wolf'` | Shrinks toward structured target; reduces out-of-sample error |
| EWMA | `'ewm'` | Weights recent observations more; adapts to regime changes |

For small sample sizes or high-dimensional universes, **Ledoit-Wolf shrinkage is recommended**.

---

## Backtesting Methodology

The backtest uses a **walk-forward** approach to avoid look-ahead bias:

```
Timeline:
 ├── In-Sample (2 years) ──► Fit optimiser
 │                           ↓
 ├── Out-of-Sample ────────► Apply weights until next rebalance
 │
 ├── Roll forward (e.g. quarterly) ──► Repeat
```

Transaction costs of 10 bps are deducted on each rebalancing based on portfolio turnover. This models realistic implementation friction.

---

## Assumptions & Limitations

| Assumption | Implication | Extension |
|------------|------------|-----------|
| Historical returns predict future | Backward-looking only | Black-Litterman (blend views + market priors) |
| Normal return distribution | Fat tails not modelled | CVaR optimisation, Cornish-Fisher VaR |
| Constant covariance matrix | Misses regime changes | Rolling/DCC-GARCH dynamic covariance |
| Long-only, no leverage | Restricted universe | Allow short positions; leverage constraints |
| No taxes or transaction costs | Over-estimates net returns | Tax-loss harvesting, full cost modelling |
| Single-period model | No intertemporal effects | Merton's dynamic portfolio problem |
| Market impact ignored | Realistic only for small portfolios | Almgren-Chriss market impact model |

---

## References

1. Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1), 77–91.
2. Sharpe, W. F. (1966). *Mutual Fund Performance*. Journal of Business, 39(1), 119–138.
3. DeMiguel, V., Garlappi, L., & Uppal, R. (2009). *Optimal versus Naive Diversification*. Review of Financial Studies, 22(5), 1915–1953.
4. Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365–411.
5. Haugen, R. A., & Baker, N. L. (1991). *The efficient market inefficiency of capitalization-weighted stock portfolios*. Journal of Portfolio Management, 17(3), 35–40.

---

## License

MIT License — free to use, modify, and distribute with attribution.
