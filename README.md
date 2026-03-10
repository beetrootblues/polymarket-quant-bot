# Polymarket Quant Bot

A quantitative trading system for [Polymarket](https://polymarket.com) prediction markets, implementing institutional-grade simulation and risk management techniques.

Based on the research paper *"How to Simulate Like a Quant Desk"*.

## Architecture

LAYER 0: Configuration          config.py             (141 lines)
LAYER 1: Data Ingestion         data_layer.py         (432 lines)
LAYER 2: Probability Engine
  ├── Monte Carlo / IS          monte_carlo.py        (292 lines)
  ├── Particle Filter (SMC)     particle_filter.py    (281 lines)
  └── Variance Reduction        variance_reduction.py (349 lines)
LAYER 3: Dependency Modeling
  ├── Copula Engine             copula_engine.py      (415 lines)
  └── Agent-Based Model         agent_based_model.py  (396 lines)
LAYER 4-5: Signals & Risk       signal_engine.py      (553 lines)
ENTRY POINT                     main.py               (203 lines)
────────────────────────────────────────────────────────────────
TOTAL                                                 3,063 lines

## Quant Models Implemented

| Model | Module | Article Part | Purpose |
|-------|--------|-------------|--------|
| GBM Monte Carlo | monte_carlo.py | Part II | Binary contract pricing via simulation |
| Brier Score / BSS | monte_carlo.py | Part II | Calibration quality metrics |
| Logit Jump-Diffusion | monte_carlo.py | Part II | Event-driven probability dynamics |
| Importance Sampling | monte_carlo.py | Part III | Tail-risk / rare event estimation |
| Particle Filter (SMC) | particle_filter.py | Part IV | Real-time Bayesian probability updates |
| Antithetic Variates | variance_reduction.py | Part V | 50-75% variance reduction (free) |
| Control Variates | variance_reduction.py | Part V | BS digital as control for MC |
| Stratified Sampling | variance_reduction.py | Part V | Divide-and-conquer variance reduction |
| Gaussian Copula | copula_engine.py | Part VI | Correlation modeling (no tail dep.) |
| Student-t Copula | copula_engine.py | Part VI | Symmetric tail dependence |
| Clayton Copula | copula_engine.py | Part VI | Lower tail dependence (crash contagion) |
| Gumbel Copula | copula_engine.py | Part VI | Upper tail dependence |
| Agent-Based Model | agent_based_model.py | Part VII | Market microstructure / Kyle's Lambda |
| Kelly Criterion | signal_engine.py | Part VIII | Optimal position sizing (quarter-Kelly) |

## Dependencies

- **numpy** — numerical computing
- **scipy** — statistical distributions, special functions
- **httpx** — async-capable HTTP client for Gamma API

## Quick Start

pip install -r requirements.txt
python main.py

The bot connects to the [Polymarket Gamma API](https://gamma-api.polymarket.com) (no auth required), fetches live market data, runs the full 5-layer pipeline, and outputs actionable trading signals.

## Configuration

All parameters are centralized in config.py using Python dataclasses:

from config import BotConfig
config = BotConfig(starting_capital=10_000)

Key risk defaults: quarter-Kelly sizing, 10% max position, 60% max exposure, 15% drawdown circuit breaker.

## Data Source

All market data comes from the **Gamma API** (https://gamma-api.polymarket.com):
- No authentication required
- 500 markets per page, paginated via offset
- Fields like outcomePrices, outcomes, clobTokenIds are JSON-encoded strings

## License

MIT