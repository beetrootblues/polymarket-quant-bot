"""
Polymarket Quant Bot — Configuration
=====================================
Central configuration for all bot modules.
"""
from dataclasses import dataclass, field
from typing import Optional


# --- API Configuration ---

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GAMMA_API_TIMEOUT = 30
GAMMA_API_MAX_LIMIT = 500


# --- Simulation Configs ---

@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation parameters."""
    n_simulations: int = 100_000
    time_horizon_days: int = 30
    dt: float = 1 / 365
    risk_free_rate: float = 0.05
    base_volatility: float = 0.5


@dataclass
class ImportanceSamplingConfig:
    """Importance sampling for tail-risk analysis."""
    n_samples: int = 100_000
    tail_threshold: float = 0.05
    tilt_parameter: float = 2.0


@dataclass
class ParticleFilterConfig:
    """Sequential Monte Carlo / Particle Filter."""
    n_particles: int = 5000
    resample_threshold: float = 0.5
    observation_noise: float = 0.03
    process_volatility: float = 0.05

    @property
    def obs_noise(self) -> float:
        """Alias for observation_noise (used by particle_filter.py)."""
        return self.observation_noise

    @property
    def process_vol(self) -> float:
        """Alias for process_volatility (used by particle_filter.py)."""
        return self.process_volatility


@dataclass
class VarianceReductionConfig:
    """Variance reduction techniques."""
    n_simulations: int = 100_000
    n_strata: int = 10
    use_antithetic: bool = True
    use_control_variate: bool = True
    use_stratified: bool = True


@dataclass
class CopulaConfig:
    """Copula / correlation modeling."""
    copula_type: str = "student_t"
    t_copula_df: int = 4
    clayton_theta: float = 2.0
    gumbel_theta: float = 2.0
    n_simulations: int = 100_000


@dataclass
class ABMConfig:
    """Agent-Based Model parameters."""
    n_informed: int = 10
    n_noise: int = 50
    n_market_makers: int = 5
    initial_price: float = 0.50
    n_steps: int = 2000


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_position_size: float = 500.0
    max_position_pct: float = 0.10          # Max 10% of capital per position
    max_portfolio_exposure: float = 0.60    # Max 60% of capital deployed
    max_drawdown_pct: float = 0.15          # 15% drawdown circuit breaker
    kelly_fraction: float = 0.25            # Quarter-Kelly for safety
    min_edge_threshold: float = 0.02        # Minimum 2% edge to trade
    var_confidence: float = 0.95
    max_correlation_exposure: float = 0.7


@dataclass
class ExecutionConfig:
    """Trade execution parameters."""
    min_liquidity: float = 1000.0
    max_spread: float = 0.10
    slippage_tolerance: float = 0.02
    min_volume_24h: float = 500.0
    fee_rate: float = 0.02               # Polymarket ~2% fee on winnings
    slippage_model: str = "sqrt"          # "sqrt", "linear", or "quadratic"
    max_slippage_pct: float = 0.05        # Cap slippage at 5%


@dataclass
class BotConfig:
    """Master configuration aggregating all sub-configs."""
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    importance_sampling: ImportanceSamplingConfig = field(default_factory=ImportanceSamplingConfig)
    particle_filter: ParticleFilterConfig = field(default_factory=ParticleFilterConfig)
    variance_reduction: VarianceReductionConfig = field(default_factory=VarianceReductionConfig)
    copula: CopulaConfig = field(default_factory=CopulaConfig)
    abm: ABMConfig = field(default_factory=ABMConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Strategy weights for signal combination
    strategy_weights: dict = field(default_factory=lambda: {
        "monte_carlo": 0.3,
        "particle_filter": 0.25,
        "copula": 0.2,
        "abm": 0.15,
        "variance_reduction": 0.1,
    })

    # Capital allocation
    starting_capital: float = 10_000.0
    initial_capital: float = 0.0  # Alias for starting_capital

    def __post_init__(self):
        # Allow initial_capital as alias for starting_capital
        if self.initial_capital > 0 and self.starting_capital == 10_000.0:
            self.starting_capital = self.initial_capital
        elif self.initial_capital == 0.0:
            self.initial_capital = self.starting_capital
