"""
Polymarket Quant Bot — Sequential Monte Carlo / Particle Filter (Article Part IV)
==================================================================================
Real-time probability updating using particle filters.

From the article:
"When the market spikes from $0.58 to $0.65 on a single trade, the filter
recognizes that the true probability might not have changed that much —
it tempers the update based on how volatile the observation process has been."

State-space model:
- Hidden state x_t: true probability (unobserved)
- Observation y_t: market prices, poll results, news signals
- State evolves via logit random walk (keeps probabilities bounded)
- Observations are noisy readings of the true state
"""
import numpy as np
from scipy.special import expit, logit
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from config import ParticleFilterConfig


@dataclass
class FilterState:
    """Snapshot of the particle filter state at a point in time."""
    timestamp: float
    estimate: float
    ci_lower: float
    ci_upper: float
    ess: float
    observation: float
    n_resamples: int


class PredictionMarketParticleFilter:
    """
    Sequential Monte Carlo filter for real-time event probability estimation.
    Direct implementation from Article Part IV.

    Usage during a live event (e.g., election night):
        pf = PredictionMarketParticleFilter(prior_prob=0.50)
        pf.update(observed_price=0.55)  # market moves on early returns
        pf.update(observed_price=0.62)  # more data
        pf.update(observed_price=0.58)  # partial correction
        print(pf.estimate())            # filtered probability
    """

    def __init__(
        self,
        N_particles: int = 5000,
        prior_prob: float = 0.5,
        process_vol: float = 0.05,
        obs_noise: float = 0.03,
        config: Optional[ParticleFilterConfig] = None,
    ):
        if config:
            N_particles = config.n_particles
            process_vol = config.process_vol
            obs_noise = config.obs_noise

        self.N = N_particles
        self.process_vol = process_vol
        self.obs_noise = obs_noise
        self.ess_threshold = N_particles * 0.5

        # Initialize particles around prior in logit space
        logit_prior = logit(np.clip(prior_prob, 0.001, 0.999))
        self.logit_particles = logit_prior + np.random.normal(0, 0.5, N_particles)
        self.weights = np.ones(N_particles) / N_particles

        # Tracking
        self.history: List[FilterState] = []
        self.n_updates = 0
        self.total_resamples = 0

    def update(self, observed_price: float, obs_noise_override: Optional[float] = None):
        """
        Incorporate a new observation (market price, poll result, etc.)

        The Bootstrap Particle Filter algorithm from the article:
        1. PROPAGATE: x_t^(i) ~ f(. | x_{t-1}^(i))
        2. REWEIGHT: w_t^(i) proportional to g(y_t | x_t^(i))
        3. NORMALIZE: w~_t^(i) = w_t^(i) / Sum_j w_t^(j)
        4. RESAMPLE if ESS = 1/Sum(w~_t^(i))^2 < N/2
        """
        obs_noise = obs_noise_override or self.obs_noise

        # 1. Propagate: random walk in logit space
        noise = np.random.normal(0, self.process_vol, self.N)
        self.logit_particles += noise

        # 2. Convert to probability space
        prob_particles = expit(self.logit_particles)

        # 3. Reweight: likelihood of observation given each particle
        log_likelihood = -0.5 * ((observed_price - prob_particles) / obs_noise) ** 2
        log_weights = np.log(self.weights + 1e-300) + log_likelihood

        # Normalize in log space for numerical stability
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum()

        # 4. Check ESS and resample if needed
        ess = 1.0 / np.sum(self.weights ** 2)
        resampled = False
        if ess < self.ess_threshold:
            self._systematic_resample()
            self.total_resamples += 1
            resampled = True

        self.n_updates += 1

        # Record state
        est = self.estimate()
        ci = self.credible_interval()
        self.history.append(FilterState(
            timestamp=float(self.n_updates),
            estimate=est,
            ci_lower=ci[0],
            ci_upper=ci[1],
            ess=float(ess),
            observation=observed_price,
            n_resamples=self.total_resamples,
        ))

    def _systematic_resample(self):
        """
        Systematic resampling — lower variance than multinomial.
        From Article Part IV.
        """
        cumsum = np.cumsum(self.weights)
        u = (np.arange(self.N) + np.random.uniform()) / self.N
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, self.N - 1)
        self.logit_particles = self.logit_particles[indices].copy()
        self.weights = np.ones(self.N) / self.N

    def estimate(self) -> float:
        """Weighted mean probability estimate."""
        probs = expit(self.logit_particles)
        return float(np.average(probs, weights=self.weights))

    def credible_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Weighted quantile-based credible interval."""
        probs = expit(self.logit_particles)
        sorted_idx = np.argsort(probs)
        sorted_probs = probs[sorted_idx]
        sorted_weights = self.weights[sorted_idx]
        cumw = np.cumsum(sorted_weights)
        lower_idx = np.searchsorted(cumw, alpha / 2)
        upper_idx = np.searchsorted(cumw, 1 - alpha / 2)
        lower_idx = min(lower_idx, len(sorted_probs) - 1)
        upper_idx = min(upper_idx, len(sorted_probs) - 1)
        return float(sorted_probs[lower_idx]), float(sorted_probs[upper_idx])

    def particle_variance(self) -> float:
        """Variance of the particle distribution — measures uncertainty."""
        probs = expit(self.logit_particles)
        mean = np.average(probs, weights=self.weights)
        return float(np.average((probs - mean) ** 2, weights=self.weights))

    def effective_sample_size(self) -> float:
        """Current ESS — diagnostic for filter health."""
        return float(1.0 / np.sum(self.weights ** 2))

    def get_diagnostics(self) -> Dict:
        """Full diagnostic report."""
        return {
            "n_updates": self.n_updates,
            "current_estimate": self.estimate(),
            "credible_interval_95": self.credible_interval(),
            "particle_variance": self.particle_variance(),
            "ess": self.effective_sample_size(),
            "ess_ratio": self.effective_sample_size() / self.N,
            "total_resamples": self.total_resamples,
            "resample_rate": self.total_resamples / max(self.n_updates, 1),
        }


class MultiMarketParticleFilter:
    """
    Run independent particle filters for multiple markets simultaneously.
    Each market gets its own filter, updated with its own price feed.

    For correlated markets, use the CopulaEngine (Part VI) on top.
    """

    def __init__(self, config: Optional[ParticleFilterConfig] = None):
        self.config = config or ParticleFilterConfig()
        self.filters: Dict[str, PredictionMarketParticleFilter] = {}

    def add_market(self, market_id: str, prior_prob: float = 0.5):
        """Register a new market to track."""
        self.filters[market_id] = PredictionMarketParticleFilter(
            prior_prob=prior_prob,
            config=self.config,
        )

    def update_market(self, market_id: str, observed_price: float):
        """Update a single market's filter."""
        if market_id not in self.filters:
            self.add_market(market_id, prior_prob=observed_price)
        self.filters[market_id].update(observed_price)

    def update_all(self, observations: Dict[str, float]):
        """Batch update: {market_id: observed_price}."""
        for market_id, price in observations.items():
            self.update_market(market_id, price)

    def get_estimates(self) -> Dict[str, float]:
        """Current filtered estimates for all markets."""
        return {mid: f.estimate() for mid, f in self.filters.items()}

    def get_estimate(self, market_id: str) -> float:
        """Get filtered estimate for a single market."""
        if market_id in self.filters:
            return self.filters[market_id].estimate()
        return 0.5

    def get_divergences(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Compare filtered estimates vs current market prices.
        Positive divergence = filter thinks true prob is HIGHER than market.
        These are potential trade signals.
        """
        divergences = {}
        for mid, price in market_prices.items():
            if mid in self.filters:
                est = self.filters[mid].estimate()
                divergences[mid] = est - price
        return divergences

    def get_all_diagnostics(self) -> Dict[str, Dict]:
        return {mid: f.get_diagnostics() for mid, f in self.filters.items()}


class AdaptiveParticleFilter(PredictionMarketParticleFilter):
    """
    Extended particle filter with adaptive process volatility.

    Learns the observation noise and process volatility from data,
    rather than using fixed values. Essential for markets where
    volatility regime shifts (e.g., election night vs. quiet period).
    """

    def __init__(self, prior_prob: float = 0.5, N_particles: int = 5000,
                 initial_process_vol: float = 0.05, initial_obs_noise: float = 0.03,
                 adaptation_rate: float = 0.05):
        super().__init__(N_particles=N_particles, prior_prob=prior_prob,
                         process_vol=initial_process_vol, obs_noise=initial_obs_noise)
        self.adaptation_rate = adaptation_rate
        self.recent_innovations = []
        self.vol_history = []

    def update(self, observed_price: float, obs_noise_override: Optional[float] = None):
        """Update with adaptive noise estimation."""
        # Track innovation (surprise) for adaptation
        predicted = self.estimate()
        innovation = observed_price - predicted
        self.recent_innovations.append(innovation)

        # Keep rolling window of 50 innovations
        if len(self.recent_innovations) > 50:
            self.recent_innovations = self.recent_innovations[-50:]

        # Adapt observation noise based on realized innovation variance
        if len(self.recent_innovations) >= 10:
            realized_var = np.var(self.recent_innovations[-20:])
            target_obs_noise = np.sqrt(realized_var)
            self.obs_noise = (
                (1 - self.adaptation_rate) * self.obs_noise
                + self.adaptation_rate * np.clip(target_obs_noise, 0.005, 0.20)
            )

        # Adapt process volatility: if ESS drops frequently, increase
        ess_ratio = self.effective_sample_size() / self.N
        if ess_ratio < 0.3:
            self.process_vol *= 1.02  # Increase exploration
        elif ess_ratio > 0.8:
            self.process_vol *= 0.98  # Decrease — particles are diverse enough

        self.process_vol = np.clip(self.process_vol, 0.01, 0.30)
        self.vol_history.append(self.process_vol)

        super().update(observed_price, obs_noise_override)
