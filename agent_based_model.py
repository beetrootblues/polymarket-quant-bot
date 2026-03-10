"""
Polymarket Quant Bot — Agent-Based Model (Article Part VII)
============================================================
Simulates prediction market microstructure with heterogeneous agents.

From the article:
"Gode & Sunder (1993) showed that zero-intelligence agents — traders who
submit random orders subject only to budget constraints — achieve near-100%
allocative efficiency in a continuous double auction."

"Farmer, Patelli & Zovko (2005) extended this to limit order books.
This explained 96% of cross-sectional spread variation on the London
Stock Exchange. One parameter. 96%."

Agent types:
- Informed: know the true probability, trade toward it
- Noise: random trades
- Market maker: provides liquidity around current price

Key metric: Kyle's Lambda — price impact parameter
  lambda = sigma_v / (2 * sigma_u)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from config import ABMConfig


@dataclass
class TradeRecord:
    """Record of a single trade in the ABM."""
    timestep: int
    agent_type: str  # "informed" | "noise" | "mm"
    direction: str   # "buy" | "sell" | "quote_update"
    size: float
    price_before: float
    price_after: float
    pnl_contribution: float


class PredictionMarketABM:
    """
    Agent-based model of a prediction market order book.
    Directly from Article Part VII.

    Agent types:
    - Informed: know the true probability, trade toward it
    - Noise: random trades
    - Market maker: provides liquidity around current price

    Key dynamics:
    - How fast prices converge depends on informed/noise ratio
    - Market maker spread responds to information flow
    - Informed traders extract profit at noise traders' expense
    """

    def __init__(
        self,
        true_prob: float,
        n_informed: int = 10,
        n_noise: int = 50,
        n_mm: int = 5,
        initial_price: float = 0.50,
        config: Optional[ABMConfig] = None,
    ):
        if config:
            n_informed = config.n_informed
            n_noise = config.n_noise
            n_mm = config.n_market_makers
            initial_price = config.initial_price

        self.true_prob = true_prob
        self.price = initial_price
        self.price_history = [self.price]

        # Order book (simplified as bid/ask queues)
        self.best_bid = initial_price - 0.01
        self.best_ask = initial_price + 0.01

        # Agent populations
        self.n_informed = n_informed
        self.n_noise = n_noise
        self.n_mm = n_mm

        # Track metrics
        self.volume = 0.0
        self.informed_pnl = 0.0
        self.noise_pnl = 0.0
        self.mm_pnl = 0.0
        self.trade_count = 0
        self.trades: List[TradeRecord] = []
        self.spread_history = [self.best_ask - self.best_bid]
        self.kyle_lambda_history = []

    def step(self):
        """
        One time step: randomly select an agent to trade.
        From Article Part VII.
        """
        total = self.n_informed + self.n_noise + self.n_mm
        r = np.random.random()

        price_before = self.price

        if r < self.n_informed / total:
            self._informed_trade()
        elif r < (self.n_informed + self.n_noise) / total:
            self._noise_trade()
        else:
            self._mm_update()

        self.price_history.append(self.price)
        self.spread_history.append(self.best_ask - self.best_bid)
        self.kyle_lambda_history.append(self._kyle_lambda())

    def _informed_trade(self):
        """
        Informed trader: buy if price < true_prob, sell otherwise.
        From Article Part VII.
        """
        signal = self.true_prob + np.random.normal(0, 0.02)  # noisy signal
        price_before = self.price

        if signal > self.best_ask + 0.01:  # buy
            size = min(0.1, abs(signal - self.price) * 2)
            self.price += size * self._kyle_lambda()
            self.volume += size
            pnl = (self.true_prob - self.best_ask) * size
            self.informed_pnl += pnl
            self.trade_count += 1
            self.trades.append(TradeRecord(
                len(self.price_history), "informed", "buy",
                size, price_before, self.price, pnl
            ))
        elif signal < self.best_bid - 0.01:  # sell
            size = min(0.1, abs(self.price - signal) * 2)
            self.price -= size * self._kyle_lambda()
            self.volume += size
            pnl = (self.best_bid - self.true_prob) * size
            self.informed_pnl += pnl
            self.trade_count += 1
            self.trades.append(TradeRecord(
                len(self.price_history), "informed", "sell",
                size, price_before, self.price, pnl
            ))

        self.price = np.clip(self.price, 0.01, 0.99)
        self._update_book()

    def _noise_trade(self):
        """
        Noise trader: random buy/sell.
        From Article Part VII.
        """
        direction = np.random.choice([-1, 1])
        size = np.random.exponential(0.02)
        price_before = self.price

        self.price += direction * size * self._kyle_lambda()
        self.price = np.clip(self.price, 0.01, 0.99)
        self.volume += size
        pnl = -abs(self.price - self.true_prob) * size * 0.5
        self.noise_pnl += pnl
        self.trade_count += 1

        self.trades.append(TradeRecord(
            len(self.price_history), "noise",
            "buy" if direction > 0 else "sell",
            size, price_before, self.price, pnl
        ))
        self._update_book()

    def _mm_update(self):
        """
        Market maker: tighten spread toward current price.
        Spread responds to information flow (volume).
        From Article Part VII.
        """
        spread = max(0.02, 0.05 * (1 - self.volume / 100))
        self.best_bid = self.price - spread / 2
        self.best_ask = self.price + spread / 2

        # MM earns the spread from noise traders
        mm_revenue = spread * 0.01  # Small per-update revenue
        self.mm_pnl += mm_revenue

    def _kyle_lambda(self) -> float:
        """
        Price impact parameter — Kyle's Lambda.
        From Article Part VII and Kyle (1985).

        lambda = sigma_v / (2 * sigma_u)

        sigma_v: uncertainty about true value
        sigma_u: noise trading intensity
        """
        sigma_v = abs(self.true_prob - self.price) + 0.05
        sigma_u = 0.1 * np.sqrt(self.n_noise)
        return sigma_v / (2 * sigma_u)

    def _update_book(self):
        """Update order book around current price."""
        spread = self.best_ask - self.best_bid
        spread = max(spread, 0.02)  # Minimum spread
        self.best_bid = self.price - spread / 2
        self.best_ask = self.price + spread / 2

    def run(self, n_steps: int = 2000) -> np.ndarray:
        """Run the simulation for n_steps."""
        for _ in range(n_steps):
            self.step()
        return np.array(self.price_history)

    def get_results(self) -> Dict:
        """Comprehensive simulation results."""
        prices = np.array(self.price_history)
        spreads = np.array(self.spread_history)

        return {
            "true_prob": self.true_prob,
            "initial_price": self.price_history[0],
            "final_price": float(prices[-1]),
            "convergence_error": float(abs(prices[-1] - self.true_prob)),
            "convergence_time": self._estimate_convergence_time(prices),
            "total_volume": float(self.volume),
            "trade_count": self.trade_count,
            "informed_pnl": float(self.informed_pnl),
            "noise_pnl": float(self.noise_pnl),
            "mm_pnl": float(self.mm_pnl),
            "avg_spread": float(spreads.mean()),
            "final_spread": float(spreads[-1]),
            "price_volatility": float(prices.std()),
            "avg_kyle_lambda": float(np.mean(self.kyle_lambda_history)) if self.kyle_lambda_history else 0.0,
            "price_at_checkpoints": {
                "t=25%": float(prices[len(prices)//4]),
                "t=50%": float(prices[len(prices)//2]),
                "t=75%": float(prices[3*len(prices)//4]),
                "t=100%": float(prices[-1]),
            },
        }

    def _estimate_convergence_time(self, prices: np.ndarray, threshold: float = 0.02) -> Optional[int]:
        """Estimate when price first stays within threshold of true_prob."""
        for t in range(len(prices)):
            if abs(prices[t] - self.true_prob) < threshold:
                # Check if it stays close for next 50 steps
                window = prices[t:t+50]
                if len(window) >= 50 and all(abs(w - self.true_prob) < threshold * 2 for w in window):
                    return t
        return None


class MarketRegimeDetector:
    """
    Uses ABM simulations to characterize current market regime.

    Runs multiple ABMs with different informed/noise ratios
    to find which best matches observed market behavior.
    """

    def __init__(self):
        self.regime_configs = {
            "high_information": {"n_informed": 30, "n_noise": 20, "n_mm": 5},
            "balanced": {"n_informed": 10, "n_noise": 50, "n_mm": 5},
            "noise_dominated": {"n_informed": 3, "n_noise": 80, "n_mm": 5},
            "thin_liquidity": {"n_informed": 5, "n_noise": 15, "n_mm": 2},
        }

    def detect_regime(
        self,
        observed_prices: np.ndarray,
        candidate_true_probs: List[float],
        n_sims_per_config: int = 20,
        n_steps: int = None,
    ) -> Dict:
        """
        Detect which market regime best explains observed price behavior.

        Compares observed price path statistics against simulated paths
        from each regime configuration.
        """
        if n_steps is None:
            n_steps = len(observed_prices)

        # Compute observed statistics
        obs_stats = self._compute_path_stats(observed_prices)

        best_regime = None
        best_distance = float("inf")
        regime_scores = {}

        for regime_name, regime_config in self.regime_configs.items():
            distances = []

            for true_prob in candidate_true_probs:
                for _ in range(n_sims_per_config):
                    abm = PredictionMarketABM(
                        true_prob=true_prob,
                        n_informed=regime_config["n_informed"],
                        n_noise=regime_config["n_noise"],
                        n_mm=regime_config["n_mm"],
                        initial_price=observed_prices[0],
                    )
                    sim_prices = abm.run(n_steps)
                    sim_stats = self._compute_path_stats(sim_prices[:len(observed_prices)])

                    dist = self._stats_distance(obs_stats, sim_stats)
                    distances.append(dist)

            avg_dist = np.mean(distances)
            regime_scores[regime_name] = float(avg_dist)

            if avg_dist < best_distance:
                best_distance = avg_dist
                best_regime = regime_name

        return {
            "detected_regime": best_regime,
            "regime_scores": regime_scores,
            "confidence": 1.0 - (best_distance / (sum(regime_scores.values()) / len(regime_scores))),
        }

    def _compute_path_stats(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute summary statistics from a price path."""
        returns = np.diff(prices)
        return {
            "mean_return": float(np.mean(returns)),
            "volatility": float(np.std(returns)),
            "autocorr_1": float(np.corrcoef(returns[:-1], returns[1:])[0, 1]) if len(returns) > 2 else 0.0,
            "skewness": float(np.mean(((returns - np.mean(returns)) / (np.std(returns) + 1e-10))**3)),
            "max_drawdown": float(np.min(np.minimum.accumulate(prices) - prices)),
            "range": float(np.max(prices) - np.min(prices)),
        }

    def _stats_distance(self, obs: Dict, sim: Dict) -> float:
        """Weighted distance between observed and simulated statistics."""
        weights = {
            "volatility": 3.0,
            "autocorr_1": 2.0,
            "skewness": 1.0,
            "range": 1.5,
            "mean_return": 1.0,
        }

        total = 0.0
        for key, weight in weights.items():
            if key in obs and key in sim:
                denom = abs(obs[key]) + 1e-10
                total += weight * abs(obs[key] - sim[key]) / denom

        return total


def estimate_kyle_lambda_from_data(
    prices: np.ndarray,
    volumes: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate Kyle's Lambda (price impact) from real market data.

    lambda = Cov(Delta_p, signed_volume) / Var(signed_volume)

    This measures how much prices move per unit of net buying pressure.
    Higher lambda = less liquid = more price impact.
    """
    if len(prices) < 3 or len(volumes) < 3:
        return {"kyle_lambda": 0.0, "r_squared": 0.0, "n_observations": 0}

    price_changes = np.diff(prices)
    # Use volume with sign inferred from price direction
    signed_vol = volumes[1:] * np.sign(price_changes)

    min_len = min(len(price_changes), len(signed_vol))
    price_changes = price_changes[:min_len]
    signed_vol = signed_vol[:min_len]

    # Kyle regression: Delta_p = lambda * signed_volume + epsilon
    if np.var(signed_vol) > 0:
        cov = np.cov(price_changes, signed_vol)
        kyle_lambda = cov[0, 1] / cov[1, 1]

        # R-squared
        predicted = kyle_lambda * signed_vol
        ss_res = np.sum((price_changes - predicted)**2)
        ss_tot = np.sum((price_changes - price_changes.mean())**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        kyle_lambda = 0.0
        r_squared = 0.0

    return {
        "kyle_lambda": float(kyle_lambda),
        "r_squared": float(max(0, r_squared)),
        "n_observations": min_len,
        "avg_impact_per_unit": float(abs(kyle_lambda)),
    }
