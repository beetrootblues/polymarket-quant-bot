"""
Polymarket Quant Bot -- Signal Engine v3
=========================================
Complete rewrite based on v2 backtest findings (2026-03-10):

  v2 FAILURES:
  1. Whale intelligence was the only real alpha source, but MockWhaleTracker
     was CHEATING -- peeking at resolved outcomes to bias signals (60-75%
     accuracy). With honest blind whales, win rate drops to 33%.
  2. Particle Filter, ABM, Copula provided zero alpha (Brier ~0.354 = 
     market-price echo). They added complexity without signal.
  3. Monte Carlo & Variance Reduction were catastrophically broken
     (Brier 0.77). Already removed in v2 but PF/ABM/Copula lingered.
  4. No pre-trade filter combination produced positive P&L once the
     whale data leak was fixed.

  v3 ARCHITECTURE -- 3-layer honest design:
  Layer A -- Whale Intelligence (weight 0.40)
      Real on-chain whale positioning from Polymarket Data API + Goldsky.
      Honest mock for backtesting (no outcome peeking).
      Primary alpha source when real data is available.

  Layer B -- Market Regime Detection (weight 0.35)
      Empirical regime classification based on price level, time to
      resolution, volume, and category. Adjusts edge requirements
      and position sizing per regime type:
      - Endgame: >95% prob + <48h to resolution (proven profitable)
      - Sweet spot: 0.30-0.55 price range (empirical 80% WR zone)
      - Extreme: <0.15 or >0.85 (usually correctly priced, need 2x edge)
      - Long-dated: >90 days (noisy, need 1.5x edge)
      - Politics: needs 1.5x edge (less smart money alpha)

  Layer C -- Orderbook Imbalance (weight 0.25)
      Real-time CLOB depth asymmetry via OrderbookScanner.
      Mock version for backtesting uses price-level heuristics (~52-55%
      accuracy, honest).

  Bayesian shrinkage (30%) toward market price is applied AFTER ensemble,
  not as a separate layer. This prevents overconfidence.

  ALL v1/v2 broken models REMOVED:
  - NO particle_filter import
  - NO agent_based_model import  
  - NO copula_engine import
  - NO monte_carlo import
  - NO variance_reduction import

Classes:
  MarketRegime         -- Detected regime for a single market
  RegimeDetector       -- Classifies markets into regimes
  SignalV3             -- A v3 trading signal with multi-layer evidence
  ProbabilityEngineV3  -- 3-layer ensemble probability estimator
  PreTradeFilterV3     -- Hard rejection gates (all must pass)
  RiskManagerV3        -- Quarter-Kelly sizing with regime/whale adjustments
  SignalGeneratorV3    -- Top-level orchestrator
"""
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from config import BotConfig, RiskConfig
from whale_intelligence import (
    WhaleTracker,
    MockWhaleTracker,
    WhaleIntelligence,
    WhaleConfig,
)
from orderbook_scanner import (
    OrderbookScanner,
    MockOrderbookScanner,
    OrderbookSignal,
    OBIConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  1. MarketRegime -- Detected market regime for a single market
# ---------------------------------------------------------------------------

@dataclass
class MarketRegime:
    """Detected market regime for a single market."""
    market_id: str = ""
    regime_type: str = "normal"  # "endgame", "sweet_spot", "long_dated", "extreme", "normal"
    # Regime characteristics
    is_endgame: bool = False          # >95% prob AND resolution within 48h
    is_sweet_spot: bool = False       # Price 0.30-0.55 (empirical 80% win rate zone)
    is_long_dated: bool = False       # >90 days to resolution
    is_extreme_price: bool = False    # <0.15 or >0.85
    is_politics: bool = False         # Politics category (needs 2x edge)
    # Adjustments
    edge_multiplier: float = 1.0      # Applied to min_edge requirement
    size_multiplier: float = 1.0      # Applied to position sizing
    confidence_boost: float = 0.0     # Added to signal confidence
    priority_score: float = 0.0       # Higher = scan first
    days_to_resolution: float = -1.0  # -1 if unknown


# ---------------------------------------------------------------------------
#  2. RegimeDetector -- Classifies markets into regimes
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Detects market regime to adjust strategy parameters."""

    POLITICS_KEYWORDS = [
        "president", "election", "congress", "senate", "house", "governor",
        "trump", "biden", "vance", "harris", "desantis", "newsom",
        "republican", "democrat", "gop", "dnc", "rnc", "primary",
        "caucus", "impeach", "scotus", "supreme court", "cabinet",
        "fed", "inflation", "gdp", "tariff", "debt ceiling", "shutdown",
        "executive order", "electoral", "ballot", "vote", "poll",
    ]

    def detect(self, market) -> MarketRegime:
        """Analyze market and return regime classification."""
        regime = MarketRegime(market_id=getattr(market, "market_id", ""))
        yes_price = getattr(market, "yes_price", 0.5)
        question = getattr(market, "question", "").lower()
        category = getattr(market, "category", "").lower()
        end_date_str = getattr(market, "end_date", None)
        volume = getattr(market, "volume", 0)
        volume_24h = getattr(market, "volume_24h", 0)

        # Politics detection
        combined_text = f"{question} {category}"
        regime.is_politics = any(kw in combined_text for kw in self.POLITICS_KEYWORDS)

        # Time to resolution
        if end_date_str:
            try:
                if isinstance(end_date_str, str):
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                else:
                    end_dt = end_date_str
                now = datetime.now(timezone.utc)
                regime.days_to_resolution = max(0, (end_dt - now).total_seconds() / 86400)
            except Exception:
                pass

        # Endgame: >95% AND resolving within 48h
        if (yes_price > 0.95 or yes_price < 0.05) and 0 <= regime.days_to_resolution <= 2:
            regime.is_endgame = True
            regime.regime_type = "endgame"
            regime.edge_multiplier = 0.5  # Lower edge requirement (endgame is proven profitable)
            regime.size_multiplier = 1.5  # Bigger size (high conviction)
            regime.confidence_boost = 0.2
            regime.priority_score = 10.0
            return regime

        # Sweet spot: 0.30-0.55 (empirically 80% win rate)
        if 0.30 <= yes_price <= 0.55:
            regime.is_sweet_spot = True
            regime.regime_type = "sweet_spot"
            regime.confidence_boost = 0.10
            regime.priority_score = 7.0

        # Extreme price: <0.15 or >0.85 (usually correctly priced, hard to beat)
        if yes_price < 0.15 or yes_price > 0.85:
            regime.is_extreme_price = True
            regime.regime_type = "extreme"
            regime.edge_multiplier = 2.0  # Need 2x edge to override market consensus
            regime.size_multiplier = 0.5  # Smaller size
            regime.priority_score = 2.0

        # Long-dated: >90 days (more noise, less efficient)
        if regime.days_to_resolution > 90:
            regime.is_long_dated = True
            if regime.regime_type == "normal":
                regime.regime_type = "long_dated"
            regime.edge_multiplier = max(regime.edge_multiplier, 1.5)  # Need 1.5x edge
            regime.size_multiplier *= 0.7
            regime.priority_score = max(regime.priority_score, 3.0)

        # Politics adjustment
        if regime.is_politics:
            regime.edge_multiplier *= 1.5  # Politics need 1.5x edge (less smart money alpha)
            regime.priority_score = max(regime.priority_score, 5.0)

        # Volume-based priority (sweet spot: $50K-$500K daily volume)
        if 50_000 <= volume_24h <= 500_000:
            regime.priority_score += 2.0

        if regime.regime_type == "normal":
            regime.priority_score = max(1.0, regime.priority_score)

        return regime


# ---------------------------------------------------------------------------
#  3. SignalV3 -- A v3 trading signal with multi-layer evidence
# ---------------------------------------------------------------------------

@dataclass
class SignalV3:
    """A v3 trading signal with multi-layer evidence."""
    market_id: str = ""
    question: str = ""
    direction: str = ""           # "BUY_YES" or "BUY_NO"
    confidence: str = "low"       # "high", "medium", "low"
    edge: float = 0.0             # Raw edge
    adjusted_edge: float = 0.0    # After regime adjustments
    entry_price: float = 0.0
    size_usd: float = 0.0
    # Layer evidence
    whale_direction: str = "NEUTRAL"
    whale_confidence: float = 0.0
    whale_strength: str = "none"
    obi_direction: str = "NEUTRAL"
    obi_imbalance: float = 0.0
    obi_strength: str = "none"
    regime_type: str = "normal"
    regime_edge_mult: float = 1.0
    # Alignment
    whale_obi_aligned: bool = False
    n_layers_agree: int = 0       # Out of 3 (whale, OBI, regime)
    # Gate results
    passed_all_gates: bool = False
    rejection_reasons: list = field(default_factory=list)


# ---------------------------------------------------------------------------
#  4. ProbabilityEngineV3 -- 3-layer ensemble probability estimator
# ---------------------------------------------------------------------------

class ProbabilityEngineV3:
    """
    V3 ensemble probability estimator.

    3-layer architecture (replacing v2's 5 broken layers):
    - Whale Intelligence: 0.40 weight (real on-chain data)
    - Market Regime: 0.35 weight (empirical regime characteristics)
    - Orderbook Imbalance: 0.25 weight (real-time depth asymmetry)

    Market price anchor is applied as Bayesian shrinkage AFTER ensemble,
    not as a separate layer.
    """
    WEIGHTS = {
        "whale": 0.40,
        "regime": 0.35,
        "orderbook": 0.25,
    }
    SHRINKAGE = 0.30  # 30% shrinkage toward market price

    def __init__(self, whale_tracker=None):
        self.whale_tracker = whale_tracker
        self.regime_detector = RegimeDetector()
        self.estimates = {}  # market_id -> layer estimates
        self.whale_intel = {}  # market_id -> WhaleIntelligence
        self.obi_signals = {}  # market_id -> OrderbookSignal
        self.regimes = {}  # market_id -> MarketRegime
        self.market_prices = {}  # market_id -> yes_price (for shrinkage)

    def update(self, market, whale_intel=None, obi_signal=None):
        """Update all layer estimates for a market."""
        mid = getattr(market, "market_id", str(market))
        yes_price = getattr(market, "yes_price", 0.5)
        self.market_prices[mid] = yes_price

        estimates = {}

        # Layer A: Whale Intelligence -> probability
        if whale_intel is None and self.whale_tracker:
            try:
                whale_intel = self.whale_tracker.analyze_market(market)
            except Exception as e:
                logger.warning(f"Whale analysis failed for {mid}: {e}")
                whale_intel = None

        if whale_intel:
            self.whale_intel[mid] = whale_intel
            # Convert whale direction + confidence to probability
            if whale_intel.whale_direction == "YES":
                whale_prob = 0.5 + whale_intel.whale_confidence_score * 0.40
            elif whale_intel.whale_direction == "NO":
                whale_prob = 0.5 - whale_intel.whale_confidence_score * 0.40
            else:
                whale_prob = yes_price  # Neutral = defer to market
            estimates["whale"] = float(np.clip(whale_prob, 0.05, 0.95))
        else:
            estimates["whale"] = yes_price

        # Layer B: Market Regime -> probability adjustment
        regime = self.regime_detector.detect(market)
        self.regimes[mid] = regime

        # Regime probability: starts at market price, adjusted by regime
        regime_prob = yes_price
        if regime.is_endgame:
            # Endgame: amplify toward extreme (the market is right, we're buying certainty)
            regime_prob = yes_price * 1.03 if yes_price > 0.5 else yes_price * 0.97
        elif regime.is_sweet_spot:
            # Sweet spot: slight mean reversion (these markets are volatile)
            regime_prob = yes_price + (0.5 - yes_price) * 0.05
        elif regime.is_extreme_price:
            # Extreme: trust the market more (it's usually right)
            regime_prob = yes_price  # No adjustment
        elif regime.is_long_dated:
            # Long-dated: slight pull toward 0.5 (uncertainty)
            regime_prob = yes_price + (0.5 - yes_price) * 0.03
        regime_prob = float(np.clip(regime_prob, 0.05, 0.95))
        estimates["regime"] = regime_prob

        # Layer C: Orderbook Imbalance -> probability
        if obi_signal:
            self.obi_signals[mid] = obi_signal
            if obi_signal.signal_direction == "YES":
                obi_prob = 0.5 + obi_signal.confidence_score * 0.30
            elif obi_signal.signal_direction == "NO":
                obi_prob = 0.5 - obi_signal.confidence_score * 0.30
            else:
                obi_prob = yes_price
            estimates["orderbook"] = float(np.clip(obi_prob, 0.05, 0.95))
        else:
            estimates["orderbook"] = yes_price

        self.estimates[mid] = estimates

    def get_combined_probability(self, market_id: str) -> float:
        """Weighted ensemble with Bayesian shrinkage toward market price."""
        if market_id not in self.estimates:
            return 0.5

        ests = self.estimates[market_id]
        weighted_sum = 0.0
        total_weight = 0.0

        for layer, weight in self.WEIGHTS.items():
            if layer in ests:
                weighted_sum += weight * ests[layer]
                total_weight += weight

        if total_weight == 0:
            return 0.5

        raw_prob = weighted_sum / total_weight

        # Bayesian shrinkage toward market price
        market_price = self.market_prices.get(market_id, 0.5)
        final_prob = (1.0 - self.SHRINKAGE) * raw_prob + self.SHRINKAGE * market_price

        return float(np.clip(final_prob, 0.05, 0.95))

    def get_layer_agreement(self, market_id: str, market_price: float) -> Tuple[int, int, str]:
        """Count how many layers agree on direction."""
        if market_id not in self.estimates:
            return 0, 0, "NEUTRAL"

        ests = self.estimates[market_id]
        n_yes = sum(1 for v in ests.values() if v > market_price + 0.02)
        n_no = sum(1 for v in ests.values() if v < market_price - 0.02)

        if n_yes > n_no:
            return n_yes, n_no, "YES"
        elif n_no > n_yes:
            return n_yes, n_no, "NO"
        return n_yes, n_no, "NEUTRAL"


# ---------------------------------------------------------------------------
#  5. PreTradeFilterV3 -- Hard rejection gates (all must pass)
# ---------------------------------------------------------------------------

class PreTradeFilterV3:
    """
    Hard pre-trade rejection gates. ALL must pass for a trade to execute.

    v3 gates:
    1. Volume >= $500K (liquidity requirement)
    2. Spread <= 0.06 (tighter than v2's 0.08)
    3. Price in [0.10, 0.90] (unless endgame regime)
    4. Min edge >= 6% * regime.edge_multiplier
    5. Whale and OBI directionally aligned
    6. At least 2/3 layers agree on direction
    7. Drawdown circuit breaker (halt if cumulative P&L < -8% for 3 cycles)
    """

    STRENGTH_ORDER = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}

    # Default thresholds
    MIN_VOLUME: float = 500_000.0
    MAX_SPREAD: float = 0.06
    MIN_PRICE: float = 0.10
    MAX_PRICE: float = 0.90
    BASE_MIN_EDGE: float = 0.06
    MIN_LAYERS_AGREE: int = 2
    DRAWDOWN_HALT_PCT: float = 0.08   # -8% cumulative triggers halt
    DRAWDOWN_HALT_CYCLES: int = 3     # Must be below for 3 consecutive cycles

    def __init__(self):
        self._drawdown_breach_count = 0

    def check(self, market, signal: SignalV3, regime: MarketRegime,
              risk_manager: 'RiskManagerV3') -> Tuple[bool, Dict[str, bool], List[str]]:
        """
        Run all gates. Returns (all_passed, per_gate_results, rejection_reasons).
        """
        results = {}
        reasons = []

        # 1. Volume gate
        vol = getattr(market, "volume", 0)
        passed = vol >= self.MIN_VOLUME
        results["volume"] = passed
        if not passed:
            reasons.append(f"volume ${vol:,.0f} < ${self.MIN_VOLUME:,.0f}")

        # 2. Spread gate
        spread = getattr(market, "spread", 0)
        passed = spread <= self.MAX_SPREAD
        results["spread"] = passed
        if not passed:
            reasons.append(f"spread {spread:.3f} > {self.MAX_SPREAD:.3f}")

        # 3. Price range gate (relaxed for endgame)
        price = getattr(market, "yes_price", 0.5)
        if regime.is_endgame:
            # Endgame allows extreme prices (that's the whole point)
            passed = True
        else:
            passed = self.MIN_PRICE <= price <= self.MAX_PRICE
        results["price_range"] = passed
        if not passed:
            reasons.append(f"price {price:.3f} outside [{self.MIN_PRICE}, {self.MAX_PRICE}]")

        # 4. Minimum edge gate (regime-adjusted)
        min_edge = self.BASE_MIN_EDGE * regime.edge_multiplier
        passed = signal.adjusted_edge >= min_edge
        results["min_edge"] = passed
        if not passed:
            reasons.append(
                f"edge {signal.adjusted_edge:.4f} < {min_edge:.4f} "
                f"(base {self.BASE_MIN_EDGE:.2f} x regime {regime.edge_multiplier:.2f})"
            )

        # 5. Whale-OBI alignment gate
        # Both must have a directional opinion AND agree
        whale_has_opinion = signal.whale_direction != "NEUTRAL"
        obi_has_opinion = signal.obi_direction != "NEUTRAL"
        if whale_has_opinion and obi_has_opinion:
            passed = signal.whale_obi_aligned
        elif whale_has_opinion or obi_has_opinion:
            # If only one has an opinion, it must match the signal direction
            if whale_has_opinion:
                expected_dir = "YES" if signal.direction == "BUY_YES" else "NO"
                passed = signal.whale_direction == expected_dir
            else:
                expected_dir = "YES" if signal.direction == "BUY_YES" else "NO"
                passed = signal.obi_direction == expected_dir
        else:
            # Neither has an opinion -- fail the gate
            passed = False
        results["whale_obi_alignment"] = passed
        if not passed:
            reasons.append(
                f"whale-OBI misaligned: whale={signal.whale_direction}, "
                f"OBI={signal.obi_direction}, signal={signal.direction}"
            )

        # 6. Layer agreement gate (>= 2 out of 3)
        passed = signal.n_layers_agree >= self.MIN_LAYERS_AGREE
        results["layer_agreement"] = passed
        if not passed:
            reasons.append(
                f"only {signal.n_layers_agree}/3 layers agree "
                f"(need {self.MIN_LAYERS_AGREE})"
            )

        # 7. Drawdown circuit breaker
        drawdown_pct = risk_manager.get_drawdown_pct()
        if drawdown_pct > self.DRAWDOWN_HALT_PCT:
            self._drawdown_breach_count += 1
        else:
            self._drawdown_breach_count = 0

        if self._drawdown_breach_count >= self.DRAWDOWN_HALT_CYCLES:
            passed = False
            results["drawdown_breaker"] = False
            reasons.append(
                f"drawdown halt: {drawdown_pct:.1%} > {self.DRAWDOWN_HALT_PCT:.1%} "
                f"for {self._drawdown_breach_count} consecutive cycles"
            )
        else:
            results["drawdown_breaker"] = True

        all_passed = all(results.values())
        return all_passed, results, reasons

    def reset_drawdown_counter(self):
        """Reset the drawdown breach counter (e.g. after recovery)."""
        self._drawdown_breach_count = 0


# ---------------------------------------------------------------------------
#  6. RiskManagerV3 -- Quarter-Kelly sizing with regime/whale adjustments
# ---------------------------------------------------------------------------

class RiskManagerV3:
    """
    V3 risk management with regime-aware position sizing.

    Key parameters:
    - Quarter-Kelly base sizing (0.25 fraction)
    - Whale confidence multiplier: 0.5x - 1.5x
    - Regime size multiplier: 0.5x - 1.5x
    - Max $400 per trade (hard cap)
    - Max 40% portfolio exposure (tighter than v2's 50%)
    - Drawdown halt tracking at -8% for 3 cycles
    """

    KELLY_FRACTION: float = 0.25
    MAX_POSITION_USD: float = 400.0
    MAX_PORTFOLIO_EXPOSURE: float = 0.40  # 40% of capital
    MAX_SINGLE_POSITION_PCT: float = 0.08  # 8% of capital per trade
    MIN_WHALE_CONF_MULT: float = 0.5
    MAX_WHALE_CONF_MULT: float = 1.5

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.deployed_capital = 0.0
        self.cumulative_pnl = 0.0
        self.trade_count = 0
        self.halted = False

    def get_drawdown_pct(self) -> float:
        """Get current drawdown as a fraction of initial capital."""
        if self.initial_capital <= 0:
            return 0.0
        pnl_pct = self.cumulative_pnl / self.initial_capital
        if pnl_pct < 0:
            return abs(pnl_pct)
        return 0.0

    def update_capital(self, pnl: float):
        """Update capital after a trade resolves."""
        self.cumulative_pnl += pnl
        self.current_capital += pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)

    def compute_size(self, edge: float, entry_price: float,
                     whale_confidence: float = 0.0,
                     regime: Optional[MarketRegime] = None) -> float:
        """
        Compute position size using quarter-Kelly with adjustments.

        Args:
            edge: Expected edge (probability advantage over market price)
            entry_price: The price we'd pay for the contract
            whale_confidence: Whale confidence score [0.0, 1.0]
            regime: Market regime for size multiplier

        Returns:
            Position size in USD
        """
        if edge <= 0 or entry_price <= 0.01 or entry_price >= 0.99:
            return 0.0

        # Binary contract Kelly criterion
        # For a binary contract at price p with true probability p + edge:
        # odds = (1 - p) / p
        # f* = (odds * win_prob - lose_prob) / odds
        odds = (1.0 - entry_price) / entry_price
        win_prob = min(entry_price + edge, 0.99)
        lose_prob = 1.0 - win_prob

        if odds <= 0 or lose_prob <= 0:
            return 0.0

        full_kelly = (win_prob * odds - lose_prob) / odds
        if full_kelly <= 0:
            return 0.0

        # Apply quarter-Kelly base
        size = full_kelly * self.KELLY_FRACTION * self.current_capital

        # Apply whale confidence multiplier [0.5, 1.5]
        # Higher whale confidence -> larger position
        whale_mult = self.MIN_WHALE_CONF_MULT + (
            whale_confidence * (self.MAX_WHALE_CONF_MULT - self.MIN_WHALE_CONF_MULT)
        )
        whale_mult = np.clip(whale_mult, self.MIN_WHALE_CONF_MULT, self.MAX_WHALE_CONF_MULT)
        size *= whale_mult

        # Apply regime size multiplier
        if regime:
            size *= regime.size_multiplier

        # Apply hard caps
        size = min(size, self.MAX_POSITION_USD)
        size = min(size, self.current_capital * self.MAX_SINGLE_POSITION_PCT)

        # Check portfolio exposure limit
        available_exposure = (
            self.current_capital * self.MAX_PORTFOLIO_EXPOSURE - self.deployed_capital
        )
        if available_exposure <= 0:
            return 0.0
        size = min(size, available_exposure)

        return max(0.0, size)

    def reserve_capital(self, size: float):
        """Reserve capital for a pending trade."""
        self.deployed_capital += size
        self.trade_count += 1

    def release_capital(self, size: float):
        """Release capital when a trade resolves."""
        self.deployed_capital = max(0.0, self.deployed_capital - size)


# ---------------------------------------------------------------------------
#  7. SignalGeneratorV3 -- Top-level orchestrator
# ---------------------------------------------------------------------------

class SignalGeneratorV3:
    """
    V3 signal generator. Orchestrates whale + OBI + regime detection.

    Clean 3-layer architecture replacing v2's bloated 5-layer design.
    All broken models (MC, VR, PF, ABM, Copula) are permanently removed.

    Usage (live):
        sg = SignalGeneratorV3(config, whale_tracker=WhaleTracker())
        signals = sg.generate_signals(markets)

    Usage (backtest):
        sg = SignalGeneratorV3(config, whale_tracker=MockWhaleTracker(),
                               orderbook_scanner=MockOrderbookScanner())
        signals = sg.generate_signals(markets)
    """

    def __init__(self, config=None, whale_tracker=None, orderbook_scanner=None,
                 initial_capital: float = 10000.0):
        self.config = config or BotConfig()
        self.whale_tracker = whale_tracker
        self.prob_engine = ProbabilityEngineV3(whale_tracker=whale_tracker)
        self.regime_detector = RegimeDetector()
        self.orderbook_scanner = orderbook_scanner or OrderbookScanner()
        self.pre_filter = PreTradeFilterV3()
        self.risk_manager = RiskManagerV3(initial_capital=initial_capital)
        self.initial_capital = initial_capital
        # Tracking
        self._signals_generated = 0
        self._signals_passed = 0
        self._signals_rejected = 0

    def generate_signals(self, markets, whale_intels=None,
                         obi_signals=None) -> List[SignalV3]:
        """
        Generate signals for a batch of markets.

        Args:
            markets: List of market objects with standard attributes
            whale_intels: Optional dict of {market_id: WhaleIntelligence}
            obi_signals: Optional dict of {market_id: OrderbookSignal}

        Returns:
            List of SignalV3 that passed all gates, sorted by priority
        """
        signals = []
        all_signals = []  # Including rejected, for logging

        for market in markets:
            mid = getattr(market, "market_id", str(market))
            try:
                wi = whale_intels.get(mid) if whale_intels else None
                obi = obi_signals.get(mid) if obi_signals else None
                signal = self._compute_signal(market, whale_intel=wi, obi_signal=obi)

                if signal is not None:
                    self._signals_generated += 1
                    all_signals.append(signal)
                    if signal.passed_all_gates:
                        signals.append(signal)
                        self._signals_passed += 1
                    else:
                        self._signals_rejected += 1
            except Exception as e:
                logger.warning(f"Signal generation failed for {mid}: {e}")
                continue

        # Sort by priority: endgame first, then by adjusted edge descending
        signals.sort(
            key=lambda s: (s.regime_type == "endgame", s.adjusted_edge),
            reverse=True,
        )

        logger.info(
            f"SignalGeneratorV3: {len(markets)} markets -> "
            f"{self._signals_generated} signals generated, "
            f"{self._signals_passed} passed, {self._signals_rejected} rejected"
        )

        return signals

    def _compute_signal(self, market, whale_intel=None,
                        obi_signal=None) -> Optional[SignalV3]:
        """
        Generate a single signal through the full v3 pipeline.

        Pipeline:
        1. Detect market regime
        2. Get orderbook imbalance (if not provided)
        3. Update probability engine with all layers
        4. Compute combined probability with Bayesian shrinkage
        5. Calculate edge and direction
        6. Build signal with all evidence
        7. Classify confidence
        8. Run pre-trade gates
        9. Compute position size (if passed)
        """
        mid = getattr(market, "market_id", str(market))
        yes_price = getattr(market, "yes_price", 0.5)

        # Step 1: Detect regime
        regime = self.regime_detector.detect(market)

        # Step 2: Get OBI if not provided
        if obi_signal is None:
            try:
                token_ids = getattr(market, "clob_token_ids", [])
                if token_ids:
                    tid = token_ids[0] if isinstance(token_ids[0], str) else str(token_ids[0])
                    obi_signal = self.orderbook_scanner.scan_market(mid, tid)
            except Exception as e:
                logger.debug(f"OBI scan failed for {mid}: {e}")
                obi_signal = None

        # Step 3: Update probability engine
        self.prob_engine.update(market, whale_intel=whale_intel, obi_signal=obi_signal)

        # Step 4: Get combined probability (includes Bayesian shrinkage)
        final_prob = self.prob_engine.get_combined_probability(mid)

        # Step 5: Compute edge and direction
        edge = final_prob - yes_price
        if abs(edge) < 0.001:
            return None  # No meaningful edge

        if edge > 0:
            direction = "BUY_YES"
            trade_edge = edge
            trade_price = yes_price
        else:
            direction = "BUY_NO"
            trade_edge = -edge
            trade_price = 1.0 - yes_price

        # Apply regime edge multiplier for threshold comparison
        min_edge = 0.06 * regime.edge_multiplier
        adjusted_edge = trade_edge

        # Step 6: Build signal with all evidence
        signal = SignalV3(
            market_id=mid,
            question=getattr(market, "question", ""),
            direction=direction,
            edge=trade_edge,
            adjusted_edge=adjusted_edge,
            entry_price=trade_price,
            regime_type=regime.regime_type,
            regime_edge_mult=regime.edge_multiplier,
        )

        # Fill whale evidence
        wi = self.prob_engine.whale_intel.get(mid)
        if wi:
            signal.whale_direction = wi.whale_direction
            signal.whale_confidence = wi.whale_confidence_score
            signal.whale_strength = wi.signal_strength

        # Fill OBI evidence
        obi = self.prob_engine.obi_signals.get(mid)
        if obi:
            signal.obi_direction = obi.signal_direction
            signal.obi_imbalance = obi.imbalance_ratio
            signal.obi_strength = obi.signal_strength

        # Check whale-OBI alignment
        if signal.whale_direction != "NEUTRAL" and signal.obi_direction != "NEUTRAL":
            signal.whale_obi_aligned = (signal.whale_direction == signal.obi_direction)
        elif signal.whale_direction != "NEUTRAL" or signal.obi_direction != "NEUTRAL":
            # If only one has opinion, check if it matches signal direction
            expected = "YES" if direction == "BUY_YES" else "NO"
            if signal.whale_direction != "NEUTRAL":
                signal.whale_obi_aligned = (signal.whale_direction == expected)
            else:
                signal.whale_obi_aligned = (signal.obi_direction == expected)

        # Count agreeing layers
        n_yes, n_no, majority = self.prob_engine.get_layer_agreement(mid, yes_price)
        signal.n_layers_agree = max(n_yes, n_no)

        # Step 7: Confidence classification
        if (signal.whale_obi_aligned
                and signal.whale_strength in ("strong", "moderate")
                and adjusted_edge > 0.10
                and signal.n_layers_agree >= 3):
            signal.confidence = "high"
        elif (adjusted_edge > min_edge
              and signal.n_layers_agree >= 2
              and signal.whale_strength != "none"):
            signal.confidence = "medium"
        else:
            signal.confidence = "low"

        # Apply regime confidence boost
        if regime.confidence_boost > 0 and signal.confidence == "low":
            # Boost low to medium if regime is favorable
            if adjusted_edge > min_edge * 0.8:
                signal.confidence = "medium"

        # Step 8: Pre-trade gates
        passed, gate_results, reasons = self.pre_filter.check(
            market, signal, regime, self.risk_manager
        )
        signal.passed_all_gates = passed
        signal.rejection_reasons = reasons

        if not passed:
            return signal  # Return with rejection reasons for logging

        # Step 9: Position sizing
        size = self.risk_manager.compute_size(
            adjusted_edge,
            trade_price,
            whale_confidence=signal.whale_confidence,
            regime=regime,
        )
        signal.size_usd = size

        if size <= 0:
            signal.passed_all_gates = False
            signal.rejection_reasons.append("position size <= 0 (exposure limit or capital)")

        return signal

    def get_stats(self) -> Dict:
        """Return generation statistics."""
        return {
            "signals_generated": self._signals_generated,
            "signals_passed": self._signals_passed,
            "signals_rejected": self._signals_rejected,
            "pass_rate": (
                self._signals_passed / max(1, self._signals_generated)
            ),
            "risk_manager": {
                "current_capital": self.risk_manager.current_capital,
                "cumulative_pnl": self.risk_manager.cumulative_pnl,
                "deployed_capital": self.risk_manager.deployed_capital,
                "drawdown_pct": self.risk_manager.get_drawdown_pct(),
                "trade_count": self.risk_manager.trade_count,
            },
        }
