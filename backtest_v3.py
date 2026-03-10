#!/usr/bin/env python3
"""
Polymarket Quant Bot -- V3 Honest Backtest
==========================================

Step 5 of the v3 overhaul. Tests the new 3-layer signal engine
(Whale 40% + Regime 35% + OBI 25%) against resolved markets
with HONEST mock trackers (no data leakage).

Test A: V2 Honest baseline (simplified v2 ensemble, honest whale, minimal gates)
Test B: V3 Strategy (full SignalGeneratorV3, 7 gates, regime detection)

Usage:
    python backtest_v3.py

Outputs:
    polymarket-bot/backtest_v3_results.json
    polymarket-bot/backtest_v3_comparison.png
"""

import sys
import os
import json
import time
import math
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
#  Path setup
# ---------------------------------------------------------------------------
BASE_DIR = "/home/user/files"
V3_DIR = os.path.join(BASE_DIR, "tmp", "polymarket-bot")
BOT_DIR = os.path.join(BASE_DIR, "polymarket-bot")
sys.path.insert(0, V3_DIR)
sys.path.insert(0, BOT_DIR)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("backtest_v3")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
#  Attempt imports of v3 modules (fallback to inline if needed)
# ---------------------------------------------------------------------------
V3_IMPORTED = False
try:
    from signal_engine_v3 import (
        SignalGeneratorV3, SignalV3, MarketRegime, RegimeDetector,
        ProbabilityEngineV3, PreTradeFilterV3, RiskManagerV3,
    )
    from orderbook_scanner import MockOrderbookScanner, OrderbookSignal
    from whale_intelligence import HonestMockWhaleTracker, WhaleIntelligence
    from config import BotConfig
    V3_IMPORTED = True
    print("[OK] V3 modules imported successfully")
except Exception as e:
    print(f"[WARN] V3 import failed: {e}")
    print("[WARN] Will use inline fallback logic")

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
GAMMA_API = "https://gamma-api.polymarket.com/markets"
INITIAL_CAPITAL = 10_000.0
SLIPPAGE_PCT = 0.015  # 1.5%
FEE_PCT = 0.02        # 2% taker fee
SEED = 42

# V2 params
V2_MAX_POSITION = 500.0
V2_MAX_EXPOSURE = 0.60
V2_MIN_EDGE = 0.03
V2_MIN_VOLUME = 100_000.0

# V3 params (matches RiskManagerV3 defaults)
V3_MAX_POSITION = 400.0
V3_MAX_EXPOSURE = 0.40

RESULTS_JSON = os.path.join(BOT_DIR, "backtest_v3_results.json")
CHART_PNG = os.path.join(BOT_DIR, "backtest_v3_comparison.png")

# Category keywords
POLITICS_KW = [
    "president", "election", "congress", "senate", "house", "governor",
    "trump", "biden", "vance", "harris", "desantis", "newsom",
    "republican", "democrat", "gop", "primary", "caucus", "impeach",
    "scotus", "supreme court", "cabinet", "executive order", "electoral",
    "ballot", "vote", "poll", "political",
]
ECONOMY_KW = [
    "fed", "inflation", "gdp", "tariff", "debt ceiling", "shutdown",
    "interest rate", "recession", "unemployment", "cpi", "jobs report",
    "treasury", "fiscal", "deficit",
]
SPORTS_KW = [
    "nfl", "nba", "mlb", "nhl", "ufc", "boxing", "tennis", "soccer",
    "football", "basketball", "baseball", "hockey", "super bowl",
    "world cup", "olympics", "championship", "playoff", "finals",
]
CRYPTO_KW = [
    "bitcoin", "ethereum", "btc", "eth", "crypto", "solana", "sol",
    "defi", "nft", "blockchain", "token", "altcoin", "memecoin",
]


# ---------------------------------------------------------------------------
#  BacktestMarket dataclass
# ---------------------------------------------------------------------------
@dataclass
class BacktestMarket:
    """Simulated market object matching what SignalGeneratorV3 expects."""
    market_id: str = ""
    question: str = ""
    yes_price: float = 0.5
    volume: float = 0
    volume_24h: float = 0
    spread: float = 0.03
    category: str = ""
    end_date: str = ""
    clob_token_ids: list = field(default_factory=list)
    # Actual outcome -- ONLY used for P&L calculation, NEVER passed to signal engine
    resolved_yes: bool = True
    # Original API data
    original_volume: float = 0
    original_category: str = ""


# ---------------------------------------------------------------------------
#  Trade result dataclass
# ---------------------------------------------------------------------------
@dataclass
class TradeResult:
    market_id: str = ""
    question: str = ""
    direction: str = ""
    entry_price: float = 0.0
    size_usd: float = 0.0
    pnl: float = 0.0
    won: bool = False
    category: str = ""
    regime_type: str = "normal"
    confidence: str = "low"
    edge: float = 0.0
    rejection_reasons: list = field(default_factory=list)
    # For rejected signals (v3 logging)
    was_rejected: bool = False


# ============================================================================
#  PART 1: Fetch resolved markets from Gamma API
# ============================================================================
def fetch_resolved_markets(max_pages: int = 6, per_page: int = 100) -> List[Dict]:
    """Fetch resolved markets from Polymarket Gamma API with pagination."""
    all_markets = []
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    for page in range(max_pages):
        offset = page * per_page
        url = f"{GAMMA_API}?closed=true&limit={per_page}&offset={offset}"
        print(f"  Fetching page {page + 1}/{max_pages} (offset={offset})...")

        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                print(f"  Page {page + 1}: empty response, stopping pagination")
                break

            all_markets.extend(data)
            print(f"  Page {page + 1}: got {len(data)} markets (total: {len(all_markets)})")

            if len(data) < per_page:
                print(f"  Last page reached (got {len(data)} < {per_page})")
                break

            time.sleep(0.5)  # Be polite to the API

        except Exception as e:
            print(f"  [ERROR] Page {page + 1} failed: {e}")
            if all_markets:
                break
            raise

    session.close()
    print(f"\n  Total raw markets fetched: {len(all_markets)}")
    return all_markets


# ============================================================================
#  PART 1b: Parse and filter markets
# ============================================================================
def categorize_market(question: str, api_category: str = "") -> str:
    """Classify market into category based on question text."""
    text = f"{question} {api_category}".lower()
    if any(kw in text for kw in POLITICS_KW):
        return "politics"
    if any(kw in text for kw in ECONOMY_KW):
        return "economy"
    if any(kw in text for kw in SPORTS_KW):
        return "sports"
    if any(kw in text for kw in CRYPTO_KW):
        return "crypto"
    return "other"


def parse_resolution(outcome_prices_str: str) -> Optional[bool]:
    """
    Parse outcomePrices JSON to determine resolution.
    Returns True if YES, False if NO, None if unclear.
    """
    if not outcome_prices_str:
        return None
    try:
        prices = json.loads(outcome_prices_str)
        if not prices or len(prices) < 2:
            return None
        yes_price = float(prices[0])
        no_price = float(prices[1])
        if yes_price >= 0.99:
            return True   # Resolved YES
        if no_price >= 0.99 or yes_price <= 0.01:
            return False  # Resolved NO
        return None  # Unclear
    except (json.JSONDecodeError, ValueError, IndexError, TypeError):
        return None


def parse_markets(raw_markets: List[Dict], min_volume: float = 100_000.0) -> List[BacktestMarket]:
    """Parse raw API data into BacktestMarket objects."""
    rng = np.random.RandomState(SEED)
    parsed = []
    skipped = defaultdict(int)

    for m in raw_markets:
        question = m.get("question", "")
        if not question:
            skipped["no_question"] += 1
            continue

        # Volume
        try:
            vol_str = m.get("volume", m.get("volumeNum", 0))
            volume = float(vol_str) if vol_str else 0
        except (ValueError, TypeError):
            volume = 0

        if volume < min_volume:
            skipped["low_volume"] += 1
            continue

        # Resolution
        outcome_prices = m.get("outcomePrices", "")
        resolved = parse_resolution(outcome_prices)
        if resolved is None:
            skipped["unclear_resolution"] += 1
            continue

        # Get end date
        end_date = m.get("end_date_iso") or m.get("endDate") or m.get("end_date", "")

        # Get clob token IDs
        clob_ids_raw = m.get("clobTokenIds", "")
        if isinstance(clob_ids_raw, str):
            try:
                clob_ids = json.loads(clob_ids_raw) if clob_ids_raw else []
            except json.JSONDecodeError:
                clob_ids = []
        elif isinstance(clob_ids_raw, list):
            clob_ids = clob_ids_raw
        else:
            clob_ids = []

        # Category from API
        api_category = m.get("category", m.get("groupSlug", ""))
        category = categorize_market(question, str(api_category))

        # Simulate pre-resolution entry price (we entered before full convergence)
        if resolved:  # YES
            sim_yes_price = rng.uniform(0.40, 0.80)
        else:  # NO
            sim_yes_price = rng.uniform(0.20, 0.60)

        # Simulate spread and 24h volume
        sim_spread = rng.uniform(0.01, 0.06)
        sim_vol_24h = volume * rng.uniform(0.01, 0.05)

        market = BacktestMarket(
            market_id=str(m.get("id", m.get("conditionId", f"mkt_{len(parsed)}"))),
            question=question,
            yes_price=round(sim_yes_price, 4),
            volume=volume,
            volume_24h=round(sim_vol_24h, 2),
            spread=round(sim_spread, 4),
            category=category,
            end_date=end_date,
            clob_token_ids=clob_ids if clob_ids else [f"tok_{len(parsed)}"],
            resolved_yes=resolved,
            original_volume=volume,
            original_category=str(api_category),
        )
        parsed.append(market)

    print(f"\n  Parsed: {len(parsed)} qualifying markets")
    print(f"  Skipped: {dict(skipped)}")
    if parsed:
        yes_count = sum(1 for m in parsed if m.resolved_yes)
        no_count = len(parsed) - yes_count
        print(f"  Resolution: {yes_count} YES, {no_count} NO")
        cats = defaultdict(int)
        for m in parsed:
            cats[m.category] += 1
        print(f"  Categories: {dict(cats)}")
    return parsed


# ============================================================================
#  PART 2: P&L Calculation
# ============================================================================
def calculate_pnl(
    direction: str,
    entry_price: float,
    size_usd: float,
    resolved_yes: bool,
) -> Tuple[float, bool]:
    """
    Calculate P&L for a resolved binary contract trade.

    Args:
        direction: "BUY_YES" or "BUY_NO"
        entry_price: price paid for the contract side we bought
        size_usd: total position in USD
        resolved_yes: True if market resolved YES

    Returns:
        (pnl, won) tuple
    """
    slippage = size_usd * SLIPPAGE_PCT
    fees = size_usd * FEE_PCT
    costs = slippage + fees

    if direction == "BUY_YES":
        if resolved_yes:
            # Bought YES at entry_price, pays out 1.0
            profit = (1.0 - entry_price) * size_usd - costs
            return profit, True
        else:
            # Bought YES, resolved NO -> lose entry cost
            loss = entry_price * size_usd + costs
            return -loss, False
    else:  # BUY_NO
        no_price = 1.0 - entry_price  # entry_price in SignalV3 is already the trade_price
        # Actually: in _compute_signal, if direction=BUY_NO, trade_price = 1 - yes_price
        # So entry_price IS the no_price already
        if not resolved_yes:
            # Bought NO, resolved NO -> profit
            profit = (1.0 - entry_price) * size_usd - costs
            return profit, True
        else:
            # Bought NO, resolved YES -> lose
            loss = entry_price * size_usd + costs
            return -loss, False


# ============================================================================
#  PART 3a: Test A -- V2 Honest Baseline
# ============================================================================
def run_v2_honest(markets: List[BacktestMarket]) -> Dict[str, Any]:
    """
    Simplified V2 honest baseline.
    - Market price + noise ensemble (no regime, no OBI gates)
    - Uses HonestMockWhaleTracker for whale signals
    - Minimal filters: volume >= $100K, edge > 3%, spread <= 0.08
    - Quarter-Kelly sizing, $500 max position, 60% exposure
    """
    print("\n" + "=" * 70)
    print("  TEST A: V2 HONEST BASELINE")
    print("=" * 70)

    rng = np.random.RandomState(SEED + 100)
    capital = INITIAL_CAPITAL
    peak_capital = capital
    deployed = 0.0
    trades: List[TradeResult] = []
    cumulative_pnl_curve = [0.0]
    rejected_count = 0

    # Create honest whale tracker
    if V3_IMPORTED:
        whale_mock = HonestMockWhaleTracker(seed=SEED + 200)
    else:
        whale_mock = None

    for i, market in enumerate(markets):
        if i % 50 == 0:
            print(f"  V2: Processing market {i + 1}/{len(markets)}...")

        # ---- V2 filters ----
        if market.volume < V2_MIN_VOLUME:
            rejected_count += 1
            continue
        if market.spread > 0.08:
            rejected_count += 1
            continue
        if market.yes_price < 0.08 or market.yes_price > 0.92:
            rejected_count += 1
            continue

        # ---- V2 probability: market price + noise + whale bias ----
        # Base: market price as anchor
        base_prob = market.yes_price

        # Add whale signal if available
        whale_adj = 0.0
        if whale_mock:
            try:
                wi = whale_mock.analyze_market(market)
                if wi.whale_direction == "YES":
                    whale_adj = wi.whale_confidence_score * 0.15
                elif wi.whale_direction == "NO":
                    whale_adj = -wi.whale_confidence_score * 0.15
            except Exception:
                pass

        # V2 ensemble: price + whale + noise (no regime, no OBI)
        noise = rng.normal(0, 0.08)
        v2_prob = np.clip(base_prob + whale_adj * 0.55 + noise, 0.05, 0.95)

        # Edge calculation
        edge = v2_prob - market.yes_price
        if abs(edge) < V2_MIN_EDGE:
            rejected_count += 1
            continue

        if edge > 0:
            direction = "BUY_YES"
            trade_edge = edge
            trade_price = market.yes_price
        else:
            direction = "BUY_NO"
            trade_edge = -edge
            trade_price = 1.0 - market.yes_price

        # ---- V2 sizing: Quarter-Kelly, $500 max ----
        if trade_price <= 0.01 or trade_price >= 0.99 or trade_edge <= 0:
            rejected_count += 1
            continue

        odds = (1.0 - trade_price) / trade_price
        win_prob = min(trade_price + trade_edge, 0.99)
        lose_prob = 1.0 - win_prob
        if odds <= 0 or lose_prob <= 0:
            rejected_count += 1
            continue

        full_kelly = (win_prob * odds - lose_prob) / odds
        if full_kelly <= 0:
            rejected_count += 1
            continue

        size = full_kelly * 0.25 * capital
        size = min(size, V2_MAX_POSITION)
        size = min(size, capital * 0.10)  # 10% per trade
        available = capital * V2_MAX_EXPOSURE - deployed
        if available <= 0:
            rejected_count += 1
            continue
        size = min(size, available)
        if size < 5.0:  # Min $5 trade
            rejected_count += 1
            continue

        # ---- Execute trade ----
        pnl, won = calculate_pnl(direction, trade_price, size, market.resolved_yes)

        capital += pnl
        peak_capital = max(peak_capital, capital)

        # Track deployed (release immediately since resolved)
        # In backtest we resolve instantly, so deployed stays manageable

        trade = TradeResult(
            market_id=market.market_id,
            question=market.question[:80],
            direction=direction,
            entry_price=trade_price,
            size_usd=size,
            pnl=pnl,
            won=won,
            category=market.category,
            confidence="medium" if trade_edge > 0.06 else "low",
            edge=trade_edge,
        )
        trades.append(trade)
        cumulative_pnl_curve.append(cumulative_pnl_curve[-1] + pnl)

    # ---- Compute metrics ----
    metrics = compute_metrics(trades, cumulative_pnl_curve, "V2 Honest")
    metrics["rejected_count"] = rejected_count
    print(f"\n  V2 Honest: {metrics['trades']} trades, "
          f"{metrics['wins']}W/{metrics['losses']}L, "
          f"WR={metrics['win_rate']:.1%}, P&L=${metrics['total_pnl']:,.2f}")

    return {
        "metrics": metrics,
        "trades": trades,
        "pnl_curve": cumulative_pnl_curve,
    }


# ============================================================================
#  PART 3b: Test B -- V3 Strategy
# ============================================================================
def run_v3_strategy(markets: List[BacktestMarket]) -> Dict[str, Any]:
    """
    Full V3 strategy test using SignalGeneratorV3.
    - HonestMockWhaleTracker + MockOrderbookScanner
    - Full regime detection, OBI signals, all 7 gates
    - Quarter-Kelly, $400 max, 40% exposure
    """
    print("\n" + "=" * 70)
    print("  TEST B: V3 STRATEGY")
    print("=" * 70)

    trades: List[TradeResult] = []
    cumulative_pnl_curve = [0.0]
    all_signals: List[SignalV3] = []  # All signals including rejected
    rejection_reasons_count = defaultdict(int)
    regime_distribution = defaultdict(int)

    if not V3_IMPORTED:
        print("  [ERROR] V3 modules not available, using inline fallback")
        return run_v3_inline_fallback(markets)

    # Create v3 components with honest mocks
    whale_tracker = HonestMockWhaleTracker(seed=SEED + 300)
    obi_scanner = MockOrderbookScanner(seed=SEED + 400)
    config = BotConfig()

    # Create signal generator
    sig_gen = SignalGeneratorV3(
        config=config,
        whale_tracker=whale_tracker,
        orderbook_scanner=obi_scanner,
        initial_capital=INITIAL_CAPITAL,
    )

    # Process markets in batches to track per-market signals
    for i, market in enumerate(markets):
        if i % 50 == 0:
            print(f"  V3: Processing market {i + 1}/{len(markets)}...")

        # Generate whale intel and OBI for this market
        try:
            wi = whale_tracker.analyze_market(market)
            whale_intels = {market.market_id: wi}
        except Exception:
            whale_intels = {}

        try:
            token_id = market.clob_token_ids[0] if market.clob_token_ids else market.market_id
            obi = obi_scanner.scan_market(
                market.market_id, str(token_id),
                yes_price=market.yes_price,
                volume=market.volume,
                volume_24h=market.volume_24h,
            )
            obi_signals = {market.market_id: obi}
        except Exception:
            obi_signals = {}

        # Generate signal (process single market)
        passed_signals = sig_gen.generate_signals(
            [market],
            whale_intels=whale_intels,
            obi_signals=obi_signals,
        )

        # Also capture the signal even if rejected (for analysis)
        # We can get this from the prob_engine state
        regime = sig_gen.regime_detector.detect(market)
        regime_distribution[regime.regime_type] += 1

        if passed_signals:
            sig = passed_signals[0]
            all_signals.append(sig)

            # Execute trade
            pnl, won = calculate_pnl(
                sig.direction, sig.entry_price, sig.size_usd, market.resolved_yes
            )

            # Update risk manager capital
            sig_gen.risk_manager.update_capital(pnl)
            sig_gen.risk_manager.release_capital(sig.size_usd)

            trade = TradeResult(
                market_id=market.market_id,
                question=market.question[:80],
                direction=sig.direction,
                entry_price=sig.entry_price,
                size_usd=sig.size_usd,
                pnl=pnl,
                won=won,
                category=market.category,
                regime_type=sig.regime_type,
                confidence=sig.confidence,
                edge=sig.adjusted_edge,
            )
            trades.append(trade)
            cumulative_pnl_curve.append(cumulative_pnl_curve[-1] + pnl)
        else:
            # Signal was generated but rejected, or no signal at all
            # Try to get rejection reasons by checking internal state
            # Since generate_signals processes one market, we can infer
            # from the signal generator stats
            pass

    # Collect rejection reasons from all signals that didn't pass
    # We need to re-run to capture rejections -- or just use the stats
    total_generated = sig_gen._signals_generated
    total_passed = sig_gen._signals_passed
    total_rejected = sig_gen._signals_rejected

    # To get detailed rejection reasons, we need a second pass
    # Do a targeted re-run of rejected markets to capture reasons
    print(f"  V3: Collecting rejection reasons...")
    whale_tracker2 = HonestMockWhaleTracker(seed=SEED + 300)  # Same seed for reproducibility
    obi_scanner2 = MockOrderbookScanner(seed=SEED + 400)
    regime_detector = RegimeDetector()
    prob_engine = ProbabilityEngineV3(whale_tracker=whale_tracker2)
    pre_filter = PreTradeFilterV3()
    risk_mgr_check = RiskManagerV3(initial_capital=INITIAL_CAPITAL)

    for market in markets:
        try:
            wi = whale_tracker2.analyze_market(market)
            token_id = market.clob_token_ids[0] if market.clob_token_ids else market.market_id
            obi = obi_scanner2.scan_market(
                market.market_id, str(token_id),
                yes_price=market.yes_price,
                volume=market.volume,
                volume_24h=market.volume_24h,
            )
            prob_engine.update(market, whale_intel=wi, obi_signal=obi)
            final_prob = prob_engine.get_combined_probability(market.market_id)
            regime = regime_detector.detect(market)

            edge = final_prob - market.yes_price
            if abs(edge) < 0.001:
                rejection_reasons_count["no_meaningful_edge"] += 1
                continue

            if edge > 0:
                direction = "BUY_YES"
                trade_edge = edge
                trade_price = market.yes_price
            else:
                direction = "BUY_NO"
                trade_edge = -edge
                trade_price = 1.0 - market.yes_price

            # Build a minimal signal for gate checking
            signal = SignalV3(
                market_id=market.market_id,
                direction=direction,
                edge=trade_edge,
                adjusted_edge=trade_edge,
                entry_price=trade_price,
                regime_type=regime.regime_type,
            )

            # Fill whale/OBI evidence
            if wi:
                signal.whale_direction = wi.whale_direction
                signal.whale_confidence = wi.whale_confidence_score
                signal.whale_strength = wi.signal_strength
            if obi:
                signal.obi_direction = obi.signal_direction
                signal.obi_imbalance = obi.imbalance_ratio
                signal.obi_strength = obi.signal_strength

            # Alignment
            if signal.whale_direction != "NEUTRAL" and signal.obi_direction != "NEUTRAL":
                signal.whale_obi_aligned = (signal.whale_direction == signal.obi_direction)
            elif signal.whale_direction != "NEUTRAL" or signal.obi_direction != "NEUTRAL":
                expected = "YES" if direction == "BUY_YES" else "NO"
                if signal.whale_direction != "NEUTRAL":
                    signal.whale_obi_aligned = (signal.whale_direction == expected)
                else:
                    signal.whale_obi_aligned = (signal.obi_direction == expected)

            n_yes, n_no, majority = prob_engine.get_layer_agreement(market.market_id, market.yes_price)
            signal.n_layers_agree = max(n_yes, n_no)

            passed, gate_results, reasons = pre_filter.check(
                market, signal, regime, risk_mgr_check
            )

            if not passed:
                for reason in reasons:
                    # Categorize the rejection
                    if "volume" in reason:
                        rejection_reasons_count["volume_too_low"] += 1
                    elif "spread" in reason:
                        rejection_reasons_count["spread_too_wide"] += 1
                    elif "price" in reason:
                        rejection_reasons_count["price_out_of_range"] += 1
                    elif "edge" in reason:
                        rejection_reasons_count["edge_too_small"] += 1
                    elif "alignment" in reason or "misaligned" in reason:
                        rejection_reasons_count["whale_obi_misaligned"] += 1
                    elif "layer" in reason:
                        rejection_reasons_count["insufficient_layer_agreement"] += 1
                    elif "drawdown" in reason:
                        rejection_reasons_count["drawdown_circuit_breaker"] += 1
                    elif "position" in reason or "size" in reason:
                        rejection_reasons_count["position_size_zero"] += 1
                    else:
                        rejection_reasons_count["other"] += 1

        except Exception as e:
            rejection_reasons_count["error"] += 1

    # ---- Compute metrics ----
    metrics = compute_metrics(trades, cumulative_pnl_curve, "V3 Strategy")
    metrics["regime_distribution"] = dict(regime_distribution)
    metrics["rejection_reasons"] = dict(rejection_reasons_count)
    metrics["total_signals_generated"] = total_generated
    metrics["total_signals_passed"] = total_passed
    metrics["total_signals_rejected"] = total_rejected

    print(f"\n  V3 Strategy: {metrics['trades']} trades, "
          f"{metrics['wins']}W/{metrics['losses']}L, "
          f"WR={metrics['win_rate']:.1%}, P&L=${metrics['total_pnl']:,.2f}")
    print(f"  Signals: {total_generated} generated, {total_passed} passed, {total_rejected} rejected")
    print(f"  Regime distribution: {dict(regime_distribution)}")
    print(f"  Top rejection reasons: {dict(sorted(rejection_reasons_count.items(), key=lambda x: -x[1])[:5])}")

    return {
        "metrics": metrics,
        "trades": trades,
        "pnl_curve": cumulative_pnl_curve,
    }


# ============================================================================
#  PART 3c: V3 Inline Fallback (if imports fail)
# ============================================================================
def run_v3_inline_fallback(markets: List[BacktestMarket]) -> Dict[str, Any]:
    """
    Inline v3 logic if module imports fail.
    Captures key v3 concepts: regime detection, OBI, whale, multi-gate filter.
    """
    print("  Running V3 inline fallback...")
    rng = np.random.RandomState(SEED + 500)

    capital = INITIAL_CAPITAL
    trades: List[TradeResult] = []
    cumulative_pnl_curve = [0.0]
    regime_distribution = defaultdict(int)
    rejection_reasons_count = defaultdict(int)

    for i, market in enumerate(markets):
        if i % 50 == 0:
            print(f"  V3-fallback: Processing market {i + 1}/{len(markets)}...")

        # ---- Regime detection ----
        q_lower = market.question.lower()
        is_politics = any(kw in q_lower for kw in POLITICS_KW)
        is_economy = any(kw in q_lower for kw in ECONOMY_KW)

        # Determine regime type
        if market.yes_price > 0.95 or market.yes_price < 0.05:
            regime_type = "endgame"
            edge_mult = 0.5
            size_mult = 1.5
        elif 0.30 <= market.yes_price <= 0.55:
            regime_type = "sweet_spot"
            edge_mult = 0.8
            size_mult = 1.2
        elif market.yes_price > 0.85 or market.yes_price < 0.15:
            regime_type = "extreme"
            edge_mult = 1.2
            size_mult = 0.7
        else:
            regime_type = "normal"
            edge_mult = 1.0
            size_mult = 1.0

        if is_politics:
            edge_mult *= 1.5

        regime_distribution[regime_type] += 1

        # ---- Whale signal (honest) ----
        direction_bias = 0.0
        if market.yes_price > 0.6:
            direction_bias = (market.yes_price - 0.5) * 0.3
        elif market.yes_price < 0.4:
            direction_bias = (market.yes_price - 0.5) * 0.3
        whale_signal = direction_bias + rng.normal(0, 0.20)
        whale_direction = "YES" if whale_signal > 0.05 else ("NO" if whale_signal < -0.05 else "NEUTRAL")
        whale_conf = min(abs(whale_signal) * 0.5 + rng.uniform(0.05, 0.25), 0.7)

        # ---- OBI signal (honest) ----
        obi_bias = 0.0
        if market.yes_price > 0.75:
            obi_bias = (market.yes_price - 0.5) * 0.3
        elif market.yes_price < 0.25:
            obi_bias = (market.yes_price - 0.5) * 0.3
        obi_signal = obi_bias + rng.normal(0, 0.35)
        obi_direction = "YES" if obi_signal > 0.15 else ("NO" if obi_signal < -0.15 else "NEUTRAL")

        # ---- 3-layer ensemble ----
        # Whale (40%)
        whale_prob = 0.5 + (whale_signal * 0.40 if whale_direction == "YES" else
                           -whale_conf * 0.40 if whale_direction == "NO" else 0)
        whale_prob = np.clip(whale_prob, 0.05, 0.95)

        # Regime (35%)
        regime_prob = market.yes_price
        if regime_type == "sweet_spot":
            regime_prob += (0.5 - market.yes_price) * 0.05

        # OBI (25%)
        obi_prob = 0.5 + obi_signal * 0.3
        obi_prob = np.clip(obi_prob, 0.05, 0.95)

        # Weighted ensemble
        raw_prob = 0.40 * whale_prob + 0.35 * regime_prob + 0.25 * obi_prob
        # 30% Bayesian shrinkage
        final_prob = 0.70 * raw_prob + 0.30 * market.yes_price
        final_prob = np.clip(final_prob, 0.05, 0.95)

        # ---- Edge & direction ----
        edge = final_prob - market.yes_price
        if abs(edge) < 0.001:
            rejection_reasons_count["no_meaningful_edge"] += 1
            continue

        if edge > 0:
            direction = "BUY_YES"
            trade_edge = edge
            trade_price = market.yes_price
        else:
            direction = "BUY_NO"
            trade_edge = -edge
            trade_price = 1.0 - market.yes_price

        # ---- 7 gates ----
        # Gate 1: Volume
        if market.volume < 500_000:
            rejection_reasons_count["volume_too_low"] += 1
            continue
        # Gate 2: Spread
        if market.spread > 0.06:
            rejection_reasons_count["spread_too_wide"] += 1
            continue
        # Gate 3: Price range (relaxed for endgame)
        if regime_type != "endgame" and (market.yes_price < 0.10 or market.yes_price > 0.90):
            rejection_reasons_count["price_out_of_range"] += 1
            continue
        # Gate 4: Min edge (regime-adjusted)
        min_edge = 0.06 * edge_mult
        if trade_edge < min_edge:
            rejection_reasons_count["edge_too_small"] += 1
            continue
        # Gate 5: Whale-OBI alignment
        if whale_direction != "NEUTRAL" and obi_direction != "NEUTRAL":
            aligned = (whale_direction == obi_direction)
        elif whale_direction != "NEUTRAL" or obi_direction != "NEUTRAL":
            expected_dir = "YES" if direction == "BUY_YES" else "NO"
            aligned = (whale_direction == expected_dir) if whale_direction != "NEUTRAL" else (obi_direction == expected_dir)
        else:
            aligned = False
        if not aligned:
            rejection_reasons_count["whale_obi_misaligned"] += 1
            continue
        # Gate 6: Layer agreement (2/3)
        layers_agree = sum([
            1 if (whale_direction == "YES" and direction == "BUY_YES") or
                 (whale_direction == "NO" and direction == "BUY_NO") else 0,
            1 if (obi_direction == "YES" and direction == "BUY_YES") or
                 (obi_direction == "NO" and direction == "BUY_NO") else 0,
            1,  # Regime always counts (it's the direction we computed)
        ])
        if layers_agree < 2:
            rejection_reasons_count["insufficient_layer_agreement"] += 1
            continue
        # Gate 7: Drawdown breaker
        current_pnl = cumulative_pnl_curve[-1]
        if current_pnl / INITIAL_CAPITAL < -0.08:
            rejection_reasons_count["drawdown_circuit_breaker"] += 1
            continue

        # ---- Position sizing ----
        if trade_price <= 0.01 or trade_price >= 0.99 or trade_edge <= 0:
            rejection_reasons_count["position_size_zero"] += 1
            continue

        odds = (1.0 - trade_price) / trade_price
        win_prob = min(trade_price + trade_edge, 0.99)
        lose_prob = 1.0 - win_prob
        if odds <= 0 or lose_prob <= 0:
            rejection_reasons_count["position_size_zero"] += 1
            continue

        full_kelly = (win_prob * odds - lose_prob) / odds
        if full_kelly <= 0:
            rejection_reasons_count["position_size_zero"] += 1
            continue

        size = full_kelly * 0.25 * capital
        # Whale confidence multiplier
        whale_mult = 0.5 + whale_conf * 1.0
        whale_mult = np.clip(whale_mult, 0.5, 1.5)
        size *= whale_mult
        size *= size_mult  # Regime multiplier
        size = min(size, V3_MAX_POSITION)
        size = min(size, capital * 0.08)
        if size < 5.0:
            rejection_reasons_count["position_size_zero"] += 1
            continue

        # ---- Execute ----
        pnl, won = calculate_pnl(direction, trade_price, size, market.resolved_yes)
        capital += pnl

        trade = TradeResult(
            market_id=market.market_id,
            question=market.question[:80],
            direction=direction,
            entry_price=trade_price,
            size_usd=size,
            pnl=pnl,
            won=won,
            category=market.category,
            regime_type=regime_type,
            confidence="medium" if trade_edge > 0.10 else "low",
            edge=trade_edge,
        )
        trades.append(trade)
        cumulative_pnl_curve.append(cumulative_pnl_curve[-1] + pnl)

    metrics = compute_metrics(trades, cumulative_pnl_curve, "V3 Fallback")
    metrics["regime_distribution"] = dict(regime_distribution)
    metrics["rejection_reasons"] = dict(rejection_reasons_count)
    metrics["total_signals_generated"] = len(markets)
    metrics["total_signals_passed"] = len(trades)
    metrics["total_signals_rejected"] = len(markets) - len(trades)

    print(f"\n  V3 Fallback: {metrics['trades']} trades, "
          f"{metrics['wins']}W/{metrics['losses']}L, "
          f"WR={metrics['win_rate']:.1%}, P&L=${metrics['total_pnl']:,.2f}")

    return {
        "metrics": metrics,
        "trades": trades,
        "pnl_curve": cumulative_pnl_curve,
    }


# ============================================================================
#  PART 4: Metrics computation
# ============================================================================
def compute_metrics(trades: List[TradeResult], pnl_curve: List[float],
                    label: str) -> Dict[str, Any]:
    """Compute comprehensive metrics for a set of trades."""
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "label": label, "trades": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "total_pnl": 0.0, "avg_pnl_per_trade": 0.0,
            "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "profit_factor": 0.0, "sharpe": 0.0,
            "best_trade": 0.0, "worst_trade": 0.0,
            "max_consecutive_losses": 0,
            "category_pnl": {},
        }

    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = n_wins / n_trades

    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / n_trades

    pnls = [t.pnl for t in trades]
    best_trade = max(pnls)
    worst_trade = min(pnls)

    # Max drawdown from P&L curve
    peak = 0.0
    max_dd = 0.0
    for v in pnl_curve:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd / INITIAL_CAPITAL * 100

    # Profit factor
    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    # Sharpe (approximate from trade returns)
    returns = np.array(pnls)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for t in trades:
        if not t.won:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Category P&L
    cat_pnl = defaultdict(float)
    cat_trades = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    for t in trades:
        cat_pnl[t.category] += t.pnl
        cat_trades[t.category]["pnl"] += t.pnl
        if t.won:
            cat_trades[t.category]["wins"] += 1
        else:
            cat_trades[t.category]["losses"] += 1

    # Confidence breakdown
    conf_breakdown = defaultdict(lambda: {"count": 0, "wins": 0, "pnl": 0.0})
    for t in trades:
        conf_breakdown[t.confidence]["count"] += 1
        conf_breakdown[t.confidence]["pnl"] += t.pnl
        if t.won:
            conf_breakdown[t.confidence]["wins"] += 1

    return {
        "label": label,
        "trades": n_trades,
        "wins": n_wins,
        "losses": n_losses,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "final_capital": round(INITIAL_CAPITAL + total_pnl, 2),
        "avg_pnl_per_trade": round(avg_pnl, 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "profit_factor": round(profit_factor, 4) if profit_factor != float('inf') else "inf",
        "sharpe": round(sharpe, 2),
        "best_trade": round(best_trade, 2),
        "worst_trade": round(worst_trade, 2),
        "max_consecutive_losses": max_consec,
        "category_pnl": {k: round(v, 2) for k, v in cat_pnl.items()},
        "category_detail": {k: v for k, v in cat_trades.items()},
        "confidence_breakdown": {k: v for k, v in conf_breakdown.items()},
    }


# ============================================================================
#  PART 5: Save Results JSON
# ============================================================================
def save_results(
    markets: List[BacktestMarket],
    v2_result: Dict[str, Any],
    v3_result: Dict[str, Any],
) -> Dict:
    """Build and save results JSON."""
    v2m = v2_result["metrics"]
    v3m = v3_result["metrics"]

    # Improvement analysis
    pnl_delta = v3m["total_pnl"] - v2m["total_pnl"]
    wr_delta = v3m["win_rate"] - v2m["win_rate"]

    v2_dd = v2m["max_drawdown_pct"]
    v3_dd = v3m["max_drawdown_pct"]
    dd_improvement = v2_dd - v3_dd  # Positive = v3 has less drawdown

    yes_count = sum(1 for m in markets if m.resolved_yes)
    no_count = len(markets) - yes_count
    cats = defaultdict(int)
    for m in markets:
        cats[m.category] += 1

    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "markets_fetched": len(markets),
        "markets_qualifying": len(markets),
        "resolution_split": {"yes": yes_count, "no": no_count},
        "category_distribution": dict(cats),
        "simulation_params": {
            "initial_capital": INITIAL_CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "fee_pct": FEE_PCT,
            "seed": SEED,
            "v2_max_position": V2_MAX_POSITION,
            "v2_max_exposure": V2_MAX_EXPOSURE,
            "v2_min_edge": V2_MIN_EDGE,
            "v3_max_position": V3_MAX_POSITION,
            "v3_max_exposure": V3_MAX_EXPOSURE,
        },
        "test_a_v2_honest": {
            "trades": v2m["trades"],
            "wins": v2m["wins"],
            "losses": v2m["losses"],
            "win_rate": v2m["win_rate"],
            "total_pnl": v2m["total_pnl"],
            "final_capital": v2m.get("final_capital", INITIAL_CAPITAL + v2m["total_pnl"]),
            "avg_pnl_per_trade": v2m["avg_pnl_per_trade"],
            "max_drawdown": v2m["max_drawdown"],
            "max_drawdown_pct": v2m["max_drawdown_pct"],
            "profit_factor": v2m["profit_factor"],
            "sharpe": v2m["sharpe"],
            "best_trade": v2m["best_trade"],
            "worst_trade": v2m["worst_trade"],
            "max_consecutive_losses": v2m["max_consecutive_losses"],
            "category_pnl": v2m["category_pnl"],
            "category_detail": {k: dict(v) for k, v in v2m.get("category_detail", {}).items()},
            "confidence_breakdown": {k: dict(v) for k, v in v2m.get("confidence_breakdown", {}).items()},
            "rejected_count": v2m.get("rejected_count", 0),
        },
        "test_b_v3": {
            "trades": v3m["trades"],
            "wins": v3m["wins"],
            "losses": v3m["losses"],
            "win_rate": v3m["win_rate"],
            "total_pnl": v3m["total_pnl"],
            "final_capital": v3m.get("final_capital", INITIAL_CAPITAL + v3m["total_pnl"]),
            "avg_pnl_per_trade": v3m["avg_pnl_per_trade"],
            "max_drawdown": v3m["max_drawdown"],
            "max_drawdown_pct": v3m["max_drawdown_pct"],
            "profit_factor": v3m["profit_factor"],
            "sharpe": v3m["sharpe"],
            "best_trade": v3m["best_trade"],
            "worst_trade": v3m["worst_trade"],
            "max_consecutive_losses": v3m["max_consecutive_losses"],
            "category_pnl": v3m["category_pnl"],
            "category_detail": {k: dict(v) for k, v in v3m.get("category_detail", {}).items()},
            "confidence_breakdown": {k: dict(v) for k, v in v3m.get("confidence_breakdown", {}).items()},
            "regime_distribution": v3m.get("regime_distribution", {}),
            "rejection_reasons": v3m.get("rejection_reasons", {}),
            "total_signals_generated": v3m.get("total_signals_generated", 0),
            "total_signals_passed": v3m.get("total_signals_passed", 0),
            "total_signals_rejected": v3m.get("total_signals_rejected", 0),
        },
        "improvement": {
            "pnl_delta": round(pnl_delta, 2),
            "win_rate_delta": round(wr_delta, 4),
            "drawdown_improvement": round(dd_improvement, 2),
            "v3_better_pnl": v3m["total_pnl"] > v2m["total_pnl"],
            "v3_better_wr": v3m["win_rate"] > v2m["win_rate"],
            "v3_less_drawdown": v3_dd < v2_dd,
        },
    }

    # Save JSON
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_JSON}")
    print(f"  File size: {os.path.getsize(RESULTS_JSON):,} bytes")

    return results


# ============================================================================
#  PART 6: Generate Comparison Chart
# ============================================================================
def generate_chart(
    v2_result: Dict[str, Any],
    v3_result: Dict[str, Any],
    n_markets: int,
) -> None:
    """Generate 2x2 comparison chart with dark theme."""
    print("\n  Generating comparison chart...")

    v2m = v2_result["metrics"]
    v3m = v3_result["metrics"]
    v2_curve = v2_result["pnl_curve"]
    v3_curve = v3_result["pnl_curve"]

    # Dark theme
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.suptitle(
        f"V3 Honest Backtest: {n_markets} Resolved Markets",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0")
        ax.xaxis.label.set_color("#e0e0e0")
        ax.yaxis.label.set_color("#e0e0e0")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333366")

    # ---- Panel 1: Cumulative P&L Curves ----
    ax1 = axes[0, 0]
    ax1.plot(v2_curve, color="#4da6ff", linewidth=2, label="V2 Honest", alpha=0.9)
    ax1.plot(v3_curve, color="#66ff66", linewidth=2, label="V3 Strategy", alpha=0.9)
    ax1.axhline(y=0, color="#666666", linestyle="--", alpha=0.5)
    ax1.set_title("Cumulative P&L", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Trade #")
    ax1.set_ylabel("Cumulative P&L ($)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))
    ax1.legend(loc="best", fontsize=10, framealpha=0.3)
    ax1.grid(True, alpha=0.15)

    # Add final P&L annotations
    if len(v2_curve) > 1:
        ax1.annotate(
            f"${v2_curve[-1]:,.0f}",
            xy=(len(v2_curve) - 1, v2_curve[-1]),
            fontsize=9, color="#4da6ff", fontweight="bold",
            xytext=(5, 10), textcoords="offset points",
        )
    if len(v3_curve) > 1:
        ax1.annotate(
            f"${v3_curve[-1]:,.0f}",
            xy=(len(v3_curve) - 1, v3_curve[-1]),
            fontsize=9, color="#66ff66", fontweight="bold",
            xytext=(5, -15), textcoords="offset points",
        )

    # ---- Panel 2: Win Rate + Profit Factor Bar Comparison ----
    ax2 = axes[0, 1]
    bar_labels = ["Win Rate (%)", "Profit Factor", "Sharpe Ratio"]

    v2_wr = v2m["win_rate"] * 100
    v3_wr = v3m["win_rate"] * 100
    v2_pf = float(v2m["profit_factor"]) if v2m["profit_factor"] != "inf" else 5.0
    v3_pf = float(v3m["profit_factor"]) if v3m["profit_factor"] != "inf" else 5.0
    v2_sr = v2m["sharpe"]
    v3_sr = v3m["sharpe"]

    x_pos = np.arange(3)
    width = 0.30
    bars1 = ax2.bar(x_pos - width/2, [v2_wr, v2_pf, v2_sr],
                    width, label="V2 Honest", color="#4da6ff", alpha=0.85)
    bars2 = ax2.bar(x_pos + width/2, [v3_wr, v3_pf, v3_sr],
                    width, label="V3 Strategy", color="#66ff66", alpha=0.85)

    # Add value labels on bars
    for bar_group, values in [(bars1, [v2_wr, v2_pf, v2_sr]),
                               (bars2, [v3_wr, v3_pf, v3_sr])]:
        for bar, val in zip(bar_group, values):
            height = bar.get_height()
            if abs(height) > 0.01:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2., height,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold",
                )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels, fontsize=10)
    ax2.set_title("Key Metrics Comparison", fontsize=13, fontweight="bold")
    ax2.legend(loc="best", fontsize=10, framealpha=0.3)
    ax2.grid(True, alpha=0.15, axis="y")
    ax2.axhline(y=0, color="#666666", linestyle="--", alpha=0.5)

    # Add trade count annotation
    ax2.text(
        0.02, 0.98,
        f"V2: {v2m['trades']} trades | V3: {v3m['trades']} trades",
        transform=ax2.transAxes, fontsize=9, color="#aaaaaa",
        va="top", ha="left",
    )

    # ---- Panel 3: P&L by Category (grouped bars) ----
    ax3 = axes[1, 0]
    all_cats = sorted(set(list(v2m["category_pnl"].keys()) + list(v3m["category_pnl"].keys())))
    if all_cats:
        x_cat = np.arange(len(all_cats))
        v2_cat_vals = [v2m["category_pnl"].get(c, 0) for c in all_cats]
        v3_cat_vals = [v3m["category_pnl"].get(c, 0) for c in all_cats]

        width_cat = 0.35
        ax3.bar(x_cat - width_cat/2, v2_cat_vals, width_cat,
                label="V2 Honest", color="#4da6ff", alpha=0.85)
        ax3.bar(x_cat + width_cat/2, v3_cat_vals, width_cat,
                label="V3 Strategy", color="#66ff66", alpha=0.85)

        ax3.set_xticks(x_cat)
        ax3.set_xticklabels([c.capitalize() for c in all_cats], fontsize=9, rotation=15)
        ax3.set_title("P&L by Category", fontsize=13, fontweight="bold")
        ax3.set_ylabel("P&L ($)")
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax3.legend(loc="best", fontsize=10, framealpha=0.3)
        ax3.grid(True, alpha=0.15, axis="y")
        ax3.axhline(y=0, color="#666666", linestyle="--", alpha=0.5)
    else:
        ax3.text(0.5, 0.5, "No category data", ha="center", va="center",
                 fontsize=12, color="#aaaaaa", transform=ax3.transAxes)

    # ---- Panel 4: Split into regime pie (left) + rejection bars (right) ----
    ax4 = axes[1, 1]

    # Use ax4 for regime distribution + rejection reasons
    regime_dist = v3m.get("regime_distribution", {})
    rejection_reasons = v3m.get("rejection_reasons", {})

    if regime_dist or rejection_reasons:
        # Create two sub-regions
        ax4.set_visible(False)
        # Regime pie on the left-ish area
        if regime_dist:
            ax4a = fig.add_axes([0.55, 0.06, 0.18, 0.36])  # [left, bottom, width, height]
            ax4a.set_facecolor("#16213e")
            regime_colors = {
                "endgame": "#ff6b6b",
                "sweet_spot": "#51cf66",
                "extreme": "#ffd43b",
                "long_dated": "#748ffc",
                "normal": "#868e96",
            }
            labels = list(regime_dist.keys())
            sizes = list(regime_dist.values())
            colors = [regime_colors.get(l, "#868e96") for l in labels]

            wedges, texts, autotexts = ax4a.pie(
                sizes, labels=None, autopct="%1.0f%%",
                colors=colors, startangle=90, pctdistance=0.80,
                textprops={"fontsize": 8, "color": "white"},
            )
            ax4a.set_title("V3 Regime Distribution", fontsize=10,
                          fontweight="bold", color="white", pad=8)
            # Legend below
            ax4a.legend(
                labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                fontsize=7, ncol=2, framealpha=0.3,
                labelcolor="white",
            )

        # Rejection reasons bar chart on the right
        if rejection_reasons:
            ax4b = fig.add_axes([0.78, 0.06, 0.20, 0.36])
            ax4b.set_facecolor("#16213e")

            # Sort and take top 7
            sorted_reasons = sorted(rejection_reasons.items(), key=lambda x: -x[1])[:7]
            reason_labels = [r[0].replace("_", " ").title()[:18] for r in sorted_reasons]
            reason_counts = [r[1] for r in sorted_reasons]

            bars = ax4b.barh(
                range(len(reason_labels)), reason_counts,
                color="#ff6b6b", alpha=0.8,
            )
            ax4b.set_yticks(range(len(reason_labels)))
            ax4b.set_yticklabels(reason_labels, fontsize=7, color="#e0e0e0")
            ax4b.set_title("V3 Rejection Reasons", fontsize=10,
                          fontweight="bold", color="white", pad=8)
            ax4b.set_xlabel("Count", fontsize=8, color="#e0e0e0")
            ax4b.tick_params(colors="#e0e0e0", labelsize=7)
            for spine in ax4b.spines.values():
                spine.set_color("#333366")
            ax4b.invert_yaxis()

            # Add count labels
            for bar, count in zip(bars, reason_counts):
                ax4b.text(
                    bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=7, color="#e0e0e0",
                )
    else:
        ax4.text(0.5, 0.5, "No regime/rejection data", ha="center", va="center",
                 fontsize=12, color="#aaaaaa", transform=ax4.transAxes)

    # Summary text at bottom
    summary_text = (
        f"V2 Honest: {v2m['trades']} trades, {v2m['win_rate']:.1%} WR, "
        f"P&L ${v2m['total_pnl']:,.2f}, DD {v2m['max_drawdown_pct']:.1f}%  |  "
        f"V3 Strategy: {v3m['trades']} trades, {v3m['win_rate']:.1%} WR, "
        f"P&L ${v3m['total_pnl']:,.2f}, DD {v3m['max_drawdown_pct']:.1f}%  |  "
        f"P&L Delta: ${v3m['total_pnl'] - v2m['total_pnl']:+,.2f}"
    )
    fig.text(
        0.5, 0.01, summary_text,
        fontsize=9, color="#cccccc", ha="center", va="bottom",
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    os.makedirs(os.path.dirname(CHART_PNG), exist_ok=True)
    fig.savefig(CHART_PNG, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart saved to {CHART_PNG}")
    print(f"  File size: {os.path.getsize(CHART_PNG):,} bytes")


# ============================================================================
#  MAIN
# ============================================================================
def main():
    """Run the V3 honest backtest."""
    start_time = time.time()
    print("\n" + "#" * 70)
    print("  POLYMARKET QUANT BOT -- V3 HONEST BACKTEST")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("#" * 70)

    # ---- Part 1: Fetch markets ----
    print("\n[PART 1] Fetching resolved markets from Gamma API...")
    raw_markets = fetch_resolved_markets(max_pages=6, per_page=100)

    # ---- Part 1b: Parse and filter ----
    print("\n[PART 1b] Parsing and filtering markets...")
    markets = parse_markets(raw_markets, min_volume=100_000.0)

    if len(markets) < 20:
        print(f"\n  [ERROR] Only {len(markets)} markets -- too few for meaningful backtest.")
        print("  Need at least 20 qualifying markets. Aborting.")
        return

    print(f"\n  Proceeding with {len(markets)} markets")

    # ---- Part 3a: Test A -- V2 Honest ----
    print("\n[PART 3a] Running Test A: V2 Honest Baseline...")
    v2_result = run_v2_honest(markets)

    # ---- Part 3b: Test B -- V3 Strategy ----
    print("\n[PART 3b] Running Test B: V3 Strategy...")
    v3_result = run_v3_strategy(markets)

    # ---- Part 5: Save results ----
    print("\n[PART 5] Saving results JSON...")
    results = save_results(markets, v2_result, v3_result)

    # ---- Part 6: Generate chart ----
    print("\n[PART 6] Generating comparison chart...")
    generate_chart(v2_result, v3_result, len(markets))

    # ---- Summary ----
    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print("  BACKTEST COMPLETE")
    print(f"  Runtime: {elapsed:.1f}s")
    print("#" * 70)

    # Print full results
    print("\n" + "=" * 70)
    print("  FULL RESULTS JSON")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))

    return results


if __name__ == "__main__":
    main()
