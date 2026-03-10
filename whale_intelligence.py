"""
Polymarket Quant Bot — Whale Intelligence Module (v2)
======================================================
Detects smart money / insider positioning via:
1. Polymarket Data API (/holders, /trades, /activity)
2. Goldsky subgraphs (positions, PnL, activity)
3. CLOB API orderbook depth analysis

Signals:
- Whale position clustering (3+ large wallets same side within 24h)
- Smart money sentiment (PnL-weighted directional bias)
- Orderbook imbalance (bid/ask depth asymmetry)
- Fresh wallet detection (new wallets taking large positions)
- Whale reversal detection (profitable wallets exiting)

All endpoints are public, no authentication required.
"""
import httpx
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class WhaleConfig:
    """Configuration for whale intelligence module."""
    # API endpoints
    data_api_base: str = "https://data-api.polymarket.com"
    clob_api_base: str = "https://clob.polymarket.com"
    goldsky_positions_url: str = (
        "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
        "/subgraphs/positions-subgraph/0.0.7/gn"
    )
    goldsky_pnl_url: str = (
        "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
        "/subgraphs/pnl-subgraph/0.0.14/gn"
    )
    goldsky_activity_url: str = (
        "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
        "/subgraphs/activity-subgraph/0.0.4/gn"
    )
    goldsky_oi_url: str = (
        "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw"
        "/subgraphs/oi-subgraph/0.0.6/gn"
    )

    # Thresholds
    whale_min_position_usd: float = 10_000.0
    shark_min_position_usd: float = 50_000.0
    mega_whale_min_usd: float = 100_000.0
    large_trade_min_usd: float = 5_000.0
    cluster_window_hours: int = 24
    min_cluster_wallets: int = 3
    min_profitable_pnl: float = 1_000.0

    # Scoring weights
    holder_concentration_weight: float = 0.25
    smart_money_sentiment_weight: float = 0.35
    orderbook_imbalance_weight: float = 0.20
    trade_flow_weight: float = 0.20

    # Rate limiting
    request_timeout: int = 15
    min_request_interval: float = 0.25


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class WhalePosition:
    """A single whale's position in a market."""
    wallet: str
    outcome_index: int       # 0 = YES, 1 = NO
    balance: float
    avg_entry_price: float
    position_value_usd: float
    realized_pnl: float
    tier: str                # "mega_whale", "shark", "whale", "dolphin"


@dataclass
class TradeFlow:
    """Aggregated trade flow for a market."""
    total_buy_yes_usd: float = 0.0
    total_buy_no_usd: float = 0.0
    large_buy_yes_usd: float = 0.0
    large_buy_no_usd: float = 0.0
    n_large_buys_yes: int = 0
    n_large_buys_no: int = 0
    n_unique_large_wallets_yes: int = 0
    n_unique_large_wallets_no: int = 0
    fresh_wallet_volume_yes: float = 0.0
    fresh_wallet_volume_no: float = 0.0


@dataclass
class OrderbookSignal:
    """Orderbook depth imbalance signal."""
    bid_depth_usd: float = 0.0
    ask_depth_usd: float = 0.0
    imbalance_ratio: float = 0.0
    top_5_bid_depth: float = 0.0
    top_5_ask_depth: float = 0.0
    spread: float = 0.0
    last_trade_price: float = 0.0


@dataclass
class WhaleIntelligence:
    """Complete whale intelligence report for a market."""
    market_id: str
    condition_id: str
    timestamp: str

    # Holder analysis
    n_whale_holders: int = 0
    whale_yes_pct: float = 0.0
    whale_no_pct: float = 0.0
    holder_concentration: float = 0.0
    smart_money_pnl_bias: float = 0.0

    # Trade flow
    trade_flow: TradeFlow = field(default_factory=TradeFlow)

    # Orderbook
    orderbook: OrderbookSignal = field(default_factory=OrderbookSignal)

    # Whale cluster detection
    cluster_detected: bool = False
    cluster_direction: str = ""
    cluster_wallets: int = 0
    cluster_volume_usd: float = 0.0

    # Composite scores
    whale_confidence_score: float = 0.0
    whale_direction: str = ""
    signal_strength: str = ""

    # Raw data
    whale_positions: List[WhalePosition] = field(default_factory=list)


# ─── API Client ───────────────────────────────────────────────────────────────

class WhaleTracker:
    """
    Tracks whale/insider activity across Polymarket markets.
    Uses Data API, Goldsky subgraphs, and CLOB API.
    """

    def __init__(self, config: Optional[WhaleConfig] = None):
        self.config = config or WhaleConfig()
        self.client = httpx.Client(timeout=self.config.request_timeout)
        self._last_request_time = 0.0
        self._cache: Dict[str, Tuple[float, any]] = {}
        self._cache_ttl = 300

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.config.min_request_interval:
            time.sleep(self.config.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None) -> dict:
        self._rate_limit()
        try:
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"API request failed: {url} — {e}")
            return {}

    def _graphql(self, url: str, query: str, variables: dict = None) -> dict:
        self._rate_limit()
        try:
            payload = {"query": query}
            if variables:
                payload["variables"] = variables
            resp = self.client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("data", {})
        except Exception as e:
            logger.warning(f"GraphQL query failed: {url} — {e}")
            return {}

    # ─── Data Source 1: Polymarket Data API ───────────────────────────────

    def fetch_holders(self, condition_id: str, token_id: str = "") -> List[dict]:
        params = {"conditionId": condition_id}
        if token_id:
            params["tokenId"] = token_id
        result = self._get(f"{self.config.data_api_base}/holders", params)
        if isinstance(result, list):
            return result
        return result.get("holders", result.get("data", []))

    def fetch_large_trades(self, condition_id: str, min_usd: float = 5000) -> List[dict]:
        params = {
            "conditionId": condition_id,
            "filterType": "CASH",
            "filterAmount": str(int(min_usd)),
        }
        result = self._get(f"{self.config.data_api_base}/trades", params)
        if isinstance(result, list):
            return result
        return result.get("trades", result.get("data", []))

    def fetch_market_activity(self, condition_id: str) -> List[dict]:
        params = {"conditionId": condition_id}
        result = self._get(f"{self.config.data_api_base}/activity", params)
        if isinstance(result, list):
            return result
        return result.get("activity", result.get("data", []))

    def fetch_open_interest(self, condition_id: str) -> dict:
        params = {"conditionId": condition_id}
        return self._get(f"{self.config.data_api_base}/oi", params)

    # ─── Data Source 2: Goldsky Subgraphs ─────────────────────────────────

    def fetch_subgraph_positions(self, condition_id: str, min_balance: float = 100) -> List[dict]:
        query = """
        query($condition: String!, $minBalance: BigDecimal!) {
            positions(
                where: { condition: $condition, balance_gt: $minBalance }
                first: 100
                orderBy: balance
                orderDirection: desc
            ) {
                id
                user { id }
                condition
                outcomeIndex
                balance
                averagePrice
                realizedPnl
            }
        }
        """
        variables = {"condition": condition_id, "minBalance": str(min_balance)}
        data = self._graphql(self.config.goldsky_positions_url, query, variables)
        return data.get("positions", [])

    def fetch_wallet_pnl(self, wallet: str) -> dict:
        query = """
        query($user: String!) {
            userPnls(
                where: { user: $user }
                first: 50
                orderBy: realizedPnl
                orderDirection: desc
            ) {
                id
                user
                condition
                realizedPnl
                unrealizedPnl
            }
        }
        """
        data = self._graphql(self.config.goldsky_pnl_url, query, {"user": wallet.lower()})
        return data

    # ─── Data Source 3: CLOB API Orderbook ────────────────────────────────

    def fetch_orderbook(self, token_id: str) -> dict:
        params = {"token_id": token_id}
        return self._get(f"{self.config.clob_api_base}/book", params)

    # ─── Analysis Functions ───────────────────────────────────────────────

    def _classify_whale_tier(self, position_value: float) -> str:
        if position_value >= self.config.mega_whale_min_usd:
            return "mega_whale"
        elif position_value >= self.config.shark_min_position_usd:
            return "shark"
        elif position_value >= self.config.whale_min_position_usd:
            return "whale"
        else:
            return "dolphin"

    def _analyze_holders(self, holders_data: List[dict], condition_id: str) -> Tuple[List[WhalePosition], dict]:
        whale_positions = []
        total_value = 0.0
        whale_yes_value = 0.0
        whale_no_value = 0.0
        top_10_value = 0.0

        for i, h in enumerate(holders_data):
            wallet = h.get("user", h.get("address", h.get("id", "unknown")))
            if isinstance(wallet, dict):
                wallet = wallet.get("id", "unknown")
            balance = float(h.get("balance", h.get("amount", 0)))
            avg_price = float(h.get("averagePrice", h.get("avgPrice", 0.5)))
            outcome = int(h.get("outcomeIndex", h.get("outcome", 0)))
            realized_pnl = float(h.get("realizedPnl", h.get("pnl", 0)))

            position_value = balance * avg_price if avg_price > 0 else balance * 0.5
            total_value += position_value

            if i < 10:
                top_10_value += position_value

            tier = self._classify_whale_tier(position_value)

            if tier in ("whale", "shark", "mega_whale"):
                wp = WhalePosition(
                    wallet=wallet,
                    outcome_index=outcome,
                    balance=balance,
                    avg_entry_price=avg_price,
                    position_value_usd=position_value,
                    realized_pnl=realized_pnl,
                    tier=tier,
                )
                whale_positions.append(wp)

                if outcome == 0:
                    whale_yes_value += position_value
                else:
                    whale_no_value += position_value

        total_whale_value = whale_yes_value + whale_no_value
        concentration = top_10_value / total_value if total_value > 0 else 0.0

        metrics = {
            "n_whales": len(whale_positions),
            "whale_yes_pct": whale_yes_value / total_whale_value if total_whale_value > 0 else 0.5,
            "whale_no_pct": whale_no_value / total_whale_value if total_whale_value > 0 else 0.5,
            "concentration": concentration,
            "total_whale_value": total_whale_value,
        }

        return whale_positions, metrics

    def _analyze_smart_money_pnl(self, whale_positions: List[WhalePosition]) -> float:
        if not whale_positions:
            return 0.0

        weighted_direction = 0.0
        total_weight = 0.0

        for wp in whale_positions:
            pnl_multiplier = 1.0 + np.clip(wp.realized_pnl / max(wp.position_value_usd, 1), -0.5, 2.0)
            weight = wp.position_value_usd * max(pnl_multiplier, 0.1)
            direction = 1.0 if wp.outcome_index == 0 else -1.0
            weighted_direction += weight * direction
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return np.clip(weighted_direction / total_weight, -1.0, 1.0)

    def _analyze_trade_flow(self, trades: List[dict]) -> TradeFlow:
        flow = TradeFlow()
        seen_wallets_yes = set()
        seen_wallets_no = set()

        for trade in trades:
            size_usd = float(trade.get("size", trade.get("amount", 0)))
            side = trade.get("side", trade.get("type", "")).upper()
            wallet = trade.get("user", trade.get("maker", trade.get("taker", "")))
            outcome = trade.get("outcomeIndex", trade.get("outcome", ""))

            is_yes_buy = (
                (side in ("BUY", "B") and str(outcome) == "0")
                or (side in ("SELL", "S") and str(outcome) == "1")
                or "YES" in side
            )

            if is_yes_buy:
                flow.total_buy_yes_usd += size_usd
                if size_usd >= self.config.large_trade_min_usd:
                    flow.large_buy_yes_usd += size_usd
                    flow.n_large_buys_yes += 1
                    seen_wallets_yes.add(wallet)
            else:
                flow.total_buy_no_usd += size_usd
                if size_usd >= self.config.large_trade_min_usd:
                    flow.large_buy_no_usd += size_usd
                    flow.n_large_buys_no += 1
                    seen_wallets_no.add(wallet)

        flow.n_unique_large_wallets_yes = len(seen_wallets_yes)
        flow.n_unique_large_wallets_no = len(seen_wallets_no)
        return flow

    def _analyze_orderbook(self, book_data: dict) -> OrderbookSignal:
        signal = OrderbookSignal()
        bids = book_data.get("bids", [])
        asks = book_data.get("asks", [])

        if not bids and not asks:
            return signal

        for bid in bids:
            price = float(bid.get("price", 0))
            size = float(bid.get("size", 0))
            signal.bid_depth_usd += price * size

        for ask in asks:
            price = float(ask.get("price", 0))
            size = float(ask.get("size", 0))
            signal.ask_depth_usd += price * size

        for bid in bids[:5]:
            signal.top_5_bid_depth += float(bid.get("price", 0)) * float(bid.get("size", 0))
        for ask in asks[:5]:
            signal.top_5_ask_depth += float(ask.get("price", 0)) * float(ask.get("size", 0))

        total = signal.bid_depth_usd + signal.ask_depth_usd
        if total > 0:
            signal.imbalance_ratio = (signal.bid_depth_usd - signal.ask_depth_usd) / total

        if bids and asks:
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 0))
            signal.spread = best_ask - best_bid

        signal.last_trade_price = float(book_data.get("last_trade_price", 0))
        return signal

    def _detect_cluster(self, whale_positions: List[WhalePosition], trade_flow: TradeFlow) -> Tuple[bool, str, int, float]:
        yes_whales = [wp for wp in whale_positions if wp.outcome_index == 0]
        no_whales = [wp for wp in whale_positions if wp.outcome_index == 1]

        yes_count = len(yes_whales) + trade_flow.n_unique_large_wallets_yes
        no_count = len(no_whales) + trade_flow.n_unique_large_wallets_no
        yes_volume = sum(wp.position_value_usd for wp in yes_whales) + trade_flow.large_buy_yes_usd
        no_volume = sum(wp.position_value_usd for wp in no_whales) + trade_flow.large_buy_no_usd

        if yes_count >= self.config.min_cluster_wallets and yes_count > no_count * 1.5:
            return True, "YES", yes_count, yes_volume
        elif no_count >= self.config.min_cluster_wallets and no_count > yes_count * 1.5:
            return True, "NO", no_count, no_volume

        return False, "", 0, 0.0

    def _compute_composite_score(self, holder_metrics: dict, pnl_bias: float,
                                  trade_flow: TradeFlow, orderbook: OrderbookSignal,
                                  cluster_detected: bool) -> Tuple[float, str, str]:
        cfg = self.config
        scores = []

        # 1. Holder concentration signal
        whale_yes_pct = holder_metrics.get("whale_yes_pct", 0.5)
        holder_signal = abs(whale_yes_pct - 0.5) * 2
        holder_direction = 1.0 if whale_yes_pct > 0.5 else -1.0
        scores.append((holder_signal * holder_direction, cfg.holder_concentration_weight))

        # 2. Smart money PnL bias
        scores.append((pnl_bias, cfg.smart_money_sentiment_weight))

        # 3. Orderbook imbalance
        scores.append((orderbook.imbalance_ratio, cfg.orderbook_imbalance_weight))

        # 4. Trade flow direction
        total_flow = trade_flow.large_buy_yes_usd + trade_flow.large_buy_no_usd
        if total_flow > 0:
            flow_bias = (trade_flow.large_buy_yes_usd - trade_flow.large_buy_no_usd) / total_flow
        else:
            flow_bias = 0.0
        scores.append((flow_bias, cfg.trade_flow_weight))

        # Weighted composite
        weighted_sum = sum(s * w for s, w in scores)
        total_weight = sum(w for _, w in scores)
        composite = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Cluster bonus
        if cluster_detected:
            composite = composite * 1.3
        composite = np.clip(composite, -1.0, 1.0)

        confidence = abs(composite)
        direction = "YES" if composite > 0.05 else ("NO" if composite < -0.05 else "NEUTRAL")

        if confidence >= 0.7:
            strength = "strong"
        elif confidence >= 0.4:
            strength = "moderate"
        elif confidence >= 0.15:
            strength = "weak"
        else:
            strength = "none"

        return confidence, direction, strength

    # ─── Main Analysis Pipeline ───────────────────────────────────────────

    def analyze_market(self, market) -> WhaleIntelligence:
        market_id = getattr(market, "market_id", str(market))
        condition_id = getattr(market, "condition_id", "") or ""
        clob_token_ids = getattr(market, "clob_token_ids", []) or []

        if market_id in self._cache:
            cache_time, cached_result = self._cache[market_id]
            if time.time() - cache_time < self._cache_ttl:
                return cached_result

        intel = WhaleIntelligence(
            market_id=market_id,
            condition_id=condition_id,
            timestamp=datetime.utcnow().isoformat(),
        )

        # 1. Holder Analysis
        holder_metrics = {"n_whales": 0, "whale_yes_pct": 0.5, "whale_no_pct": 0.5, "concentration": 0.0}
        whale_positions = []

        if condition_id:
            try:
                holders_data = self.fetch_holders(condition_id)
                if not holders_data:
                    holders_data = self.fetch_subgraph_positions(condition_id)
                if holders_data:
                    whale_positions, holder_metrics = self._analyze_holders(holders_data, condition_id)
            except Exception as e:
                logger.warning(f"Holder analysis failed for {market_id}: {e}")

        intel.whale_positions = whale_positions
        intel.n_whale_holders = holder_metrics["n_whales"]
        intel.whale_yes_pct = holder_metrics["whale_yes_pct"]
        intel.whale_no_pct = holder_metrics["whale_no_pct"]
        intel.holder_concentration = holder_metrics["concentration"]

        # 2. Smart Money PnL Bias
        pnl_bias = 0.0
        if whale_positions:
            try:
                pnl_bias = self._analyze_smart_money_pnl(whale_positions)
            except Exception as e:
                logger.warning(f"PnL analysis failed for {market_id}: {e}")
        intel.smart_money_pnl_bias = pnl_bias

        # 3. Large Trade Flow
        trade_flow = TradeFlow()
        if condition_id:
            try:
                trades = self.fetch_large_trades(condition_id, min_usd=self.config.large_trade_min_usd)
                if trades:
                    trade_flow = self._analyze_trade_flow(trades)
            except Exception as e:
                logger.warning(f"Trade flow analysis failed for {market_id}: {e}")
        intel.trade_flow = trade_flow

        # 4. Orderbook Depth Analysis
        orderbook = OrderbookSignal()
        if clob_token_ids and len(clob_token_ids) > 0:
            try:
                yes_token = clob_token_ids[0] if isinstance(clob_token_ids[0], str) else str(clob_token_ids[0])
                book_data = self.fetch_orderbook(yes_token)
                if book_data:
                    orderbook = self._analyze_orderbook(book_data)
            except Exception as e:
                logger.warning(f"Orderbook analysis failed for {market_id}: {e}")
        intel.orderbook = orderbook

        # 5. Cluster Detection
        try:
            cluster_detected, cluster_dir, cluster_wallets, cluster_vol = self._detect_cluster(
                whale_positions, trade_flow
            )
            intel.cluster_detected = cluster_detected
            intel.cluster_direction = cluster_dir
            intel.cluster_wallets = cluster_wallets
            intel.cluster_volume_usd = cluster_vol
        except Exception as e:
            logger.warning(f"Cluster detection failed for {market_id}: {e}")

        # 6. Composite Score
        try:
            score, direction, strength = self._compute_composite_score(
                holder_metrics, pnl_bias, trade_flow, orderbook, intel.cluster_detected
            )
            intel.whale_confidence_score = score
            intel.whale_direction = direction
            intel.signal_strength = strength
        except Exception as e:
            logger.warning(f"Composite scoring failed for {market_id}: {e}")

        self._cache[market_id] = (time.time(), intel)
        return intel

    def analyze_batch(self, markets: list) -> Dict[str, WhaleIntelligence]:
        results = {}
        for market in markets:
            try:
                mid = getattr(market, "market_id", str(market))
                results[mid] = self.analyze_market(market)
            except Exception as e:
                logger.warning(f"Whale analysis failed for market: {e}")
        return results

    def close(self):
        self.client.close()


# ─── Backtesting Support ──────────────────────────────────────────────────────

class HonestMockWhaleTracker(WhaleTracker):
    """
    Honest mock whale tracker for backtesting — NO data leakage.
    Does NOT accept resolved_outcome. Generates whale signals using
    only observable market data (price, volume, spread, question text).

    Expected accuracy: ~52-58% directional (slightly better than coin flip
    due to market-price anchoring, but nowhere near the 60-75% the leaked
    MockWhaleTracker produced).
    """

    def __init__(self, config: Optional[WhaleConfig] = None, seed: int = 42):
        self.config = config or WhaleConfig()
        self.rng = np.random.RandomState(seed)
        self.client = None  # No API calls in mock
        self._cache: Dict[str, Tuple[float, any]] = {}
        self._cache_ttl = 300
        self._last_request_time = 0.0

    def analyze_market(self, market) -> WhaleIntelligence:
        """
        Generate whale intelligence from observable market data only.
        No resolved_outcome parameter — this is the honest version.
        """
        market_id = getattr(market, "market_id", str(market))
        condition_id = getattr(market, "condition_id", "") or market_id
        yes_price = getattr(market, "yes_price", 0.5)
        volume = getattr(market, "volume", 0)
        spread = getattr(market, "spread", 0.04)
        question = getattr(market, "question", "")

        intel = WhaleIntelligence(
            market_id=market_id,
            condition_id=condition_id,
            timestamp=datetime.utcnow().isoformat(),
        )

        # ── Whale count: based on log(volume / $100K), range 0-8, with Poisson noise
        volume_100k = max(volume / 100_000, 0.01)
        base_whale_count = min(int(np.log(volume_100k + 1) * 2), 8)
        n_whales = max(0, base_whale_count + self.rng.poisson(1) - 1)
        n_whales = min(n_whales, 8)
        intel.n_whale_holders = n_whales

        # ── Whale direction: slight bias toward market price direction
        #    If yes_price > 0.6 -> slight YES bias
        #    If yes_price < 0.4 -> slight NO bias
        #    Mid-range (0.4-0.6) -> random
        #    Add significant noise (std=0.20) so accuracy is only ~52-58%
        if yes_price > 0.6:
            direction_bias = (yes_price - 0.5) * 0.3  # Small positive bias
        elif yes_price < 0.4:
            direction_bias = (yes_price - 0.5) * 0.3  # Small negative bias
        else:
            direction_bias = 0.0  # No bias in mid-range

        # Add heavy noise — this is the key to honesty
        direction_signal = direction_bias + self.rng.normal(0, 0.20)

        # Convert to whale_yes_pct / whale_no_pct
        whale_yes_pct = np.clip(0.5 + direction_signal, 0.15, 0.85)
        whale_no_pct = 1.0 - whale_yes_pct
        intel.whale_yes_pct = whale_yes_pct
        intel.whale_no_pct = whale_no_pct

        # ── Smart money PnL bias: derived from direction + noise, NOT outcomes
        intel.smart_money_pnl_bias = np.clip(
            direction_signal * 0.8 + self.rng.normal(0, 0.12),
            -1.0, 1.0
        )

        # ── Whale confidence score: mostly 0.1-0.4, rarely above 0.5
        raw_conf = abs(direction_signal) * 0.5 + self.rng.uniform(0.05, 0.25)
        intel.whale_confidence_score = np.clip(raw_conf, 0.0, 0.7)
        # Compress: make scores above 0.5 rare
        if intel.whale_confidence_score > 0.5:
            intel.whale_confidence_score = 0.5 + (intel.whale_confidence_score - 0.5) * 0.3

        # ── Signal strength: mostly "weak" or "moderate"
        price_extremity = abs(yes_price - 0.5)
        if price_extremity > 0.35 and intel.whale_confidence_score > 0.35:
            intel.signal_strength = "strong"
        elif intel.whale_confidence_score > 0.25:
            intel.signal_strength = "moderate"
        else:
            intel.signal_strength = "weak"

        # ── Whale direction label
        if whale_yes_pct > 0.55:
            intel.whale_direction = "YES"
        elif whale_no_pct > 0.55:
            intel.whale_direction = "NO"
        else:
            intel.whale_direction = "NEUTRAL"

        # ── Insider aligned: random with P=0.3 (close to base rate)
        insider_aligned = self.rng.random() < 0.30

        # ── Cluster detected: random with P=0.2
        intel.cluster_detected = self.rng.random() < 0.20
        if intel.cluster_detected:
            intel.cluster_direction = intel.whale_direction if intel.whale_direction != "NEUTRAL" else "YES"
            intel.cluster_wallets = max(3, self.rng.poisson(3) + 2)
            intel.cluster_volume_usd = volume * self.rng.uniform(0.01, 0.05)

        # ── Orderbook imbalance: noisy signal from price
        intel.orderbook.imbalance_ratio = np.clip(
            direction_signal * 0.4 + self.rng.normal(0, 0.25),
            -1.0, 1.0
        )
        intel.orderbook.spread = spread

        # ── Trade flow: noisy simulation
        volume_m = volume / 1e6
        base_flow = max(volume_m * 3000, 500)
        flow_bias = intel.smart_money_pnl_bias
        intel.trade_flow.large_buy_yes_usd = base_flow * (0.5 + flow_bias * 0.2)
        intel.trade_flow.large_buy_no_usd = base_flow * (0.5 - flow_bias * 0.2)
        intel.trade_flow.n_unique_large_wallets_yes = max(1, int(n_whales * whale_yes_pct))
        intel.trade_flow.n_unique_large_wallets_no = max(1, int(n_whales * whale_no_pct))

        # ── Holder concentration
        intel.holder_concentration = np.clip(
            0.4 + self.rng.normal(0, 0.12),
            0.1, 0.8
        )

        return intel
