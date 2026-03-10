"""
Polymarket Quant Bot — Data Layer (Article Part VIII Layer 1)
=============================================================
Handles all data ingestion from the Polymarket Gamma API.

Gamma API (no auth required):
- Base URL: https://gamma-api.polymarket.com
- Max limit per page: 500
- Pagination via offset parameter
- Several fields are JSON-encoded strings (outcomePrices, outcomes, clobTokenIds)
"""
import json
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import httpx

from config import GAMMA_API_BASE, GAMMA_API_TIMEOUT, GAMMA_API_MAX_LIMIT

logger = logging.getLogger(__name__)


# --- Data Models ---

@dataclass
class MarketData:
    """Parsed market data from the Gamma API."""
    market_id: str
    question: str
    slug: str
    yes_price: float
    no_price: float
    best_bid: float
    best_ask: float
    spread: float
    last_trade_price: float
    volume: float
    volume_24h: float
    liquidity: float
    open_interest: float
    active: bool
    closed: bool
    end_date: Optional[str] = None
    start_date: Optional[str] = None
    category: Optional[str] = None
    condition_id: Optional[str] = None
    clob_token_ids: list = field(default_factory=lambda: ["Yes", "No"])
    outcomes: List[str] = field(default_factory=lambda: ["Yes", "No"])
    market_type: str = "normal"
    one_day_price_change: float = 0.0
    one_week_price_change: float = 0.0

    @property
    def mid_price(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.yes_price

    @property
    def is_tradeable(self) -> bool:
        """Market is tradeable if active, not closed, has liquidity, and reasonable spread."""
        return (
            self.active
            and not self.closed
            and self.liquidity > 0
            and self.spread < 0.15
            and 0.01 < self.yes_price < 0.99
        )

    @property
    def time_to_expiry_days(self) -> float:
        """Days until market closes."""
        if not self.end_date:
            return 90.0  # default
        try:
            end = datetime.fromisoformat(self.end_date.replace("+00:00", "+00:00"))
            if end.tzinfo is None:
                now = datetime.utcnow()
            else:
                now = datetime.utcnow()
            delta = (end.replace(tzinfo=None) - now).total_seconds() / 86400
            return max(delta, 0.0)
        except (ValueError, TypeError):
            return 90.0


@dataclass
class EventData:
    """Parsed event data (contains multiple markets)."""
    event_id: str
    title: str
    slug: str
    category: str
    volume: float
    liquidity: float
    active: bool
    closed: bool
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    markets: List[MarketData] = field(default_factory=list)


# --- Parsing Utilities ---

def _safe_json_loads(val, default=None):
    """Parse JSON-encoded string fields (outcomePrices, outcomes, clobTokenIds)."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return default
    return val if val is not None else default


def _safe_float(val, default=0.0):
    """Safely convert to float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val, default=0):
    """Safely convert to int."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def parse_market(raw) -> MarketData:
    """Parse a raw market dict from the API into MarketData.
    
    Idempotent: if passed a MarketData object, returns it as-is.
    """
    if isinstance(raw, MarketData):
        return raw
    prices = _safe_json_loads(raw.get("outcomePrices"), [0.5, 0.5])
    outcomes = _safe_json_loads(raw.get("outcomes"), ["Yes", "No"])
    clob_ids = _safe_json_loads(raw.get("clobTokenIds"), [])

    yes_price = _safe_float(prices[0]) if len(prices) > 0 else 0.5
    no_price = _safe_float(prices[1]) if len(prices) > 1 else 0.5

    return MarketData(
        market_id=str(raw.get("id", "")),
        question=raw.get("question", ""),
        slug=raw.get("slug", ""),
        yes_price=yes_price,
        no_price=no_price,
        best_bid=_safe_float(raw.get("bestBid")),
        best_ask=_safe_float(raw.get("bestAsk")),
        spread=_safe_float(raw.get("spread")),
        last_trade_price=_safe_float(raw.get("lastTradePrice")),
        volume=_safe_float(raw.get("volumeNum")),
        volume_24h=_safe_float(raw.get("volume24hr")),
        liquidity=_safe_float(raw.get("liquidityNum")),
        open_interest=_safe_float(raw.get("openInterest")),
        active=bool(raw.get("active", False)),
        closed=bool(raw.get("closed", False)),
        end_date=raw.get("endDate"),
        start_date=raw.get("startDate"),
        category=raw.get("category"),
        condition_id=raw.get("conditionId"),
        clob_token_ids=clob_ids,
        market_type=raw.get("marketType", "normal"),
        one_day_price_change=_safe_float(raw.get("oneDayPriceChange")),
        one_week_price_change=_safe_float(raw.get("oneWeekPriceChange")),
    )


def parse_event(raw) -> EventData:
    """Parse a raw event dict from the API into EventData.
    
    Idempotent: if passed an EventData object, returns it as-is.
    """
    if isinstance(raw, EventData):
        return raw
    markets_raw = raw.get("markets", [])
    markets = [parse_market(m) for m in markets_raw]
    return EventData(
        event_id=str(raw.get("id", "")),
        title=raw.get("title", ""),
        slug=raw.get("slug", ""),
        category=raw.get("category", ""),
        volume=_safe_float(raw.get("volume")),
        liquidity=_safe_float(raw.get("liquidity")),
        active=bool(raw.get("active", False)),
        closed=bool(raw.get("closed", False)),
        start_date=raw.get("startDate"),
        end_date=raw.get("endDate"),
        markets=markets,
    )


# --- Gamma API Client ---

class GammaAPIClient:
    """
    Client for the Polymarket Gamma API (no auth required).

    Handles pagination (limit=500 max per page), rate limiting,
    and JSON-string field parsing.
    """

    def __init__(self, base_url: str = GAMMA_API_BASE, timeout: int = GAMMA_API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self._last_request_time = 0.0
        self._min_interval = 0.2  # rate limit: max 5 req/sec

    def _rate_limit(self):
        """Simple rate limiter."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict = None) -> list:
        """Make a GET request with rate limiting."""
        self._rate_limit()
        url = f"{self.base_url}{endpoint}"
        resp = self.client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def fetch_events(self, limit: int = 500, offset: int = 0,
                     active: Optional[bool] = None, closed: Optional[bool] = None) -> List[EventData]:
        """Fetch events with pagination."""
        params = {
            "limit": min(limit, GAMMA_API_MAX_LIMIT),
            "offset": offset,
        }
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        raw = self._get("/events", params)
        return [parse_event(e) for e in raw]

    def fetch_all_events(self, max_events: int = 1500,
                         active: Optional[bool] = None,
                         closed: Optional[bool] = None) -> List[EventData]:
        """Fetch all events with automatic pagination."""
        all_events = []
        offset = 0
        while len(all_events) < max_events:
            batch = self.fetch_events(
                limit=GAMMA_API_MAX_LIMIT,
                offset=offset,
                active=active,
                closed=closed,
            )
            if not batch:
                break
            all_events.extend(batch)
            logger.info(f"Fetched {len(all_events)} events (offset={offset})")
            offset += len(batch)
            if len(batch) < GAMMA_API_MAX_LIMIT:
                break
        return all_events[:max_events]

    def get_market_snapshot(self, max_events: int = 500,
                            active: Optional[bool] = None,
                            closed: Optional[bool] = None) -> Dict:
        """
        Fetch events and compute market-level statistics.
        Returns a snapshot with all markets and aggregate stats.
        """
        events = self.fetch_all_events(max_events=max_events, active=active, closed=closed)
        all_markets = []
        categories = {}
        prices = []
        spreads = []

        for event in events:
            for m in event.markets:
                all_markets.append(m)
                cat = m.category or "Unknown"
                if cat not in categories:
                    categories[cat] = {"count": 0, "volume": 0.0, "liquidity": 0.0}
                categories[cat]["count"] += 1
                categories[cat]["volume"] += m.volume
                categories[cat]["liquidity"] += m.liquidity
                if 0.05 < m.yes_price < 0.95:
                    prices.append(m.yes_price)
                if m.spread > 0:
                    spreads.append(m.spread)

        total_vol = sum(m.volume for m in all_markets)
        total_liq = sum(m.liquidity for m in all_markets)

        price_dist = {
            "mean": float(np.mean(prices)) if prices else 0.0,
            "median": float(np.median(prices)) if prices else 0.0,
            "std": float(np.std(prices)) if prices else 0.0,
            "pct_extreme": len([p for p in prices if p < 0.05 or p > 0.95]) / max(len(prices), 1),
        }

        spread_dist = {
            "mean": float(np.mean(spreads)) if spreads else 0.0,
            "median": float(np.median(spreads)) if spreads else 0.0,
            "std": float(np.std(spreads)) if spreads else 0.0,
            "p90": float(np.percentile(spreads, 90)) if spreads else 0.0,
        }

        tradeable = [m for m in all_markets if m.is_tradeable]

        return {
            "n_events": len(events),
            "n_markets": len(all_markets),
            "n_tradeable": len(tradeable),
            "total_volume": total_vol,
            "total_liquidity": total_liq,
            "markets": all_markets,
            "tradeable": tradeable,
            "categories": categories,
            "price_distribution": price_dist,
            "spread_distribution": spread_dist,
        }

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def fetch_markets_direct(self, limit: int = 500, offset: int = 0,
                             active: Optional[bool] = None, closed: Optional[bool] = None,
                             order: str = "liquidityNum", ascending: bool = False) -> List[MarketData]:
        """
        Fetch markets directly from /markets endpoint (better for live trading).
        Supports ordering by liquidity, volume, etc.
        """
        params = {
            "limit": min(limit, GAMMA_API_MAX_LIMIT),
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        raw = self._get("/markets", params)
        return [parse_market(m) for m in raw]

    def fetch_all_markets_direct(self, max_markets: int = 1500,
                                 active: Optional[bool] = None,
                                 closed: Optional[bool] = None,
                                 order: str = "liquidityNum",
                                 ascending: bool = False) -> List[MarketData]:
        """Fetch all markets with automatic pagination via /markets endpoint."""
        all_markets = []
        offset = 0
        while len(all_markets) < max_markets:
            batch = self.fetch_markets_direct(
                limit=GAMMA_API_MAX_LIMIT,
                offset=offset,
                active=active,
                closed=closed,
                order=order,
                ascending=ascending,
            )
            if not batch:
                break
            all_markets.extend(batch)
            logger.info(f"Fetched {len(all_markets)} markets (offset={offset})")
            offset += len(batch)
            if len(batch) < GAMMA_API_MAX_LIMIT:
                break
        return all_markets[:max_markets]

    def get_live_market_snapshot(self, max_markets: int = 500) -> Dict:
        """
        Fetch LIVE tradeable markets sorted by liquidity.
        Uses /markets endpoint directly for best results.
        """
        all_markets = self.fetch_all_markets_direct(
            max_markets=max_markets,
            active=True,
            closed=False,
            order="liquidityNum",
        )

        categories = {}
        prices = []
        spreads = []

        for m in all_markets:
            cat = m.category or "Unknown"
            if cat not in categories:
                categories[cat] = {"count": 0, "volume": 0.0, "liquidity": 0.0}
            categories[cat]["count"] += 1
            categories[cat]["volume"] += m.volume
            categories[cat]["liquidity"] += m.liquidity
            if 0.05 < m.yes_price < 0.95:
                prices.append(m.yes_price)
            if m.spread > 0:
                spreads.append(m.spread)

        total_vol = sum(m.volume for m in all_markets)
        total_liq = sum(m.liquidity for m in all_markets)
        tradeable = [m for m in all_markets if m.is_tradeable]

        price_dist = {
            "mean": float(np.mean(prices)) if prices else 0.0,
            "median": float(np.median(prices)) if prices else 0.0,
            "std": float(np.std(prices)) if prices else 0.0,
            "pct_extreme": len([p for p in prices if p < 0.05 or p > 0.95]) / max(len(prices), 1),
        }

        spread_dist = {
            "mean": float(np.mean(spreads)) if spreads else 0.0,
            "median": float(np.median(spreads)) if spreads else 0.0,
            "std": float(np.std(spreads)) if spreads else 0.0,
            "p90": float(np.percentile(spreads, 90)) if spreads else 0.0,
        }

        return {
            "n_events": 0,
            "n_markets": len(all_markets),
            "n_tradeable": len(tradeable),
            "total_volume": total_vol,
            "total_liquidity": total_liq,
            "markets": all_markets,
            "tradeable": tradeable,
            "categories": categories,
            "price_distribution": price_dist,
            "spread_distribution": spread_dist,
        }
