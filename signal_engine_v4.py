"""Polymarket Signal Engine V4.1 — Arbitrage & Near-Certainty Alpha
================================================================
4 Active Alpha Layers:
  L2: Endgame Near-Certainty — 90-98% outcomes within 7 days of resolution
  L3: Multi-Outcome Mispricing — /events endpoint, grouped market sum != 1.0
  L4: Cross-Market Correlation — logically dependent markets priced inconsistently
  L5: Sentiment Velocity — orderbook imbalance + depth asymmetry
"""

import asyncio
import httpx
import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
POLYMARKET_FEE = 0.02
ENDGAME_MIN_VOLUME = 10000
ENDGAME_MIN_PROB = 0.90
ENDGAME_MAX_HOURS = 168
ENDGAME_MAX_PRICE = 0.979
ENDGAME_CALIBRATION_BONUS = 0.012
MULTI_MIN_EDGE = 0.03
MULTI_MIN_VOLUME = 10000
CORR_MIN_EDGE = 0.04
CORR_MIN_VOLUME = 25000
CORR_MIN_WORD_OVERLAP = 2
CORR_STOP_WORDS = frozenset([
    'will','the','this','that','what','when','does','have','been','before',
    'after','above','below','from','with','about','into','during','between',
    'under','over','than','more','less','each','every','some','which','there',
    'their','would','could','should','being','other','these','those','where',
    'while','because','through','only','also','just','much','many','such'
])
SENT_MIN_VOLUME = 50000
SENT_MAX_SPREAD = 0.08
SENT_MIN_IMBALANCE = 0.20
SENT_STRONG_DEPTH = 50000
MAX_POSITION_SIZE = 500
PORTFOLIO_MAX_POSITIONS = 30
DRAWDOWN_CIRCUIT_BREAKER = 0.10

class V4SignalType(Enum):
    ENDGAME = "endgame_certainty"
    MULTI_OUTCOME = "multi_outcome_arb"
    CORRELATION = "cross_market_corr"
    SENTIMENT = "sentiment_velocity"

@dataclass
class V4Signal:
    signal_type: V4SignalType
    market_id: str
    market_question: str
    action: str
    entry_price: float
    expected_payout: float
    edge: float
    confidence: float
    time_to_resolution_hours: Optional[float] = None
    annualized_return: Optional[float] = None
    risk_notes: str = ""
    related_markets: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    recommended_size: float = 0.0

@dataclass
class V4CycleResult:
    timestamp: str
    markets_scanned: int
    events_scanned: int
    signals: List[V4Signal]
    endgame_opportunities: int
    multi_outcome_opportunities: int
    correlation_opportunities: int
    sentiment_opportunities: int
    top_correlation_edge: float = 0.0
    endgame_candidates_found: int = 0
    errors: List[str] = field(default_factory=list)

class EndgameScanner:
    """Buy 90-97.9% probability outcomes within 7 days of resolution."""
    async def scan(self, markets: List[Dict]) -> Tuple[List[V4Signal], int]:
        signals = []
        candidates = 0
        now = datetime.now(timezone.utc)
        for market in markets:
            try:
                if not market.get('active') or market.get('closed'):
                    continue
                end_date = market.get('endDate')
                if not end_date:
                    continue
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                hours = (end_dt - now).total_seconds() / 3600
                if hours <= 0 or hours > ENDGAME_MAX_HOURS:
                    continue
                prices = json.loads(market.get('outcomePrices', '[]'))
                if len(prices) < 2:
                    continue
                yes_p = float(prices[0])
                no_p = float(prices[1])
                vol = float(market.get('volume', 0))
                if vol < ENDGAME_MIN_VOLUME:
                    continue
                for side, price, opp in [('YES', yes_p, no_p), ('NO', no_p, yes_p)]:
                    if price < ENDGAME_MIN_PROB or price >= ENDGAME_MAX_PRICE:
                        continue
                    candidates += 1
                    net_payout = 1.0 - POLYMARKET_FEE
                    dist = 1.0 - price
                    bonus = ENDGAME_CALIBRATION_BONUS * (dist / 0.10)
                    adj_prob = min(0.999, price + bonus)
                    adj_ev = adj_prob * net_payout
                    adj_edge = adj_ev - price
                    if adj_edge <= 0.003:
                        continue
                    time_bonus = max(0, 0.02 * (1 - hours / ENDGAME_MAX_HOURS))
                    final_conf = min(0.999, adj_prob + time_bonus)
                    days = hours / 24
                    ann = ((1 + adj_edge/price) ** (365/max(days, 0.1)) - 1) if days > 0 else None
                    signals.append(V4Signal(
                        signal_type=V4SignalType.ENDGAME,
                        market_id=market.get('conditionId', ''),
                        market_question=market.get('question', ''),
                        action=f"BUY_{side}",
                        entry_price=price, expected_payout=net_payout,
                        edge=adj_edge, confidence=final_conf,
                        time_to_resolution_hours=hours, annualized_return=ann,
                        risk_notes=f"Endgame {side}@{price:.3f}, {hours:.1f}h left, adj_prob={adj_prob:.4f}",
                        metadata={'side': side, 'price': price, 'volume': vol,
                                  'hours': hours, 'slug': market.get('slug', '')}
                    ))
            except Exception:
                continue
        signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
        return signals, candidates

class MultiOutcomeScanner:
    """Scan /events for grouped markets where YES prices sum != 1.0."""
    async def scan(self, events: List[Dict]) -> List[V4Signal]:
        signals = []
        for event in events:
            try:
                event_markets = event.get('markets', [])
                if len(event_markets) < 3:
                    continue
                total_yes = 0
                outcomes = []
                all_active = True
                min_vol = float('inf')
                for m in event_markets:
                    if not m.get('active') or m.get('closed'):
                        all_active = False
                        break
                    prices = json.loads(m.get('outcomePrices', '[]'))
                    if not prices:
                        continue
                    yp = float(prices[0])
                    vol = float(m.get('volume', 0))
                    min_vol = min(min_vol, vol)
                    total_yes += yp
                    outcomes.append({'question': m.get('question',''), 'yes_price': yp,
                                     'conditionId': m.get('conditionId',''), 'volume': vol})
                if not all_active or len(outcomes) < 3:
                    continue
                net_payout = 1.0 - POLYMARKET_FEE
                edge = net_payout - total_yes
                overpriced = total_yes - (1.0 + POLYMARKET_FEE)
                if edge >= MULTI_MIN_EDGE and min_vol >= MULTI_MIN_VOLUME:
                    outcomes.sort(key=lambda x: x['yes_price'], reverse=True)
                    signals.append(V4Signal(
                        signal_type=V4SignalType.MULTI_OUTCOME,
                        market_id=event.get('slug', event.get('id', '')),
                        market_question=f"Multi: {event.get('title', event.get('slug', '?'))} ({len(outcomes)} outcomes)",
                        action="BUY_ALL_YES", entry_price=total_yes,
                        expected_payout=net_payout, edge=edge, confidence=0.98,
                        risk_notes=f"Sum={total_yes:.4f}, {len(outcomes)} outcomes, min_vol=${min_vol:,.0f}",
                        related_markets=[o['conditionId'] for o in outcomes],
                        metadata={'event_slug': event.get('slug',''), 'num_outcomes': len(outcomes),
                                  'total_yes': total_yes, 'outcomes': outcomes[:10], 'min_vol': min_vol}
                    ))
                elif overpriced >= 0.03:
                    signals.append(V4Signal(
                        signal_type=V4SignalType.MULTI_OUTCOME,
                        market_id=event.get('slug', event.get('id', '')),
                        market_question=f"OVERPRICED: {event.get('title', '?')} ({len(outcomes)} outcomes)",
                        action="INTEL_ONLY", entry_price=total_yes,
                        expected_payout=1.0, edge=-overpriced, confidence=0.5,
                        risk_notes=f"Sum={total_yes:.4f} > 1.0 by {overpriced:.4f}. Cannot short, intel only.",
                        metadata={'event_slug': event.get('slug',''), 'total_yes': total_yes,
                                  'num_outcomes': len(outcomes), 'overpriced_by': overpriced}
                    ))
            except Exception:
                continue
        signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
        return signals

class CorrelationScanner:
    """Find logically dependent markets with inconsistent pricing.
    Enhanced: word overlap scoring, date patterns, numeric thresholds."""

    def _significant_tokens(self, question: str) -> set:
        words = re.findall(r'[a-z0-9]+', question.lower())
        return {w for w in words if len(w) > 3 and w not in CORR_STOP_WORDS}

    def _extract_numbers(self, question: str) -> List[float]:
        raw = re.findall(r'\$?([\d,]+\.?\d*)\s*[kmbt]?', question.lower())
        nums = []
        for r in raw:
            try:
                v = float(r.replace(',', ''))
                if v > 0:
                    nums.append(v)
            except ValueError:
                continue
        return nums

    def _extract_date_deadline(self, question: str) -> Optional[datetime]:
        months = {'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
                  'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
                  'jan':1,'feb':2,'mar':3,'apr':4,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        q = question.lower()
        for mname, mnum in months.items():
            pat = rf'by\s+{mname}\s*(\d{{4}})?'
            match = re.search(pat, q)
            if match:
                year = int(match.group(1)) if match.group(1) else 2026
                return datetime(year, mnum, 28, tzinfo=timezone.utc)
            pat2 = rf'{mname}\s+\d{{1,2}}(?:st|nd|rd|th)?\s*,?\s*(\d{{4}})?'
            match2 = re.search(pat2, q)
            if match2:
                year = int(match2.group(1)) if match2.group(1) else 2026
                return datetime(year, mnum, 15, tzinfo=timezone.utc)
        return None

    async def scan(self, markets: List[Dict]) -> List[V4Signal]:
        signals = []
        active = []
        for m in markets:
            if not m.get('active') or m.get('closed'):
                continue
            vol = float(m.get('volume', 0))
            if vol < CORR_MIN_VOLUME:
                continue
            prices = json.loads(m.get('outcomePrices', '[]'))
            if len(prices) < 2:
                continue
            tokens = self._significant_tokens(m.get('question', ''))
            nums = self._extract_numbers(m.get('question', ''))
            deadline = self._extract_date_deadline(m.get('question', ''))
            active.append({
                'market': m, 'yes': float(prices[0]), 'volume': vol,
                'tokens': tokens, 'nums': nums, 'deadline': deadline
            })

        checked = set()
        for i in range(len(active)):
            for j in range(i+1, len(active)):
                a, b = active[i], active[j]
                pair = tuple(sorted([a['market'].get('conditionId',''), b['market'].get('conditionId','')]))
                if pair in checked:
                    continue
                checked.add(pair)

                overlap = a['tokens'] & b['tokens']
                if len(overlap) < CORR_MIN_WORD_OVERLAP:
                    continue

                overlap_score = len(overlap) / max(len(a['tokens'] | b['tokens']), 1)
                sig = self._check_inconsistency(a, b, overlap_score)
                if sig:
                    signals.append(sig)

        signals.sort(key=lambda s: s.edge, reverse=True)
        return signals

    def _check_inconsistency(self, a: Dict, b: Dict, overlap_score: float) -> Optional[V4Signal]:
        try:
            qa = a['market'].get('question', '')
            qb = b['market'].get('question', '')
            ya, yb = a['yes'], b['yes']

            # Pattern 1: Numeric thresholds (e.g. "BTC > $100K" vs "BTC > $80K")
            if a['nums'] and b['nums']:
                va, vb = a['nums'][0], b['nums'][0]
                if va > vb and ya > yb + CORR_MIN_EDGE:
                    edge = ya - yb
                    conf = min(0.85, 0.60 + overlap_score * 0.3)
                    return V4Signal(
                        signal_type=V4SignalType.CORRELATION,
                        market_id=b['market'].get('conditionId', ''),
                        market_question=f"Corr: '{qa[:55]}' vs '{qb[:55]}'",
                        action="BUY_YES", entry_price=yb,
                        expected_payout=1.0 - POLYMARKET_FEE, edge=edge,
                        confidence=conf,
                        risk_notes=f"Threshold {va} @ {ya:.3f} > threshold {vb} @ {yb:.3f}. Overlap={overlap_score:.2f}",
                        related_markets=[a['market'].get('conditionId',''), b['market'].get('conditionId','')],
                        metadata={'qa': qa, 'qb': qb, 'price_a': ya, 'price_b': yb,
                                  'threshold_a': va, 'threshold_b': vb, 'overlap': overlap_score}
                    )
                if vb > va and yb > ya + CORR_MIN_EDGE:
                    edge = yb - ya
                    conf = min(0.85, 0.60 + overlap_score * 0.3)
                    return V4Signal(
                        signal_type=V4SignalType.CORRELATION,
                        market_id=a['market'].get('conditionId', ''),
                        market_question=f"Corr: '{qb[:55]}' vs '{qa[:55]}'",
                        action="BUY_YES", entry_price=ya,
                        expected_payout=1.0 - POLYMARKET_FEE, edge=edge,
                        confidence=conf,
                        risk_notes=f"Threshold {vb} @ {yb:.3f} > threshold {va} @ {ya:.3f}. Overlap={overlap_score:.2f}",
                        related_markets=[a['market'].get('conditionId',''), b['market'].get('conditionId','')],
                        metadata={'qa': qa, 'qb': qb, 'price_a': ya, 'price_b': yb,
                                  'threshold_a': va, 'threshold_b': vb, 'overlap': overlap_score}
                    )

            # Pattern 2: Date deadlines ("X by March" vs "X by June")
            if a['deadline'] and b['deadline'] and a['deadline'] != b['deadline']:
                if a['deadline'] < b['deadline'] and ya > yb + CORR_MIN_EDGE:
                    edge = ya - yb
                    conf = min(0.80, 0.55 + overlap_score * 0.3)
                    return V4Signal(
                        signal_type=V4SignalType.CORRELATION,
                        market_id=b['market'].get('conditionId', ''),
                        market_question=f"DateCorr: '{qa[:50]}' vs '{qb[:50]}'",
                        action="BUY_YES", entry_price=yb,
                        expected_payout=1.0 - POLYMARKET_FEE, edge=edge,
                        confidence=conf,
                        risk_notes=f"Shorter deadline @ {ya:.3f} > longer deadline @ {yb:.3f}. Overlap={overlap_score:.2f}",
                        related_markets=[a['market'].get('conditionId',''), b['market'].get('conditionId','')],
                        metadata={'qa': qa, 'qb': qb, 'deadline_a': str(a['deadline']),
                                  'deadline_b': str(b['deadline']), 'overlap': overlap_score}
                    )
                if b['deadline'] < a['deadline'] and yb > ya + CORR_MIN_EDGE:
                    edge = yb - ya
                    conf = min(0.80, 0.55 + overlap_score * 0.3)
                    return V4Signal(
                        signal_type=V4SignalType.CORRELATION,
                        market_id=a['market'].get('conditionId', ''),
                        market_question=f"DateCorr: '{qb[:50]}' vs '{qa[:50]}'",
                        action="BUY_YES", entry_price=ya,
                        expected_payout=1.0 - POLYMARKET_FEE, edge=edge,
                        confidence=conf,
                        risk_notes=f"Shorter deadline @ {yb:.3f} > longer deadline @ {ya:.3f}. Overlap={overlap_score:.2f}",
                        related_markets=[a['market'].get('conditionId',''), b['market'].get('conditionId','')],
                        metadata={'qa': qa, 'qb': qb, 'overlap': overlap_score}
                    )

            return None
        except Exception:
            return None

class SentimentVelocityScanner:
    """Detect orderbook imbalance + depth asymmetry as proxy for sentiment shifts."""
    async def scan(self, markets: List[Dict], client: httpx.AsyncClient) -> List[V4Signal]:
        signals = []
        for market in markets:
            try:
                if not market.get('active') or market.get('closed'):
                    continue
                vol = float(market.get('volume', 0))
                if vol < SENT_MIN_VOLUME:
                    continue
                prices = json.loads(market.get('outcomePrices', '[]'))
                if len(prices) < 2:
                    continue
                yes_p = float(prices[0])
                if yes_p < 0.15 or yes_p > 0.85:
                    continue
                clob_ids = json.loads(market.get('clobTokenIds', '[]'))
                if not clob_ids:
                    continue
                try:
                    resp = await client.get(f"{CLOB_API}/book", params={"token_id": clob_ids[0]}, timeout=5)
                    if resp.status_code != 200:
                        continue
                    book = resp.json()
                except Exception:
                    continue
                bids = book.get('bids', [])
                asks = book.get('asks', [])
                if not bids or not asks:
                    continue
                best_bid = float(bids[0].get('price', 0))
                best_ask = float(asks[0].get('price', 1))
                spread = best_ask - best_bid
                if spread > SENT_MAX_SPREAD:
                    continue
                bid_depth = sum(float(b.get('size', 0)) for b in bids[:5])
                ask_depth = sum(float(a.get('size', 0)) for a in asks[:5])
                total_depth = bid_depth + ask_depth
                if total_depth == 0:
                    continue
                imbalance = (bid_depth - ask_depth) / total_depth
                strong_depth = max(bid_depth, ask_depth) >= SENT_STRONG_DEPTH
                if abs(imbalance) < SENT_MIN_IMBALANCE and not strong_depth:
                    continue
                direction = "BUY_YES" if imbalance > 0 else "BUY_NO"
                entry = best_ask if imbalance > 0 else (1 - best_bid)
                edge = abs(imbalance) * 0.10
                conf = 0.55 + abs(imbalance) * 0.15
                if strong_depth:
                    conf = min(0.80, conf + 0.10)
                    edge *= 1.5
                signals.append(V4Signal(
                    signal_type=V4SignalType.SENTIMENT,
                    market_id=market.get('conditionId', ''),
                    market_question=market.get('question', ''),
                    action=direction, entry_price=entry,
                    expected_payout=entry + edge, edge=edge, confidence=conf,
                    risk_notes=f"OBI={imbalance:.3f}, spread={spread:.4f}, bid${bid_depth:,.0f}/ask${ask_depth:,.0f}",
                    metadata={'imbalance': imbalance, 'spread': spread,
                              'bid_depth': bid_depth, 'ask_depth': ask_depth,
                              'best_bid': best_bid, 'best_ask': best_ask,
                              'volume': vol, 'slug': market.get('slug',''),
                              'strong_depth': strong_depth}
                ))
            except Exception:
                continue
        signals.sort(key=lambda s: abs(s.metadata.get('imbalance', 0)), reverse=True)
        return signals

class V4RiskManager:
    SIZE_MULT = {
        V4SignalType.ENDGAME: 1.5,
        V4SignalType.MULTI_OUTCOME: 1.5,
        V4SignalType.CORRELATION: 0.8,
        V4SignalType.SENTIMENT: 0.5,
    }
    def size_position(self, signal: V4Signal, equity: float, open_count: int) -> float:
        if open_count >= PORTFOLIO_MAX_POSITIONS:
            return 0
        base = min(MAX_POSITION_SIZE, equity * 0.05)
        mult = self.SIZE_MULT.get(signal.signal_type, 1.0)
        size = base * mult * signal.confidence * min(2.0, signal.edge / 0.02)
        return min(size, MAX_POSITION_SIZE * 2)
    def check_drawdown(self, equity: float, peak: float) -> bool:
        if peak <= 0:
            return False
        return (peak - equity) / peak >= DRAWDOWN_CIRCUIT_BREAKER

class SignalGeneratorV4:
    def __init__(self):
        self.endgame = EndgameScanner()
        self.multi = MultiOutcomeScanner()
        self.correlation = CorrelationScanner()
        self.sentiment = SentimentVelocityScanner()
        self.risk = V4RiskManager()

    async def fetch_markets(self, limit: int = 1000) -> List[Dict]:
        markets = []
        async with httpx.AsyncClient(timeout=30) as client:
            offset = 0
            while offset < limit:
                try:
                    resp = await client.get(f"{GAMMA_API}/markets", params={
                        'active': 'true', 'closed': 'false',
                        'limit': min(100, limit - offset), 'offset': offset,
                        'order': 'volume', 'ascending': 'false'
                    })
                    if resp.status_code != 200:
                        break
                    batch = resp.json()
                    if not batch:
                        break
                    markets.extend(batch)
                    offset += len(batch)
                    if len(batch) < 100:
                        break
                except Exception as e:
                    print(f"Market fetch error at {offset}: {e}")
                    break
        return markets

    async def fetch_events(self, limit: int = 200) -> List[Dict]:
        events = []
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(f"{GAMMA_API}/events", params={
                    'active': 'true', 'closed': 'false', 'limit': limit
                })
                if resp.status_code == 200:
                    events = resp.json()
            except Exception as e:
                print(f"Events fetch error: {e}")
        return events

    async def generate_signals(self, markets: List[Dict] = None,
                               max_markets: int = 1000) -> V4CycleResult:
        ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        errors = []

        if markets is None:
            markets = await self.fetch_markets(max_markets)
        for m in markets:
            if m.get('category') is None:
                m['category'] = 'unknown'

        events = await self.fetch_events()
        print(f"[V4.1] Scanning {len(markets)} markets + {len(events)} events...")

        all_signals = []

        try:
            eg_sigs, eg_cands = await self.endgame.scan(markets)
            all_signals.extend(eg_sigs)
            print(f"  L2 Endgame: {len(eg_sigs)} signals ({eg_cands} candidates)")
        except Exception as e:
            errors.append(f"Endgame: {e}")
            eg_sigs, eg_cands = [], 0

        try:
            mo_sigs = await self.multi.scan(events)
            all_signals.extend(mo_sigs)
            print(f"  L3 Multi-Outcome: {len(mo_sigs)} signals from {len(events)} events")
        except Exception as e:
            errors.append(f"Multi: {e}")
            mo_sigs = []

        try:
            co_sigs = await self.correlation.scan(markets)
            all_signals.extend(co_sigs)
            print(f"  L4 Correlation: {len(co_sigs)} signals")
        except Exception as e:
            errors.append(f"Correlation: {e}")
            co_sigs = []

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                top100 = sorted(markets, key=lambda m: float(m.get('volume',0)), reverse=True)[:100]
                se_sigs = await self.sentiment.scan(top100, client)
                all_signals.extend(se_sigs)
                print(f"  L5 Sentiment: {len(se_sigs)} signals")
        except Exception as e:
            errors.append(f"Sentiment: {e}")
            se_sigs = []

        all_signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)

        # Assign recommended sizes
        for i, sig in enumerate(all_signals):
            sig.recommended_size = self.risk.size_position(sig, 10000, i)

        top_corr = max((s.edge for s in co_sigs), default=0.0)

        return V4CycleResult(
            timestamp=ts, markets_scanned=len(markets), events_scanned=len(events),
            signals=all_signals, endgame_opportunities=len(eg_sigs),
            multi_outcome_opportunities=len(mo_sigs), correlation_opportunities=len(co_sigs),
            sentiment_opportunities=len(se_sigs), top_correlation_edge=top_corr,
            endgame_candidates_found=eg_cands, errors=errors
        )

    def format_telegram_summary(self, result: V4CycleResult,
                                 equity: float = 10000, pnl: float = 0,
                                 open_pos: int = 0) -> str:
        lines = [
            f"POLYMARKET V4.1 SCANNER | {result.timestamp}",
            f"Scanned: {result.markets_scanned} markets + {result.events_scanned} events",
            "",
            f"SIGNALS: {len(result.signals)} total",
            f"  Endgame: {result.endgame_opportunities} (of {result.endgame_candidates_found} candidates)",
            f"  Multi-Outcome: {result.multi_outcome_opportunities}",
            f"  Correlation: {result.correlation_opportunities} (top edge: {result.top_correlation_edge:.3f})",
            f"  Sentiment: {result.sentiment_opportunities}",
            "",
        ]
        for st in V4SignalType:
            typed = [s for s in result.signals if s.signal_type == st]
            if typed:
                lines.append(f"--- {st.value.upper()} ---")
                for s in typed[:3]:
                    ann = f" ({s.annualized_return:.0%} ann)" if s.annualized_return else ""
                    sz = f" sz=${s.recommended_size:.0f}" if s.recommended_size > 0 else ""
                    lines.append(f"  {s.action} edge={s.edge:.3f} conf={s.confidence:.2f}{ann}{sz}")
                    lines.append(f"    {s.market_question[:70]}")
                lines.append("")
        if result.errors:
            lines.append(f"Errors: {len(result.errors)}")
            for e in result.errors[:3]:
                lines.append(f"  {e[:80]}")
        lines.append(f"\nEquity: ${equity:,.2f} | P&L: ${pnl:+,.2f} | Open: {open_pos}")
        lines.append("[PAPER MODE v4.1 - Endgame + Multi + Corr + Sentiment]")
        return "\n".join(lines)

class PaperTraderV4:
    def __init__(self, initial_equity: float = 10000):
        self.engine = SignalGeneratorV4()
        self.risk = V4RiskManager()
        self.journal = {
            'version': 'v4.1', 'created': datetime.now(timezone.utc).isoformat(),
            'portfolio': {'cash': initial_equity, 'initial_equity': initial_equity,
                          'peak_equity': initial_equity},
            'trades': [], 'open_positions': [], 'closed_positions': [],
            'equity_history': [], 'cycle_count': 0
        }

    def load_journal(self, data: Dict):
        self.journal = data
        self.journal.setdefault('version', 'v4.1')
        self.journal.setdefault('open_positions', [])
        self.journal.setdefault('closed_positions', [])

    async def run_cycle(self, max_markets: int = 1000) -> Dict:
        self.journal['cycle_count'] = self.journal.get('cycle_count', 0) + 1
        cn = self.journal['cycle_count']
        result = await self.engine.generate_signals(max_markets=max_markets)
        port = self.journal['portfolio']
        eq = port['cash']
        peak = port.get('peak_equity', eq)
        if self.risk.check_drawdown(eq, peak):
            print(f"[V4.1] CIRCUIT BREAKER at {(peak-eq)/peak:.1%} drawdown")
            return self._output(result, [], cn)
        new_trades = []
        open_n = len(self.journal.get('open_positions', []))
        for sig in result.signals[:10]:
            sz = self.risk.size_position(sig, eq, open_n + len(new_trades))
            if sz < 10:
                continue
            if sz > port['cash']:
                sz = port['cash'] * 0.9
                if sz < 10:
                    break
            trade = {
                'cycle': cn, 'timestamp': datetime.now(timezone.utc).isoformat(),
                'signal_type': sig.signal_type.value, 'market_id': sig.market_id,
                'question': sig.market_question, 'action': sig.action,
                'entry_price': sig.entry_price, 'size': round(sz, 2),
                'shares': round(sz / sig.entry_price, 4) if sig.entry_price > 0 else 0,
                'edge': sig.edge, 'confidence': sig.confidence,
                'annualized_return': sig.annualized_return,
                'risk_notes': sig.risk_notes, 'status': 'open'
            }
            new_trades.append(trade)
            port['cash'] -= sz
            self.journal['trades'].append(trade)
            self.journal['open_positions'].append(trade)
        port['peak_equity'] = max(peak, eq)
        self.journal['equity_history'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'equity': eq, 'cash': port['cash'],
            'open_positions': len(self.journal.get('open_positions', []))
        })
        return self._output(result, new_trades, cn)

    def _output(self, result: V4CycleResult, trades: List[Dict], cn: int) -> Dict:
        p = self.journal['portfolio']
        return {
            'cycle': cn, 'timestamp': result.timestamp,
            'markets_scanned': result.markets_scanned,
            'events_scanned': result.events_scanned,
            'signals_found': len(result.signals), 'new_trades': len(trades),
            'trades_detail': trades,
            'breakdown': {
                'endgame': result.endgame_opportunities,
                'multi_outcome': result.multi_outcome_opportunities,
                'correlation': result.correlation_opportunities,
                'sentiment': result.sentiment_opportunities,
            },
            'portfolio': {'cash': p['cash'], 'equity': p['cash'],
                          'open': len(self.journal.get('open_positions', [])),
                          'total_trades': len(self.journal.get('trades', []))},
            'errors': result.errors
        }

    def telegram_summary(self, out: Dict) -> str:
        return self.engine.format_telegram_summary(
            V4CycleResult(
                timestamp=out['timestamp'], markets_scanned=out['markets_scanned'],
                events_scanned=out.get('events_scanned', 0), signals=[],
                endgame_opportunities=out['breakdown']['endgame'],
                multi_outcome_opportunities=out['breakdown']['multi_outcome'],
                correlation_opportunities=out['breakdown']['correlation'],
                sentiment_opportunities=out['breakdown']['sentiment'],
            ),
            equity=out['portfolio']['equity'], pnl=0,
            open_pos=out['portfolio']['open']
        )

if __name__ == "__main__":
    async def test():
        engine = SignalGeneratorV4()
        result = await engine.generate_signals(max_markets=500)
        print(f"\nTotal signals: {len(result.signals)}")
        for s in result.signals[:15]:
            ann = f" ({s.annualized_return:.0%} ann)" if s.annualized_return else ""
            print(f"  [{s.signal_type.value}] {s.action} edge={s.edge:.4f} conf={s.confidence:.3f}{ann} sz=${s.recommended_size:.0f}")
            print(f"    {s.market_question[:80]}")
        if result.errors:
            print(f"\nErrors: {result.errors}")
    asyncio.run(test())
