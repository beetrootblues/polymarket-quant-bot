"""
Polymarket Quant Bot - Paper Trader v2
========================================
Wraps the v2 whale-intelligence pipeline with paper trade logging.
No real money - tracks hypothetical entries, exits, P&L, and win rate.
Persists a rolling trade journal as JSON for GitHub commit.

v2 changes:
  - Uses SignalGenerator v2 with whale intelligence as primary alpha
  - Removed Monte Carlo + Variance Reduction layers
  - Whale confidence, direction, cluster data logged per trade
  - Telegram summary includes whale intelligence stats
  - Stricter pre-trade filters (5% edge, $500K volume, whale confirmation)

Usage:
    python paper_trader.py              # Run one cycle
    python paper_trader.py --journal    # Print journal summary
"""
import json
import sys
import logging
import hashlib
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from config import BotConfig
from data_layer import GammaAPIClient
from signal_engine import SignalGenerator, make_v2_config
from whale_intelligence import WhaleTracker, WhaleConfig, WhaleIntelligence
from copula_engine import compute_tail_dependence
from agent_based_model import PredictionMarketABM

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PaperTrade:
    """A single hypothetical trade entry."""
    trade_id: str
    timestamp: str
    market_id: str
    question: str
    direction: str          # BUY_YES / BUY_NO
    entry_price: float
    model_probability: float
    edge: float
    edge_zscore: float
    confidence: str         # high / medium / low
    size_usd: float
    kelly_fraction: float
    status: str = "open"    # open / closed / expired
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_hours: float = 0.0
    exit_reason: Optional[str] = None
    layers_used: List[str] = field(default_factory=list)
    # v2 whale intelligence fields
    whale_confidence_score: float = 0.0
    whale_direction: str = ""
    whale_strength: str = ""
    cluster_detected: bool = False
    cluster_direction: str = ""
    cluster_wallets: int = 0
    insider_aligned: bool = False
    n_layers_agreeing: int = 0


@dataclass
class JournalSnapshot:
    """Cumulative paper trading journal."""
    created_at: str = ""
    updated_at: str = ""
    strategy_version: str = "v2"
    total_cycles: int = 0
    paper_capital: float = 10_000.0
    paper_deployed: float = 0.0
    cumulative_pnl: float = 0.0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    total_trades: int = 0
    open_trades: int = 0
    closed_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_capital: float = 10_000.0
    sharpe_estimate: float = 0.0
    # v2 whale intelligence aggregate stats
    whale_signals_received: int = 0
    whale_clusters_detected: int = 0
    whale_aligned_trades: int = 0
    avg_whale_confidence: float = 0.0
    trades: List[Dict] = field(default_factory=list)
    cycle_history: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------

class PaperTrader:
    """Runs the v2 quant pipeline and logs paper trades."""

    JOURNAL_FILE = "paper_journal.json"
    MAX_OPEN_TRADES = 20
    TRADE_EXPIRY_HOURS = 72  # Auto-close after 72h

    def __init__(self, config: BotConfig = None, journal_path: str = None):
        self.config = config or make_v2_config()
        self.journal_path = journal_path or self.JOURNAL_FILE
        self.journal = self._load_journal()
        self.now = datetime.now(timezone.utc)
        # v2: create whale tracker instance (shared across cycle)
        self.whale_tracker = WhaleTracker(WhaleConfig())

    # --- Journal persistence ---

    def _load_journal(self) -> JournalSnapshot:
        """Load existing journal or create fresh one."""
        try:
            with open(self.journal_path, 'r') as f:
                data = json.load(f)
            j = JournalSnapshot()
            for k, v in data.items():
                if hasattr(j, k):
                    setattr(j, k, v)
            logger.info(f"Loaded journal: {j.total_trades} trades, {j.total_cycles} cycles")
            return j
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No existing journal - starting fresh")
            j = JournalSnapshot()
            j.created_at = datetime.now(timezone.utc).isoformat()
            j.paper_capital = self.config.starting_capital
            j.peak_capital = self.config.starting_capital
            return j

    def _save_journal(self):
        """Persist journal to disk."""
        self.journal.updated_at = self.now.isoformat()
        with open(self.journal_path, 'w') as f:
            json.dump(asdict(self.journal), f, indent=2, default=str)
        logger.info(f"Journal saved: {self.journal_path}")

    # --- Trade lifecycle ---

    def _make_trade_id(self, market_id: str) -> str:
        """Deterministic short hash for trade ID."""
        raw = f"{market_id}:{self.now.isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def _check_duplicate(self, market_id: str) -> bool:
        """Prevent doubling up on the same market."""
        return any(
            t['market_id'] == market_id and t['status'] == 'open'
            for t in self.journal.trades
        )

    def _close_expired_trades(self, current_prices: Dict[str, float]):
        """Auto-close trades older than TRADE_EXPIRY_HOURS or if market resolved."""
        for trade in self.journal.trades:
            if trade['status'] != 'open':
                continue

            entry_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            hours_held = (self.now - entry_time).total_seconds() / 3600
            trade['hold_hours'] = round(hours_held, 2)

            # Check expiry
            if hours_held >= self.TRADE_EXPIRY_HOURS:
                exit_price = current_prices.get(trade['market_id'], trade['entry_price'])
                self._close_trade(trade, exit_price, "expired_72h")
                continue

            # Mark-to-market unrealised P&L
            if trade['market_id'] in current_prices:
                current = current_prices[trade['market_id']]
                if trade['direction'] == 'BUY_YES':
                    trade['pnl'] = round((current - trade['entry_price']) * trade['size_usd'] / trade['entry_price'], 2)
                else:
                    trade['pnl'] = round((trade['entry_price'] - current) * trade['size_usd'] / trade['entry_price'], 2)
                trade['pnl_pct'] = round(trade['pnl'] / trade['size_usd'] * 100, 2) if trade['size_usd'] > 0 else 0.0

    def _close_trade(self, trade: Dict, exit_price: float, reason: str):
        """Close a paper trade and book P&L."""
        trade['status'] = 'closed'
        trade['exit_price'] = exit_price
        trade['exit_timestamp'] = self.now.isoformat()
        trade['exit_reason'] = reason

        if trade['direction'] == 'BUY_YES':
            trade['pnl'] = round((exit_price - trade['entry_price']) * trade['size_usd'] / trade['entry_price'], 2)
        else:
            trade['pnl'] = round((trade['entry_price'] - exit_price) * trade['size_usd'] / trade['entry_price'], 2)

        trade['pnl_pct'] = round(trade['pnl'] / trade['size_usd'] * 100, 2) if trade['size_usd'] > 0 else 0.0

        # Update journal stats
        self.journal.realised_pnl = round(self.journal.realised_pnl + trade['pnl'], 2)
        self.journal.closed_trades += 1
        self.journal.open_trades = max(0, self.journal.open_trades - 1)

        if trade['pnl'] > 0:
            self.journal.wins += 1
        else:
            self.journal.losses += 1

        logger.info(f"CLOSED {trade['trade_id']}: {trade['direction']} {trade['question'][:40]} "
                    f"PnL=${trade['pnl']:+.2f} ({reason})")

    def _check_exit_signals(self, signal_gen: SignalGenerator, current_prices: Dict[str, float]):
        """Close trades where edge has flipped or price target hit."""
        for trade in self.journal.trades:
            if trade['status'] != 'open':
                continue
            if trade['market_id'] not in current_prices:
                continue

            current = current_prices[trade['market_id']]
            entry = trade['entry_price']

            # Take profit: 15%+ return
            if trade['direction'] == 'BUY_YES' and current >= entry * 1.15:
                self._close_trade(trade, current, "take_profit_15pct")
            elif trade['direction'] == 'BUY_NO' and current <= entry * 0.85:
                self._close_trade(trade, current, "take_profit_15pct")
            # Stop loss: -10% return
            elif trade['direction'] == 'BUY_YES' and current <= entry * 0.90:
                self._close_trade(trade, current, "stop_loss_10pct")
            elif trade['direction'] == 'BUY_NO' and current >= entry * 1.10:
                self._close_trade(trade, current, "stop_loss_10pct")

    def _update_journal_stats(self):
        """Recalculate aggregate stats from trade list."""
        j = self.journal
        open_trades = [t for t in j.trades if t['status'] == 'open']
        closed_trades = [t for t in j.trades if t['status'] == 'closed']

        j.open_trades = len(open_trades)
        j.closed_trades = len(closed_trades)
        j.total_trades = len(j.trades)

        j.paper_deployed = round(sum(t['size_usd'] for t in open_trades), 2)
        j.unrealised_pnl = round(sum(t['pnl'] for t in open_trades), 2)
        j.cumulative_pnl = round(j.realised_pnl + j.unrealised_pnl, 2)

        # Current equity
        equity = j.paper_capital + j.cumulative_pnl
        if equity > j.peak_capital:
            j.peak_capital = equity
        if j.peak_capital > 0:
            drawdown = (j.peak_capital - equity) / j.peak_capital
            j.max_drawdown_pct = round(max(j.max_drawdown_pct, drawdown), 4)

        # Win rate
        if j.closed_trades > 0:
            j.win_rate = round(j.wins / j.closed_trades, 4)
            winners = [t['pnl'] for t in closed_trades if t['pnl'] > 0]
            losers = [t['pnl'] for t in closed_trades if t['pnl'] <= 0]
            j.avg_win = round(np.mean(winners), 2) if winners else 0.0
            j.avg_loss = round(np.mean(losers), 2) if losers else 0.0

        if closed_trades:
            pnls = [t['pnl'] for t in closed_trades]
            j.best_trade_pnl = round(max(pnls), 2)
            j.worst_trade_pnl = round(min(pnls), 2)

            # Simple Sharpe estimate (per-trade)
            if len(pnls) > 1:
                avg = np.mean(pnls)
                std = np.std(pnls, ddof=1)
                j.sharpe_estimate = round(avg / std if std > 0 else 0.0, 3)

        # v2: Whale intelligence aggregate stats
        all_trades = j.trades
        whale_scores = [t.get('whale_confidence_score', 0) for t in all_trades if t.get('whale_confidence_score', 0) > 0]
        j.whale_signals_received = len(whale_scores)
        j.whale_clusters_detected = sum(1 for t in all_trades if t.get('cluster_detected', False))
        j.whale_aligned_trades = sum(1 for t in all_trades if t.get('insider_aligned', False))
        j.avg_whale_confidence = round(np.mean(whale_scores), 4) if whale_scores else 0.0

    # --- Main pipeline ---

    def run_cycle(self, max_markets: int = 50) -> Dict[str, Any]:
        """Execute one full paper trading cycle with v2 whale intelligence."""
        self.now = datetime.now(timezone.utc)
        cycle_start = self.now.isoformat()

        print('======================================================================')
        print('  POLYMARKET PAPER TRADER v2 - CYCLE START')
        print(f'  Timestamp: {cycle_start}')
        print(f'  Cycle #{self.journal.total_cycles + 1}')
        print(f'  Paper Capital: ${self.journal.paper_capital + self.journal.cumulative_pnl:,.2f}')
        print(f'  Strategy: Whale Intelligence + Strict Filters')
        print('======================================================================')

        # ----- LAYER 1: Data Ingestion -----
        print('\n[LAYER 1] Data Ingestion...')
        client = GammaAPIClient()
        try:
            snapshot = client.get_market_snapshot()
        except Exception as e:
            logger.error(f'Market data fetch failed: {e}')
            client.close()
            return {'error': str(e)}

        all_markets = snapshot['markets']
        tradeable = [m for m in all_markets if m.is_tradeable]
        tradeable.sort(key=lambda m: m.liquidity, reverse=True)
        target_markets = tradeable[:max_markets]

        print(f'  Tradeable: {len(tradeable)}, Targeting: {len(target_markets)}')
        print(f'  Total volume: ${snapshot["total_volume"]:,.0f}')

        # Build current price map for mark-to-market
        current_prices = {m.market_id: m.yes_price for m in tradeable}

        # ----- MANAGE EXISTING POSITIONS -----
        print('\n[POSITIONS] Managing open trades...')
        n_open_before = self.journal.open_trades

        # v2: Create SignalGenerator with shared whale tracker
        signal_gen = SignalGenerator(self.config, whale_tracker=self.whale_tracker)
        self._check_exit_signals(signal_gen, current_prices)
        self._close_expired_trades(current_prices)

        n_closed_this_cycle = n_open_before - len([t for t in self.journal.trades if t['status'] == 'open'])
        print(f'  Closed this cycle: {n_closed_this_cycle}')

        # ----- LAYER 2: Whale Intelligence -----
        print(f'\n[LAYER 2] Whale Intelligence - Scanning {len(target_markets)} markets...')

        whale_intels: Dict[str, WhaleIntelligence] = {}
        whale_scan_stats = {'total': 0, 'strong': 0, 'moderate': 0, 'weak': 0, 'none': 0, 'clusters': 0, 'errors': 0}

        for market in target_markets:
            try:
                intel = self.whale_tracker.analyze_market(market)
                whale_intels[market.market_id] = intel
                whale_scan_stats['total'] += 1
                strength = intel.signal_strength or 'none'
                whale_scan_stats[strength] = whale_scan_stats.get(strength, 0) + 1
                if intel.cluster_detected:
                    whale_scan_stats['clusters'] += 1
            except Exception as e:
                whale_scan_stats['errors'] += 1
                logger.warning(f"Whale scan failed for {market.market_id}: {e}")

        print(f'  Scanned: {whale_scan_stats["total"]} markets')
        print(f'  Signals: {whale_scan_stats["strong"]} strong, {whale_scan_stats["moderate"]} moderate, {whale_scan_stats["weak"]} weak')
        print(f'  Clusters: {whale_scan_stats["clusters"]} | Errors: {whale_scan_stats["errors"]}')

        # ----- LAYER 3: Signal Generation (v2 ensemble) -----
        print(f'\n[LAYER 3] Signal Engine v2 - Processing {len(target_markets)} markets...')

        # Update probability engine with whale intelligence
        for market in target_markets:
            whale_intel = whale_intels.get(market.market_id)
            signal_gen.prob_engine.update(market, whale_intel=whale_intel)

        # Generate signals with whale data passed through
        signals = signal_gen.process_market_batch(
            target_markets,
            whale_intels=whale_intels,
        )
        print(f'  Signals: {len(signal_gen._all_signals)} total, {len(signals)} actionable (passed all gates)')
        print(f'  Rejected: {len(signal_gen._rejected_signals)}')

        # ----- LAYER 4: Paper Trade Entry -----
        print('\n[LAYER 4] Paper Trade Entry...')
        new_trades = []
        current_open = len([t for t in self.journal.trades if t['status'] == 'open'])
        slots_available = self.MAX_OPEN_TRADES - current_open

        for sig in signals[:slots_available]:
            # Skip duplicates
            if self._check_duplicate(sig.market_id):
                continue

            # Risk check: position sizing
            equity = self.journal.paper_capital + self.journal.cumulative_pnl
            max_size = min(
                sig.suggested_size,
                equity * self.config.risk.max_position_pct,
                self.config.risk.max_position_size
            )

            if max_size < 5.0:  # Minimum $5 paper trade
                continue

            # Check portfolio exposure limit
            total_deployed = sum(t['size_usd'] for t in self.journal.trades if t['status'] == 'open')
            if (total_deployed + max_size) > equity * self.config.risk.max_portfolio_exposure:
                continue

            # Execute simulated fill with slippage
            market = next((m for m in target_markets if m.market_id == sig.market_id), None)
            if not market:
                continue

            result = signal_gen.execute_signal(sig, market)
            if not result['executed']:
                continue

            trade = PaperTrade(
                trade_id=self._make_trade_id(sig.market_id),
                timestamp=self.now.isoformat(),
                market_id=sig.market_id,
                question=sig.question,
                direction=sig.direction,
                entry_price=result['fill_price'],
                model_probability=sig.model_probability,
                edge=sig.edge,
                edge_zscore=sig.edge_zscore,
                confidence=sig.confidence,
                size_usd=round(result['size'], 2),
                kelly_fraction=round(sig.suggested_size / equity if equity > 0 else 0, 4),
                layers_used=list(sig.components.keys()) if sig.components else ['ensemble'],
                # v2 whale intelligence fields
                whale_confidence_score=round(sig.whale_confidence, 4),
                whale_direction=sig.whale_direction,
                whale_strength=sig.whale_strength,
                cluster_detected=sig.whale_cluster,
                cluster_direction=whale_intels.get(sig.market_id, WhaleIntelligence(
                    market_id=sig.market_id, condition_id="", timestamp=""
                )).cluster_direction,
                cluster_wallets=whale_intels.get(sig.market_id, WhaleIntelligence(
                    market_id=sig.market_id, condition_id="", timestamp=""
                )).cluster_wallets,
                insider_aligned=sig.insider_aligned,
                n_layers_agreeing=sig.n_layers_agreeing,
            )

            self.journal.trades.append(asdict(trade))
            new_trades.append(trade)
            logger.info(
                f"PAPER TRADE: {trade.direction} ${trade.size_usd:.0f} @ {trade.entry_price:.3f} "
                f"| edge={trade.edge:+.4f} | whale={trade.whale_direction}({trade.whale_confidence_score:.2f}) "
                f"| cluster={'YES' if trade.cluster_detected else 'NO'} "
                f"| {trade.question[:50]}"
            )

        print(f'  New paper trades: {len(new_trades)}')

        # ----- LAYER 5: Model Health Checks (lightweight) -----
        print('\n[LAYER 5] Model Health Checks...')

        # Quick ABM convergence check
        abm = PredictionMarketABM(true_prob=0.65, n_informed=10, n_noise=50, n_mm=5)
        abm.run(n_steps=500)
        abm_r = abm.get_results()
        print(f'  ABM convergence: true=0.65, final={abm_r["final_price"]:.4f}, err={abm_r["convergence_error"]:.4f}')

        # Whale tracker health
        print(f'  Whale tracker: {whale_scan_stats["total"]} scans, {whale_scan_stats["errors"]} errors')

        # ----- LAYER 6: Update Journal -----
        print('\n[LAYER 6] Journal Update...')
        self.journal.total_cycles += 1
        self._update_journal_stats()

        cycle_summary = {
            'cycle': self.journal.total_cycles,
            'timestamp': cycle_start,
            'strategy_version': 'v2',
            'markets_scanned': len(target_markets),
            'signals_generated': len(signal_gen._all_signals),
            'signals_actionable': len(signals),
            'signals_rejected': len(signal_gen._rejected_signals),
            'new_trades': len(new_trades),
            'trades_closed': n_closed_this_cycle,
            'open_positions': self.journal.open_trades,
            'paper_equity': round(self.journal.paper_capital + self.journal.cumulative_pnl, 2),
            'cumulative_pnl': self.journal.cumulative_pnl,
            'realised_pnl': self.journal.realised_pnl,
            'unrealised_pnl': self.journal.unrealised_pnl,
            'win_rate': self.journal.win_rate,
            'max_drawdown': self.journal.max_drawdown_pct,
            # v2 whale stats for this cycle
            'whale_scan': whale_scan_stats,
        }

        # Keep last 168 cycles (7 days of hourly runs)
        self.journal.cycle_history.append(cycle_summary)
        if len(self.journal.cycle_history) > 168:
            self.journal.cycle_history = self.journal.cycle_history[-168:]

        self._save_journal()

        # ----- Print Summary -----
        print('\n======================================================================')
        print('  CYCLE COMPLETE (v2 Whale Intelligence)')
        print('======================================================================')
        print(f'  Cycle #{self.journal.total_cycles} | {self.now.strftime("%Y-%m-%d %H:%M UTC")}')
        print(f'  Markets scanned: {len(target_markets)} | Signals: {len(signals)} actionable')
        print(f'  Whale signals: {whale_scan_stats["strong"]}S/{whale_scan_stats["moderate"]}M/{whale_scan_stats["weak"]}W | Clusters: {whale_scan_stats["clusters"]}')
        print(f'  New trades: {len(new_trades)} | Closed: {n_closed_this_cycle}')
        print(f'  Open positions: {self.journal.open_trades} / {self.MAX_OPEN_TRADES}')
        print(f'  Deployed: ${self.journal.paper_deployed:,.2f}')
        print(f'  Paper Equity: ${cycle_summary["paper_equity"]:,.2f}')
        print(f'  P&L: ${self.journal.cumulative_pnl:+,.2f} (realised: ${self.journal.realised_pnl:+,.2f})')
        print(f'  Win rate: {self.journal.win_rate:.0%} ({self.journal.wins}W/{self.journal.losses}L)')
        print(f'  Max DD: {self.journal.max_drawdown_pct:.1%}')
        if self.journal.sharpe_estimate != 0:
            print(f'  Sharpe (per-trade): {self.journal.sharpe_estimate:.3f}')
        print(f'  Avg whale confidence: {self.journal.avg_whale_confidence:.3f}')
        print('======================================================================')

        client.close()
        return cycle_summary

    # --- Telegram summary ---

    def format_telegram_summary(self, cycle: Dict[str, Any]) -> str:
        """Compact Telegram message for hourly updates with whale intelligence."""
        j = self.journal
        equity = j.paper_capital + j.cumulative_pnl
        whale_scan = cycle.get('whale_scan', {})

        lines = [
            f"POLYMARKET PAPER TRADER v2 | Cycle #{cycle['cycle']}",
            f"{self.now.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            f"Markets: {cycle['markets_scanned']} | Signals: {cycle['signals_actionable']} actionable / {cycle['signals_generated']} total",
            f"New: {cycle['new_trades']} | Closed: {cycle['trades_closed']} | Open: {cycle['open_positions']}/{self.MAX_OPEN_TRADES}",
            "",
            f"Equity: ${equity:,.2f}",
            f"P&L: ${j.cumulative_pnl:+,.2f} (R: ${j.realised_pnl:+,.2f} / U: ${j.unrealised_pnl:+,.2f})",
            f"Win: {j.win_rate:.0%} ({j.wins}W/{j.losses}L) | DD: {j.max_drawdown_pct:.1%}",
        ]

        if j.sharpe_estimate != 0:
            lines.append(f"Sharpe: {j.sharpe_estimate:.3f}")

        # v2: Whale intelligence summary
        lines.append("")
        lines.append("Whale Intel:")
        lines.append(f"  Scanned: {whale_scan.get('total', 0)} | Strong: {whale_scan.get('strong', 0)} | Mod: {whale_scan.get('moderate', 0)} | Weak: {whale_scan.get('weak', 0)}")
        lines.append(f"  Clusters: {whale_scan.get('clusters', 0)} | Aligned trades: {j.whale_aligned_trades} | Avg conf: {j.avg_whale_confidence:.3f}")

        # Top new trades (with whale data)
        recent_open = [t for t in j.trades if t['status'] == 'open']
        recent_open.sort(key=lambda t: abs(t.get('edge', 0)), reverse=True)

        if recent_open:
            lines.append("")
            lines.append("Top positions:")
            for t in recent_open[:5]:
                q = t['question'][:40] + ('...' if len(t['question']) > 40 else '')
                whale_tag = f"W:{t.get('whale_direction', '?')[:1]}" if t.get('whale_direction') else ""
                cluster_tag = " [CL]" if t.get('cluster_detected') else ""
                lines.append(
                    f"  {t['direction'][:3]} ${t['size_usd']:.0f} @ {t['entry_price']:.3f} "
                    f"| e={t['edge']:+.3f} {whale_tag}{cluster_tag} | {q}"
                )

        # Recent closes
        recent_closed = [t for t in j.trades if t['status'] == 'closed']
        recent_closed.sort(key=lambda t: t.get('exit_timestamp', ''), reverse=True)

        if recent_closed[:3]:
            lines.append("")
            lines.append("Recent closes:")
            for t in recent_closed[:3]:
                q = t['question'][:35] + ('...' if len(t['question']) > 35 else '')
                lines.append(f"  ${t['pnl']:+.2f} ({t['exit_reason']}) | {q}")

        lines.append("")
        lines.append("[PAPER MODE v2 - Whale Intel Active - No real funds]")
        return "\n".join(lines)

    def get_journal_json(self) -> str:
        """Return journal as formatted JSON string (for GitHub commit)."""
        return json.dumps(asdict(self.journal), indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    config = make_v2_config()
    trader = PaperTrader(config)

    if '--journal' in sys.argv:
        print(trader.get_journal_json())
        return

    cycle = trader.run_cycle(max_markets=50)

    if 'error' not in cycle:
        summary = trader.format_telegram_summary(cycle)
        print('\n--- TELEGRAM MESSAGE ---')
        print(summary)
        print('--- END ---')

        # Output structured data for pipeline consumption
        print('\n--- CYCLE_JSON_START ---')
        print(json.dumps(cycle, indent=2))
        print('--- CYCLE_JSON_END ---')

        print('\n--- JOURNAL_JSON_START ---')
        print(trader.get_journal_json())
        print('--- JOURNAL_JSON_END ---')

        print('\n--- TELEGRAM_MSG_START ---')
        print(summary)
        print('--- TELEGRAM_MSG_END ---')


if __name__ == '__main__':
    main()
