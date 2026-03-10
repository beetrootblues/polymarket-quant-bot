"""Run V4.1 Paper Trade Cycle
================================
Orchestrator script for the Polymarket V4.1 signal engine.
Loads journal, runs a full scan+trade cycle, saves updated journal,
and outputs a Telegram-ready summary.

Usage:
  python run_v4_cycle.py                  # default 1000 markets
  python run_v4_cycle.py --markets 500    # custom market count
  python run_v4_cycle.py --journal path   # custom journal path
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add parent dir so signal_engine_v4 can be imported from various locations
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'polymarket-bot'))

from signal_engine_v4 import PaperTraderV4, SignalGeneratorV4, V4SignalType

# --- Config ---
DEFAULT_JOURNAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'polymarket-bot', 'paper_journal_v4.json')
DEFAULT_MAX_MARKETS = 1000
INITIAL_EQUITY = 10000


def load_journal(path: str) -> dict:
    """Load existing journal or return None for fresh start."""
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"[V4.1] Loaded journal: cycle {data.get('cycle_count', 0)}, "
                  f"equity ${data.get('portfolio', {}).get('cash', 0):,.2f}, "
                  f"{len(data.get('open_positions', []))} open positions")
            return data
        except Exception as e:
            print(f"[V4.1] Journal load error: {e} -- starting fresh")
    return None


def save_journal(trader: PaperTraderV4, path: str):
    """Save journal to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(trader.journal, f, indent=2, default=str)
    print(f"[V4.1] Journal saved to {path}")


def format_full_report(result: dict, trader: PaperTraderV4) -> str:
    """Build detailed Telegram summary from cycle output."""
    lines = [
        f"POLYMARKET V4.1 CYCLE #{result['cycle']}",
        f"{result['timestamp']}",
        f"Scanned: {result['markets_scanned']} markets + {result.get('events_scanned', 0)} events",
        "",
        "SIGNAL BREAKDOWN:",
        f"  L2 Endgame:      {result['breakdown']['endgame']}",
        f"  L3 Multi-Outcome: {result['breakdown']['multi_outcome']}",
        f"  L4 Correlation:   {result['breakdown']['correlation']}",
        f"  L5 Sentiment:     {result['breakdown']['sentiment']}",
        f"  TOTAL:            {result['signals_found']}",
        "",
    ]

    # New trades this cycle
    if result['new_trades'] > 0:
        lines.append(f"NEW TRADES: {result['new_trades']}")
        for t in result.get('trades_detail', []):
            ann_str = f" ({t['annualized_return']:.0%} ann)" if t.get('annualized_return') else ""
            lines.append(f"  {t['action']} ${t['size']:.0f} edge={t['edge']:.3f}{ann_str}")
            lines.append(f"    {t['question'][:70]}")
        lines.append("")
    else:
        lines.append("NEW TRADES: 0")
        lines.append("")

    # Portfolio status
    port = result['portfolio']
    equity = port.get('equity', port.get('cash', 0))
    initial = trader.journal.get('portfolio', {}).get('initial_equity', INITIAL_EQUITY)
    pnl = equity - initial
    total_trades = port.get('total_trades', 0)
    open_count = port.get('open', 0)

    lines.extend([
        "PORTFOLIO:",
        f"  Equity:  ${equity:,.2f}",
        f"  P&L:     ${pnl:+,.2f} ({pnl/initial*100:+.1f}%)",
        f"  Cash:    ${port.get('cash', 0):,.2f}",
        f"  Open:    {open_count} positions",
        f"  Total:   {total_trades} trades",
        "",
    ])

    # Errors
    if result.get('errors'):
        lines.append(f"ERRORS ({len(result['errors'])})")
        for e in result['errors'][:5]:
            lines.append(f"  {str(e)[:80]}")
        lines.append("")

    lines.append("[PAPER MODE v4.1]")
    return "\n".join(lines)


async def run_cycle(journal_path: str = DEFAULT_JOURNAL,
                    max_markets: int = DEFAULT_MAX_MARKETS) -> dict:
    """Execute a full V4.1 paper trade cycle."""
    print(f"\n{'='*60}")
    print(f"  POLYMARKET V4.1 PAPER TRADE CYCLE")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"{'='*60}\n")

    # Initialize trader
    trader = PaperTraderV4(initial_equity=INITIAL_EQUITY)

    # Load existing journal if available
    existing = load_journal(journal_path)
    if existing:
        trader.load_journal(existing)

    # Run the cycle
    result = await trader.run_cycle(max_markets=max_markets)

    # Save updated journal
    save_journal(trader, journal_path)

    # Generate and print summary
    summary = format_full_report(result, trader)
    print(f"\n{summary}")

    # Also save summary to a text file for Telegram
    summary_path = os.path.join(os.path.dirname(journal_path) if os.path.dirname(journal_path) else '.',
                                'v4_telegram_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary)
    print(f"\n[V4.1] Telegram summary saved to {summary_path}")

    # Save cycle result JSON
    result_path = os.path.join(os.path.dirname(journal_path) if os.path.dirname(journal_path) else '.',
                               'v4_cycle_result.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[V4.1] Cycle result saved to {result_path}")

    return {
        'result': result,
        'summary': summary,
        'journal': trader.journal,
        'journal_path': journal_path,
        'summary_path': summary_path,
        'result_path': result_path
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run V4.1 Paper Trade Cycle')
    parser.add_argument('--markets', type=int, default=DEFAULT_MAX_MARKETS,
                        help=f'Max markets to scan (default: {DEFAULT_MAX_MARKETS})')
    parser.add_argument('--journal', type=str, default=DEFAULT_JOURNAL,
                        help=f'Journal file path (default: {DEFAULT_JOURNAL})')
    args = parser.parse_args()

    output = asyncio.run(run_cycle(
        journal_path=args.journal,
        max_markets=args.markets
    ))

    # Exit with signal count for CI/automation
    sys.exit(0 if output['result']['signals_found'] > 0 else 1)
