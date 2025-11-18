"""Daemon and CLI for OG (Own Ghost).

This module provides the command-line interface and daemon functionality
for running OG as a background service.
"""

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from og.query import OG


class OGDaemon:
    """Daemon for running OG observers in the background."""

    def __init__(self, storage_dir: str | None = None):
        """Initialize the daemon.

        Args:
            storage_dir: Directory for storing observations
        """
        self.og = OG(storage_dir=storage_dir)
        self._shutdown = False

    def start(self):
        """Start the daemon."""
        print("Starting Own Ghost daemon...")
        print(f"Storage: {self.og.mall.root_dir}")

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start observers
        self.og.start()

        print("Own Ghost is now observing...")
        print("Press Ctrl+C to stop")

        # Keep running until shutdown
        try:
            while not self._shutdown:
                time.sleep(1)
        finally:
            self.stop()

    def stop(self):
        """Stop the daemon."""
        print("\nStopping Own Ghost daemon...")
        self.og.stop()
        print("Stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self._shutdown = True


def cli():
    """Command-line interface for OG."""
    parser = argparse.ArgumentParser(
        description='Own Ghost - Personal Activity Observer and AI Assistant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the daemon
  og start

  # Get summary of today
  og summary

  # Get summary of last 7 days
  og summary --days 7

  # Ask a question
  og ask "What did I work on yesterday?"

  # Generate a report
  og report --days 7 --type productivity

  # Get system status
  og status

  # List observers
  og observers
        """,
    )

    parser.add_argument(
        '--storage',
        type=str,
        help='Storage directory for observations',
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start the OG daemon')

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop the OG daemon')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')

    # Summary command
    summary_parser = subparsers.add_parser(
        'summary', help='Get activity summary'
    )
    summary_parser.add_argument(
        '--days', type=int, default=1, help='Number of days to summarize'
    )
    summary_parser.add_argument(
        '--detail',
        choices=['brief', 'medium', 'detailed'],
        default='medium',
        help='Level of detail',
    )

    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question about activity')
    ask_parser.add_argument('question', nargs='+', help='Question to ask')
    ask_parser.add_argument(
        '--days', type=int, default=7, help='Days of history to consider'
    )

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate a detailed report')
    report_parser.add_argument(
        '--days', type=int, default=7, help='Number of days to analyze'
    )
    report_parser.add_argument(
        '--type',
        choices=['productivity', 'technical', 'comprehensive'],
        default='productivity',
        help='Type of report',
    )

    # Observers command
    observers_parser = subparsers.add_parser('observers', help='List observers')
    observers_parser.add_argument(
        '--enable', type=str, help='Enable an observer'
    )
    observers_parser.add_argument(
        '--disable', type=str, help='Disable an observer'
    )

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')

    # Recent command
    recent_parser = subparsers.add_parser('recent', help='Show recent activity')
    recent_parser.add_argument(
        '--hours', type=int, help='Show last N hours'
    )
    recent_parser.add_argument(
        '--days', type=int, help='Show last N days'
    )
    recent_parser.add_argument(
        '--type', type=str, help='Filter by event type'
    )

    # Setup command
    setup_parser = subparsers.add_parser(
        'setup',
        help='Interactive setup and configuration wizard',
        aliases=['config', 'configure']
    )

    args = parser.parse_args()

    # Create OG instance
    og = OG(storage_dir=args.storage)

    # Handle commands
    if args.command == 'start':
        daemon = OGDaemon(storage_dir=args.storage)
        daemon.start()

    elif args.command == 'stop':
        print("Stopping OG daemon...")
        # TODO: Implement proper daemon stopping (PID file, etc.)
        print("Note: Use Ctrl+C to stop the daemon process")

    elif args.command == 'status':
        status = og.status()
        print("\nOwn Ghost Status")
        print("=" * 50)
        print(f"Running: {status['running']}")
        print(f"Total Observers: {status['total_observers']}")
        print(f"Enabled Observers: {status['enabled_observers']}")
        print(f"\nEnabled: {', '.join(status['observers']['enabled'])}")
        print(f"\nStores: {', '.join(status['stores'])}")

    elif args.command == 'summary':
        print(f"\nSummary (last {args.days} days):")
        print("=" * 50)
        summary = og.summary(days=args.days, detail=args.detail)
        print(summary)

    elif args.command == 'ask':
        question = ' '.join(args.question)
        print(f"\nQuestion: {question}")
        print("=" * 50)
        answer = og.ask(question, days=args.days)
        print(answer)

    elif args.command == 'report':
        print(f"\n{args.type.title()} Report (last {args.days} days):")
        print("=" * 50)
        report = og.report(days=args.days, report_type=args.type)
        print(report)

    elif args.command == 'observers':
        if args.enable:
            og.enable_observer(args.enable)
        elif args.disable:
            og.disable_observer(args.disable)
        else:
            observers = og.list_observers()
            print("\nObservers:")
            print("=" * 50)
            for name, info in observers.items():
                status = "✓" if info['enabled'] else "✗"
                running = "(running)" if info['running'] else ""
                print(f"{status} {name} {running}")
                if info['metadata']:
                    print(f"  Category: {info['metadata'].get('category', 'N/A')}")

    elif args.command == 'stats':
        stats = og.stats()
        print("\nStatistics:")
        print("=" * 50)
        print(f"Total Observations: {stats['total_observations']}")
        print("\nBy Store:")
        for store_name, count in stats['stores'].items():
            print(f"  {store_name}: {count}")

    elif args.command == 'recent':
        observations = og.recent_activity(
            hours=args.hours,
            days=args.days,
            event_type=args.type,
        )
        print(f"\nRecent Activity ({len(observations)} observations):")
        print("=" * 50)
        for obs in observations[-20:]:  # Show last 20
            print(
                f"[{obs.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"{obs.event_type} - {obs.observer_name}"
            )
            # Show a brief preview of data
            if obs.event_type == 'github_push':
                print(f"  Repo: {obs.data.get('repo')}")
            elif obs.event_type == 'browser_visit':
                print(f"  {obs.data.get('title')} - {obs.data.get('url')}")

    elif args.command in ['setup', 'config', 'configure']:
        from og.setup import SetupWizard
        wizard = SetupWizard()
        wizard.run()

    else:
        parser.print_help()


if __name__ == '__main__':
    cli()
