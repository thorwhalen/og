"""Basic usage examples for Own Ghost (OG).

This demonstrates the core functionality of OG:
- Starting observation
- Querying activity
- Getting AI-powered summaries
"""

from og import OG

# Create an OG instance
og = OG()

# ============================================================================
# Example 1: Start observing (daemon mode)
# ============================================================================

def example_daemon():
    """Run OG as a daemon to observe activity."""
    print("Starting OG daemon...")
    print("OG will now observe your activity.")
    print("Press Ctrl+C to stop")

    og.start()

    # The observers will run in the background
    # In a real scenario, this would be a long-running process

    # To stop:
    # og.stop()


# ============================================================================
# Example 2: Query recent activity
# ============================================================================

def example_query_activity():
    """Query and display recent activity."""

    # Get summary of today
    print("\n=== Today's Summary ===")
    summary = og.today(detail='medium')
    print(summary)

    # Get summary of last week
    print("\n=== This Week's Summary ===")
    weekly = og.week(detail='brief')
    print(weekly)

    # Get recent observations
    print("\n=== Recent Activity (Last 24 hours) ===")
    recent = og.recent_activity(days=1)
    print(f"Found {len(recent)} observations")

    # Show sample observations
    for obs in recent[:5]:
        print(f"- [{obs.timestamp}] {obs.event_type}: {obs.observer_name}")


# ============================================================================
# Example 3: Ask questions
# ============================================================================

def example_ask_questions():
    """Ask questions about your activity."""

    questions = [
        "What did I work on yesterday?",
        "How much time did I spend coding this week?",
        "What websites did I visit most frequently?",
        "What were my main accomplishments this week?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        answer = og.ask(question, days=7)
        print(f"A: {answer}")


# ============================================================================
# Example 4: Generate reports
# ============================================================================

def example_generate_reports():
    """Generate different types of reports."""

    # Productivity report
    print("\n=== Productivity Report (Last 7 Days) ===")
    productivity_report = og.report(days=7, report_type='productivity')
    print(productivity_report)

    # Technical report
    print("\n=== Technical Report (Last 7 Days) ===")
    tech_report = og.report(days=7, report_type='technical')
    print(tech_report)

    # Comprehensive report
    print("\n=== Comprehensive Report (Last 7 Days) ===")
    comprehensive = og.report(days=7, report_type='comprehensive')
    print(comprehensive)


# ============================================================================
# Example 5: Manage observers
# ============================================================================

def example_manage_observers():
    """Manage and configure observers."""

    # List all observers
    print("\n=== Available Observers ===")
    observers = og.list_observers()
    for name, info in observers.items():
        status = "Enabled" if info['enabled'] else "Disabled"
        print(f"- {name}: {status}")
        if info['metadata']:
            print(f"  Category: {info['metadata'].get('category')}")

    # Enable/disable specific observers
    print("\n=== Disabling keyboard observer ===")
    og.disable_observer('keyboard')

    print("\n=== Enabling it again ===")
    og.enable_observer('keyboard')

    # Get system status
    print("\n=== System Status ===")
    status = og.status()
    print(f"Running: {status['running']}")
    print(f"Enabled observers: {status['enabled_observers']}")
    print(f"Stores: {len(status['stores'])}")


# ============================================================================
# Example 6: Get statistics
# ============================================================================

def example_statistics():
    """Get statistics about collected observations."""

    print("\n=== OG Statistics ===")
    stats = og.stats()

    print(f"Total observations: {stats['total_observations']}")
    print("\nObservations by store:")
    for store_name, count in stats['stores'].items():
        print(f"  {store_name}: {count}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    import sys

    print("Own Ghost (OG) - Basic Usage Examples")
    print("=" * 60)

    # You can run specific examples or all of them
    if len(sys.argv) > 1:
        example_name = sys.argv[1]

        examples = {
            'daemon': example_daemon,
            'query': example_query_activity,
            'ask': example_ask_questions,
            'report': example_generate_reports,
            'manage': example_manage_observers,
            'stats': example_statistics,
        }

        if example_name in examples:
            examples[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        print("\nRun with: python basic_usage.py <example_name>")
        print("Examples: daemon, query, ask, report, manage, stats")
        print("\nOr use the OG CLI:")
        print("  og start          # Start the daemon")
        print("  og summary        # Get today's summary")
        print("  og ask 'question' # Ask a question")
        print("  og report         # Generate a report")
        print("  og status         # Check status")
