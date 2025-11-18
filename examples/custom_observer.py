"""Example of creating a custom observer for OG.

This shows how to create your own observer to track
custom activities.
"""

from datetime import datetime
from collections.abc import Iterator
import time

from og import BaseObserver, Observation, OG, ObserverRegistry


# ============================================================================
# Example 1: Simple custom observer
# ============================================================================

class SimpleTaskObserver(BaseObserver):
    """A simple observer that tracks task completion.

    This could be integrated with a todo list app, task manager, etc.
    """

    def __init__(self, name: str = 'tasks', enabled: bool = True):
        super().__init__(name, enabled)
        self.completed_tasks = []

    def mark_task_complete(self, task_name: str, category: str = 'general'):
        """Mark a task as complete."""
        obs = self.create_observation(
            event_type='task_completed',
            data={
                'task': task_name,
                'category': category,
            },
            tags=['task', 'productivity', category],
        )
        self.completed_tasks.append(obs)
        return obs

    def observe(self) -> Iterator[Observation]:
        """Yield task completion observations."""
        self.start()

        while self._running:
            # Yield any completed tasks
            while self.completed_tasks:
                yield self.completed_tasks.pop(0)

            time.sleep(0.1)


# ============================================================================
# Example 2: Polling-based custom observer
# ============================================================================

from og import PollingObserver

class EmailObserver(PollingObserver):
    """Observer that checks email activity.

    This is a mock example - a real implementation would
    connect to email APIs (Gmail, Outlook, etc.)
    """

    def __init__(
        self,
        name: str = 'email',
        poll_interval: float = 300.0,  # Check every 5 minutes
        enabled: bool = True,
    ):
        super().__init__(name, poll_interval, enabled)
        self._last_email_count = 0

    def poll(self) -> list[Observation]:
        """Check for new emails."""
        observations = []

        # In a real implementation, this would:
        # 1. Connect to email API
        # 2. Get unread count
        # 3. Get recent emails
        # 4. Create observations

        # Mock: Pretend we got 3 new emails
        new_emails = self._mock_get_new_emails()

        for email in new_emails:
            obs = self.create_observation(
                event_type='email_received',
                data={
                    'from': email['from'],
                    'subject': email['subject'],
                    'has_attachments': email.get('attachments', False),
                },
                tags=['email', 'communication'],
            )
            observations.append(obs)

        return observations

    def _mock_get_new_emails(self):
        """Mock function to simulate getting emails."""
        # In real implementation, this would call email API
        return []


# ============================================================================
# Example 3: Using custom observers with OG
# ============================================================================

def example_use_custom_observer():
    """Example of using custom observers."""

    # Create custom observer
    task_observer = SimpleTaskObserver()

    # Create custom registry
    registry = ObserverRegistry()

    # Register the custom observer
    registry['tasks'] = task_observer

    # Also register default observers
    from og.observers import register_default_observers
    register_default_observers(registry)

    # Create OG instance with custom registry
    og = OG(registry=registry)

    # Now you can use it
    print("Custom observer example:")
    print(f"Observers: {list(og.list_observers().keys())}")

    # Mark some tasks complete
    task_observer.mark_task_complete("Finish OG implementation", category="coding")
    task_observer.mark_task_complete("Write documentation", category="docs")

    print("\nMarked 2 tasks complete!")

    # These observations will be picked up by OG
    # and can be queried/summarized


# ============================================================================
# Example 4: Event-driven observer
# ============================================================================

class WebhookObserver(BaseObserver):
    """Observer that listens for webhook events.

    This could receive events from external services
    (Slack, Discord, Calendar, etc.)
    """

    def __init__(self, name: str = 'webhooks', port: int = 8080, enabled: bool = True):
        super().__init__(name, enabled)
        self.port = port
        self.events_queue = []

    def on_webhook_event(self, event_data: dict):
        """Handle incoming webhook event."""
        obs = self.create_observation(
            event_type=f"webhook_{event_data.get('type', 'unknown')}",
            data=event_data,
            tags=['webhook', 'external'],
        )
        self.events_queue.append(obs)

    def observe(self) -> Iterator[Observation]:
        """Yield webhook events as they arrive."""
        self.start()

        # In a real implementation, this would:
        # 1. Start a web server (Flask, FastAPI, etc.)
        # 2. Listen for POST requests
        # 3. Queue events
        # 4. Yield them here

        while self._running:
            while self.events_queue:
                yield self.events_queue.pop(0)

            time.sleep(0.1)


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("Custom Observer Examples")
    print("=" * 60)

    example_use_custom_observer()

    print("\n\nCustom observer patterns demonstrated:")
    print("1. SimpleTaskObserver - Manual event recording")
    print("2. EmailObserver - Polling-based observer")
    print("3. WebhookObserver - Event-driven observer")

    print("\n\nTo create your own observer:")
    print("1. Inherit from BaseObserver or PollingObserver")
    print("2. Implement the observe() or poll() method")
    print("3. Use create_observation() to create observations")
    print("4. Register with ObserverRegistry")
    print("5. Use with OG!")
