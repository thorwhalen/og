"""Observer implementations for OG (Own Ghost).

This package contains implementations of various observers
for tracking different types of activities.
"""

from og.observers.github import GithubObserver
from og.observers.keyboard import KeyboardObserver
from og.observers.browser import BrowserObserver
from og.observers.apps import AppUsageObserver
from og.observers.filesystem import FileSystemObserver
from og.observers.git import GitCommitObserver
from og.observers.terminal import TerminalHistoryObserver
from og.observers.email import EmailObserver
from og.observers.calendar import CalendarObserver
from og.observers.slack import SlackObserver
from og.observers.music import MusicObserver
from og.observers.ide import IDEObserver

__all__ = [
    'GithubObserver',
    'KeyboardObserver',
    'BrowserObserver',
    'AppUsageObserver',
    'FileSystemObserver',
    'GitCommitObserver',
    'TerminalHistoryObserver',
    'EmailObserver',
    'CalendarObserver',
    'SlackObserver',
    'MusicObserver',
    'IDEObserver',
]


def register_default_observers(registry):
    """Register all default observers with a registry.

    Args:
        registry: ObserverRegistry instance
    """
    from og.observers.github import GithubObserver
    from og.observers.keyboard import KeyboardObserver
    from og.observers.browser import BrowserObserver
    from og.observers.apps import AppUsageObserver
    from og.observers.filesystem import FileSystemObserver
    from og.observers.git import GitCommitObserver
    from og.observers.terminal import TerminalHistoryObserver
    from og.observers.email import EmailObserver
    from og.observers.calendar import CalendarObserver
    from og.observers.slack import SlackObserver
    from og.observers.music import MusicObserver
    from og.observers.ide import IDEObserver

    # Register with metadata (core observers - always enabled)
    registry.register('github', category='version_control')(GithubObserver())
    registry.register('keyboard', category='input')(KeyboardObserver())
    registry.register('browser', category='web')(BrowserObserver())
    registry.register('apps', category='system')(AppUsageObserver())
    registry.register('filesystem', category='files')(FileSystemObserver())
    registry.register('git', category='version_control')(GitCommitObserver())
    registry.register('terminal', category='system')(TerminalHistoryObserver())

    # Additional observers (disabled by default - require configuration)
    registry.register('email', category='communication')(EmailObserver(enabled=False))
    registry.register('calendar', category='schedule')(CalendarObserver(enabled=False))
    registry.register('slack', category='communication')(SlackObserver(enabled=False))
    registry.register('music', category='lifestyle')(MusicObserver(enabled=False))
    registry.register('ide', category='development')(IDEObserver(enabled=False))
