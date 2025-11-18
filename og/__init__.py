"""Own Ghost (OG) - A daemon that observes your activities and helps you make AI contexts from them.

Own Ghost is a modular, extensible system for self-observation and intelligent reflection.
It allows you to register observers that monitor activity across various sources:
GitHub commits, keyboard input, application usage, browser history, and more.

Quick Start:
    >>> from og import OG
    >>> og = OG()
    >>> og.start()  # Start observing
    >>> # Later...
    >>> print(og.summary(days=1))  # Get summary of today
    >>> print(og.ask("What did I work on yesterday?"))

Main Classes:
    - OG: Main interface for querying and controlling the system
    - Observation: Data class for observations
    - Observer: Protocol for implementing custom observers
    - ObserverRegistry: Registry for managing observers
    - ObservationMall: Storage system for observations

Observers:
    - GithubObserver: Tracks GitHub activity
    - KeyboardObserver: Tracks keyboard activity
    - BrowserObserver: Tracks browser history
    - AppUsageObserver: Tracks application usage
    - FileSystemObserver: Tracks file system changes
    - GitCommitObserver: Tracks local git commits
    - TerminalHistoryObserver: Tracks shell commands
"""

# Core classes
from og.base import Observation, Observer, BaseObserver, PollingObserver
from og.registry import ObserverRegistry, create_default_registry
from og.storage import ObservationMall, ObservationStore, mk_observation_store
from og.query import OG

# AI agent
from og.ai import OGAgent

# Observers
from og.observers import (
    GithubObserver,
    KeyboardObserver,
    BrowserObserver,
    AppUsageObserver,
    FileSystemObserver,
    GitCommitObserver,
    TerminalHistoryObserver,
    EmailObserver,
    CalendarObserver,
    SlackObserver,
    MusicObserver,
    IDEObserver,
)

__version__ = '0.0.1'

__all__ = [
    # Main interface
    'OG',
    # Core classes
    'Observation',
    'Observer',
    'BaseObserver',
    'PollingObserver',
    'ObserverRegistry',
    'create_default_registry',
    'ObservationMall',
    'ObservationStore',
    'mk_observation_store',
    # AI
    'OGAgent',
    # Core observers
    'GithubObserver',
    'KeyboardObserver',
    'BrowserObserver',
    'AppUsageObserver',
    'FileSystemObserver',
    'GitCommitObserver',
    'TerminalHistoryObserver',
    # Additional observers
    'EmailObserver',
    'CalendarObserver',
    'SlackObserver',
    'MusicObserver',
    'IDEObserver',
]
