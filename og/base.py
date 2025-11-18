"""Base classes and protocols for OG (Own Ghost) observers.

This module defines the core abstractions for the observer system:
- Observer protocol for implementing custom observers
- Observation data class for storing captured events
- Base observer implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from collections.abc import AsyncIterator, Iterator


@dataclass
class Observation:
    """A single observation captured by an observer.

    Attributes:
        timestamp: When the observation was made
        observer_name: Name of the observer that created this
        event_type: Type of event (e.g., 'keystroke', 'github_commit', 'page_visit')
        data: The actual observation data
        metadata: Additional metadata about the observation
        tags: Optional tags for categorization and filtering
    """

    timestamp: datetime
    observer_name: str
    event_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure timestamp is a datetime object."""
        if isinstance(self.timestamp, (int, float)):
            self.timestamp = datetime.fromtimestamp(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary for storage."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'observer_name': self.observer_name,
            'event_type': self.event_type,
            'data': self.data,
            'metadata': self.metadata,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Observation':
        """Create observation from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(d['timestamp']),
            observer_name=d['observer_name'],
            event_type=d['event_type'],
            data=d['data'],
            metadata=d.get('metadata', {}),
            tags=d.get('tags', []),
        )


@runtime_checkable
class Observer(Protocol):
    """Protocol for implementing observers.

    An observer monitors a specific activity source and yields observations.
    Observers can be synchronous or asynchronous.
    """

    name: str
    enabled: bool

    def observe(self) -> Iterator[Observation]:
        """Synchronously observe and yield observations.

        This method should be implemented for observers that can
        synchronously poll or check for events.
        """
        ...

    async def observe_async(self) -> AsyncIterator[Observation]:
        """Asynchronously observe and yield observations.

        This method should be implemented for observers that
        need to run asynchronously (e.g., listening to events).
        """
        ...

    def start(self) -> None:
        """Start the observer (e.g., setup listeners)."""
        ...

    def stop(self) -> None:
        """Stop the observer (e.g., cleanup resources)."""
        ...


class BaseObserver(ABC):
    """Base class for implementing observers.

    This provides a common implementation that can be extended.
    Subclasses should implement either observe() or observe_async().
    """

    def __init__(self, name: str, enabled: bool = True, **config):
        """Initialize the observer.

        Args:
            name: Unique name for this observer
            enabled: Whether the observer is enabled by default
            **config: Additional configuration options
        """
        self.name = name
        self.enabled = enabled
        self.config = config
        self._running = False

    @abstractmethod
    def observe(self) -> Iterator[Observation]:
        """Synchronously observe and yield observations.

        Subclasses should implement this method for synchronous observation.
        """
        raise NotImplementedError("Subclass must implement observe()")

    async def observe_async(self) -> AsyncIterator[Observation]:
        """Asynchronously observe and yield observations.

        Default implementation wraps the synchronous observe() method.
        Override for true async behavior.
        """
        for observation in self.observe():
            yield observation

    def start(self) -> None:
        """Start the observer."""
        self._running = True

    def stop(self) -> None:
        """Stop the observer."""
        self._running = False

    def is_running(self) -> bool:
        """Check if observer is running."""
        return self._running

    def create_observation(
        self,
        event_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> Observation:
        """Helper method to create an observation.

        Args:
            event_type: Type of event
            data: Event data
            metadata: Optional metadata
            tags: Optional tags

        Returns:
            A new Observation instance
        """
        return Observation(
            timestamp=datetime.now(),
            observer_name=self.name,
            event_type=event_type,
            data=data,
            metadata=metadata or {},
            tags=tags or [],
        )


class PollingObserver(BaseObserver):
    """Base class for observers that poll at regular intervals.

    This is useful for observers that check state periodically
    rather than listening to events.
    """

    def __init__(
        self,
        name: str,
        poll_interval: float = 60.0,
        enabled: bool = True,
        **config
    ):
        """Initialize polling observer.

        Args:
            name: Observer name
            poll_interval: How often to poll (in seconds)
            enabled: Whether enabled by default
            **config: Additional configuration
        """
        super().__init__(name, enabled, **config)
        self.poll_interval = poll_interval

    @abstractmethod
    def poll(self) -> list[Observation]:
        """Poll for new observations.

        This method is called at each poll interval.
        Should return a list of new observations since last poll.
        """
        raise NotImplementedError("Subclass must implement poll()")

    def observe(self) -> Iterator[Observation]:
        """Poll at regular intervals and yield observations."""
        import time

        self.start()
        last_poll = time.time()

        while self._running:
            current = time.time()

            if current - last_poll >= self.poll_interval:
                try:
                    observations = self.poll()
                    for obs in observations:
                        yield obs
                    last_poll = current
                except Exception as e:
                    # Log error but continue polling
                    print(f"Error in {self.name} poll: {e}")

            # Sleep a bit to avoid busy waiting
            time.sleep(min(1.0, self.poll_interval / 10))
