"""Observer registry system for OG (Own Ghost).

This module provides a registry for managing observers,
similar to the ComponentRegistry pattern in ef but tailored
for observer management.
"""

from collections.abc import Iterator, MutableMapping
from typing import Any, Callable, Optional

from og.base import Observer


class ObserverRegistry(MutableMapping):
    """Registry for managing observers.

    Provides a Mapping interface to observers, allowing easy
    discovery, registration, and access.

    Example:
        >>> registry = ObserverRegistry()
        >>> registry['github'] = GithubObserver()
        >>> 'github' in registry
        True
        >>> observer = registry['github']
    """

    def __init__(self):
        """Initialize the registry."""
        self._observers: dict[str, Observer] = {}
        self._metadata: dict[str, dict] = {}

    def __getitem__(self, key: str) -> Observer:
        """Get an observer by name."""
        return self._observers[key]

    def __setitem__(self, key: str, value: Observer) -> None:
        """Register an observer."""
        if not hasattr(value, 'observe'):
            raise TypeError(
                f"Observer must have an 'observe' method, got {type(value)}"
            )
        self._observers[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove an observer from the registry."""
        del self._observers[key]
        if key in self._metadata:
            del self._metadata[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over observer names."""
        return iter(self._observers)

    def __len__(self) -> int:
        """Get number of registered observers."""
        return len(self._observers)

    def register(
        self, name: str | None = None, **metadata
    ) -> Callable[[Observer], Observer]:
        """Decorator to register an observer.

        Args:
            name: Optional name for the observer (uses observer.name if None)
            **metadata: Additional metadata to store

        Returns:
            Decorator function

        Example:
            >>> registry = ObserverRegistry()
            >>> @registry.register('github', category='version_control')
            ... class GithubObserver:
            ...     name = 'github'
            ...     def observe(self): pass
        """

        def decorator(observer_cls_or_instance: Observer) -> Observer:
            # Handle both classes and instances
            if isinstance(observer_cls_or_instance, type):
                # It's a class, instantiate it
                observer = observer_cls_or_instance()
            else:
                # It's already an instance
                observer = observer_cls_or_instance

            key = name or getattr(observer, 'name', None)
            if key is None:
                raise ValueError(
                    "Observer must have a 'name' attribute or name must be provided"
                )

            self[key] = observer

            if metadata:
                self._metadata[key] = metadata

            return observer

        return decorator

    def get_metadata(self, key: str) -> dict:
        """Get metadata for an observer.

        Args:
            key: Observer name

        Returns:
            Dictionary of metadata
        """
        return self._metadata.get(key, {})

    def list_by_category(self, category: str) -> list[str]:
        """List observers by category.

        Args:
            category: Category to filter by

        Returns:
            List of observer names in that category
        """
        return [
            name
            for name, meta in self._metadata.items()
            if meta.get('category') == category
        ]

    def get_enabled(self) -> list[Observer]:
        """Get all enabled observers.

        Returns:
            List of enabled observer instances
        """
        return [
            obs for obs in self._observers.values() if getattr(obs, 'enabled', True)
        ]

    def enable(self, name: str) -> None:
        """Enable an observer.

        Args:
            name: Observer name
        """
        observer = self[name]
        if hasattr(observer, 'enabled'):
            observer.enabled = True

    def disable(self, name: str) -> None:
        """Disable an observer.

        Args:
            name: Observer name
        """
        observer = self[name]
        if hasattr(observer, 'enabled'):
            observer.enabled = False

    def start_all(self) -> None:
        """Start all enabled observers."""
        for observer in self.get_enabled():
            if hasattr(observer, 'start'):
                try:
                    observer.start()
                except Exception as e:
                    print(f"Error starting {observer.name}: {e}")

    def stop_all(self) -> None:
        """Stop all observers."""
        for observer in self._observers.values():
            if hasattr(observer, 'stop'):
                try:
                    observer.stop()
                except Exception as e:
                    print(f"Error stopping {observer.name}: {e}")


def create_default_registry() -> ObserverRegistry:
    """Create a registry with default observers.

    This function creates a registry and registers the default observers
    that come with OG.

    Returns:
        ObserverRegistry with default observers registered
    """
    from og.observers import (
        register_default_observers,
    )  # Import here to avoid circular imports

    registry = ObserverRegistry()
    register_default_observers(registry)
    return registry
