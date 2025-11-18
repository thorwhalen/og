"""File system activity observer for OG (Own Ghost).

This observer tracks file system changes.
"""

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Optional

from og.base import BaseObserver, Observation


class FileSystemObserver(BaseObserver):
    """Observer for file system activity.

    Tracks:
    - Files created
    - Files modified
    - Files deleted
    - Directory changes
    """

    def __init__(
        self,
        name: str = 'filesystem',
        enabled: bool = True,
        watch_paths: Optional[list[str]] = None,
        ignore_patterns: Optional[list[str]] = None,
        **config
    ):
        """Initialize filesystem observer.

        Args:
            name: Observer name
            enabled: Whether observer is enabled
            watch_paths: Paths to watch (watches home directory if None)
            ignore_patterns: Patterns to ignore (e.g., '*.pyc', '.git/*')
            **config: Additional configuration
        """
        super().__init__(name, enabled, **config)
        self.watch_paths = watch_paths or [str(Path.home())]
        self.ignore_patterns = ignore_patterns or [
            '*.pyc',
            '__pycache__/*',
            '.git/*',
            'node_modules/*',
            '.DS_Store',
        ]
        self._observer = None

    def start(self):
        """Start watching file system."""
        super().start()

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class OGFileSystemHandler(FileSystemEventHandler):
                def __init__(self, parent):
                    self.parent = parent
                    self.observations = []

                def on_created(self, event):
                    if not event.is_directory:
                        obs = self.parent.create_observation(
                            event_type='file_created',
                            data={
                                'path': event.src_path,
                                'is_directory': event.is_directory,
                            },
                            tags=['filesystem', 'create'],
                        )
                        self.observations.append(obs)

                def on_modified(self, event):
                    if not event.is_directory:
                        obs = self.parent.create_observation(
                            event_type='file_modified',
                            data={
                                'path': event.src_path,
                                'is_directory': event.is_directory,
                            },
                            tags=['filesystem', 'modify'],
                        )
                        self.observations.append(obs)

                def on_deleted(self, event):
                    obs = self.parent.create_observation(
                        event_type='file_deleted',
                        data={
                            'path': event.src_path,
                            'is_directory': event.is_directory,
                        },
                        tags=['filesystem', 'delete'],
                    )
                    self.observations.append(obs)

            self._handler = OGFileSystemHandler(self)
            self._observer = Observer()

            for path in self.watch_paths:
                self._observer.schedule(self._handler, path, recursive=True)

            self._observer.start()

        except ImportError:
            print(
                "watchdog is required for filesystem observer. "
                "Install it with: pip install watchdog"
            )

    def stop(self):
        """Stop watching file system."""
        super().stop()
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def observe(self) -> Iterator[Observation]:
        """Observe file system and yield observations."""
        import time

        self.start()

        while self._running:
            # Yield any observations collected by the handler
            if hasattr(self, '_handler') and self._handler.observations:
                for obs in self._handler.observations:
                    yield obs
                self._handler.observations.clear()

            time.sleep(0.1)
