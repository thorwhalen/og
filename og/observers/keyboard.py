"""Keyboard activity observer for OG (Own Ghost).

This observer tracks keyboard activity including keystrokes, shortcuts,
and typing patterns.
"""

from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Optional

from og.base import BaseObserver, Observation


class KeyboardObserver(BaseObserver):
    """Observer for keyboard activity.

    Tracks:
    - Keystroke counts
    - Application-specific typing activity
    - Keyboard shortcuts
    - Typing sessions

    Note: This observer respects privacy by NOT logging actual text content,
    only metadata about typing activity (counts, timing, applications).
    """

    def __init__(
        self,
        name: str = 'keyboard',
        enabled: bool = True,
        aggregate_interval: float = 60.0,  # Aggregate stats every minute
        track_shortcuts: bool = True,
        **config
    ):
        """Initialize keyboard observer.

        Args:
            name: Observer name
            enabled: Whether observer is enabled
            aggregate_interval: How often to emit aggregated observations (seconds)
            track_shortcuts: Whether to track keyboard shortcuts
            **config: Additional configuration
        """
        super().__init__(name, enabled, **config)
        self.aggregate_interval = aggregate_interval
        self.track_shortcuts = track_shortcuts

        # Statistics tracking
        self._key_counts = Counter()
        self._app_key_counts = defaultdict(Counter)
        self._shortcuts = []
        self._last_aggregate = datetime.now()
        self._session_start: Optional[datetime] = None

        # Lazy import
        self._listener = None

    def _setup_listener(self):
        """Setup keyboard listener (lazy loading)."""
        if self._listener is not None:
            return

        try:
            from pynput import keyboard

            def on_press(key):
                """Handle key press event."""
                if not self._running:
                    return

                # Update statistics
                self._key_counts['total'] += 1

                # Track modifiers for shortcuts
                if self.track_shortcuts:
                    try:
                        # Check if it's a special key combination
                        if hasattr(key, 'char') and keyboard.Controller().ctrl:
                            self._shortcuts.append(
                                {
                                    'key': key.char,
                                    'modifiers': ['ctrl'],
                                    'timestamp': datetime.now(),
                                }
                            )
                    except Exception:
                        pass

                # Get active application (platform-specific)
                try:
                    app_name = self._get_active_application()
                    if app_name:
                        self._app_key_counts[app_name]['keys'] += 1
                except Exception:
                    pass

            self._listener = keyboard.Listener(on_press=on_press)

        except ImportError:
            raise ImportError(
                "pynput is required for keyboard observer. "
                "Install it with: pip install pynput"
            )

    def _get_active_application(self) -> Optional[str]:
        """Get the name of the currently active application.

        This is platform-specific.
        """
        import platform
        import subprocess

        system = platform.system()

        try:
            if system == 'Darwin':  # macOS
                script = 'tell application "System Events" to get name of first application process whose frontmost is true'
                result = subprocess.run(
                    ['osascript', '-e', script],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout.strip()

            elif system == 'Linux':
                # Try using xdotool (if available)
                result = subprocess.run(
                    ['xdotool', 'getactivewindow', 'getwindowname'],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout.strip()

            elif system == 'Windows':
                # Use win32gui (if available)
                try:
                    import win32gui

                    window = win32gui.GetForegroundWindow()
                    return win32gui.GetWindowText(window)
                except ImportError:
                    pass

        except Exception:
            pass

        return None

    def start(self):
        """Start observing keyboard activity."""
        super().start()
        self._setup_listener()
        if self._listener:
            self._listener.start()
        self._session_start = datetime.now()

    def stop(self):
        """Stop observing keyboard activity."""
        super().stop()
        if self._listener:
            self._listener.stop()

    def observe(self) -> Iterator[Observation]:
        """Observe keyboard activity and yield aggregated observations."""
        import time

        self.start()

        while self._running:
            now = datetime.now()

            # Check if it's time to aggregate
            if (now - self._last_aggregate).total_seconds() >= self.aggregate_interval:
                # Create observation from aggregated stats
                if self._key_counts['total'] > 0:
                    obs = self._create_aggregate_observation()
                    if obs:
                        yield obs

                # Reset counters
                self._reset_counters()
                self._last_aggregate = now

            # Sleep briefly
            time.sleep(1.0)

    def _create_aggregate_observation(self) -> Optional[Observation]:
        """Create an observation from aggregated statistics."""
        if self._key_counts['total'] == 0:
            return None

        data = {
            'total_keystrokes': self._key_counts['total'],
            'duration_seconds': self.aggregate_interval,
            'keystrokes_per_minute': (
                self._key_counts['total'] / self.aggregate_interval * 60
            ),
        }

        # Add application breakdown
        if self._app_key_counts:
            data['by_application'] = {
                app: dict(counts) for app, counts in self._app_key_counts.items()
            }

        # Add shortcuts if tracked
        if self.track_shortcuts and self._shortcuts:
            data['shortcuts_used'] = len(self._shortcuts)
            # Don't log the actual shortcuts for privacy, just count

        return self.create_observation(
            event_type='keyboard_activity',
            data=data,
            metadata={
                'session_start': self._session_start.isoformat()
                if self._session_start
                else None,
            },
            tags=['keyboard', 'typing', 'productivity'],
        )

    def _reset_counters(self):
        """Reset statistical counters."""
        self._key_counts.clear()
        self._app_key_counts.clear()
        self._shortcuts.clear()
