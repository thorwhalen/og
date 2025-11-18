"""Application usage observer for OG (Own Ghost).

This observer tracks which applications are being used and for how long.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from og.base import Observation, PollingObserver


class AppUsageObserver(PollingObserver):
    """Observer for application usage.

    Tracks:
    - Active application
    - Time spent in each application
    - Application switches
    """

    def __init__(
        self,
        name: str = 'apps',
        poll_interval: float = 10.0,  # Check every 10 seconds
        enabled: bool = True,
        **config
    ):
        """Initialize app usage observer.

        Args:
            name: Observer name
            poll_interval: How often to check active app (seconds)
            enabled: Whether observer is enabled
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)
        self._current_app: Optional[str] = None
        self._app_start_time: Optional[datetime] = None
        self._app_durations = defaultdict(float)

    def poll(self) -> list[Observation]:
        """Poll for active application."""
        observations = []

        try:
            active_app = self._get_active_application()

            if active_app != self._current_app:
                # Application switched
                now = datetime.now()

                # Record time for previous app
                if self._current_app and self._app_start_time:
                    duration = (now - self._app_start_time).total_seconds()
                    self._app_durations[self._current_app] += duration

                    obs = self.create_observation(
                        event_type='app_usage',
                        data={
                            'application': self._current_app,
                            'duration_seconds': duration,
                            'started_at': self._app_start_time.isoformat(),
                            'ended_at': now.isoformat(),
                        },
                        tags=['app', 'usage', 'focus'],
                    )
                    observations.append(obs)

                # Switch to new app
                if active_app:
                    obs = self.create_observation(
                        event_type='app_switch',
                        data={
                            'from_app': self._current_app,
                            'to_app': active_app,
                        },
                        tags=['app', 'switch'],
                    )
                    observations.append(obs)

                self._current_app = active_app
                self._app_start_time = now

        except Exception as e:
            print(f"Error polling active application: {e}")

        return observations

    def _get_active_application(self) -> Optional[str]:
        """Get the currently active application.

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
                # Try using xdotool
                result = subprocess.run(
                    ['xdotool', 'getactivewindow', 'getwindowname'],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    # Extract application name from window title
                    window_title = result.stdout.strip()
                    # Try to get process name
                    result2 = subprocess.run(
                        ['xdotool', 'getactivewindow', 'getwindowpid'],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result2.returncode == 0:
                        pid = result2.stdout.strip()
                        result3 = subprocess.run(
                            ['ps', '-p', pid, '-o', 'comm='],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if result3.returncode == 0:
                            return result3.stdout.strip()
                    return window_title

            elif system == 'Windows':
                try:
                    import win32gui
                    import win32process
                    import psutil

                    window = win32gui.GetForegroundWindow()
                    _, pid = win32process.GetWindowThreadProcessId(window)
                    process = psutil.Process(pid)
                    return process.name()
                except ImportError:
                    pass

        except Exception as e:
            print(f"Error getting active application: {e}")

        return None
