"""Focus mode with active intervention.

Blocks distractions, silences notifications, protects calendar, and
actively helps maintain deep work sessions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import subprocess
import platform


@dataclass
class FocusSession:
    """A focus mode session."""

    start: datetime
    end: Optional[datetime] = None
    duration_minutes: int = 60
    blocked_sites: List[str] = None
    blocked_apps: List[str] = None
    interruptions: int = 0

    def __post_init__(self):
        if self.blocked_sites is None:
            self.blocked_sites = []
        if self.blocked_apps is None:
            self.blocked_apps = []


class FocusMode:
    """Active focus mode intervention."""

    DEFAULT_BLOCKED_SITES = [
        'facebook.com',
        'twitter.com',
        'reddit.com',
        'youtube.com',
        'news.ycombinator.com',
    ]

    DEFAULT_BLOCKED_APPS = [
        'Slack',
        'Discord',
        'Messages',
    ]

    def __init__(self, og_instance=None):
        self.og = og_instance
        self._active = False
        self._current_session: Optional[FocusSession] = None

    def enable(self, duration_minutes: int = 60):
        """Enable focus mode."""
        if self._active:
            return

        self._active = True
        self._current_session = FocusSession(
            start=datetime.now(),
            duration_minutes=duration_minutes,
            blocked_sites=self.DEFAULT_BLOCKED_SITES.copy(),
            blocked_apps=self.DEFAULT_BLOCKED_APPS.copy(),
        )

        # Block websites
        self._block_websites(self._current_session.blocked_sites)

        # Silence notifications
        self._silence_notifications()

        # Update calendar status
        if self.og and hasattr(self.og, 'calendar_observer'):
            self._protect_calendar(duration_minutes)

        print(f"Focus mode enabled for {duration_minutes} minutes")

    def disable(self):
        """Disable focus mode."""
        if not self._active:
            return

        self._active = False

        # Unblock websites
        if self._current_session:
            self._unblock_websites(self._current_session.blocked_sites)
            self._current_session.end = datetime.now()

        # Restore notifications
        self._restore_notifications()

        print("Focus mode disabled")

    def _block_websites(self, sites: List[str]):
        """Block distracting websites."""
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            # Modify /etc/hosts
            hosts_path = '/etc/hosts'
            try:
                with open(hosts_path, 'a') as f:
                    f.write('\n# OG Focus Mode\n')
                    for site in sites:
                        f.write(f'127.0.0.1 {site}\n')
                        f.write(f'127.0.0.1 www.{site}\n')
            except PermissionError:
                print("Warning: Cannot modify hosts file. Run with sudo.")

    def _unblock_websites(self, sites: List[str]):
        """Unblock websites."""
        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            hosts_path = '/etc/hosts'
            try:
                with open(hosts_path, 'r') as f:
                    lines = f.readlines()

                with open(hosts_path, 'w') as f:
                    in_og_section = False
                    for line in lines:
                        if '# OG Focus Mode' in line:
                            in_og_section = True
                            continue
                        if in_og_section and line.strip() and not line.startswith('127.0.0.1'):
                            in_og_section = False
                        if not in_og_section:
                            f.write(line)
            except PermissionError:
                print("Warning: Cannot modify hosts file. Run with sudo.")

    def _silence_notifications(self):
        """Silence system notifications."""
        system = platform.system()

        if system == 'Darwin':  # macOS
            # Enable Do Not Disturb
            subprocess.run([
                'defaults', 'write', 'com.apple.ncprefs',
                'dnd_prefs', '-dict', 'dnd Enabled', '-bool', 'true'
            ])
        elif system == 'Linux':
            # Depends on desktop environment
            pass

    def _restore_notifications(self):
        """Restore system notifications."""
        system = platform.system()

        if system == 'Darwin':  # macOS
            subprocess.run([
                'defaults', 'write', 'com.apple.ncprefs',
                'dnd_prefs', '-dict', 'dnd_enabled', '-bool', 'false'
            ])

    def _protect_calendar(self, duration_minutes: int):
        """Protect calendar from new meetings."""
        # Would integrate with calendar API to:
        # 1. Mark as busy
        # 2. Auto-decline meeting invites
        # 3. Set status message
        pass

    def add_blocked_site(self, site: str):
        """Add a site to block list."""
        if self._current_session:
            self._current_session.blocked_sites.append(site)
            if self._active:
                self._block_websites([site])

    def add_blocked_app(self, app: str):
        """Add an app to block list."""
        if self._current_session:
            self._current_session.blocked_apps.append(app)
