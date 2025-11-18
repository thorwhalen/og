"""Pattern detection and alert system for OG.

This module detects behavioral patterns and triggers alerts
to help users stay focused and aware of their activity.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from og.base import Observation


@dataclass
class Pattern:
    """A behavioral pattern to detect."""

    name: str
    description: str
    condition: Callable[[list[Observation]], bool]
    alert_message: str
    cooldown: int = 3600  # Don't alert more than once per hour by default
    severity: str = 'info'  # info, warning, critical
    enabled: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class Alert:
    """An alert triggered by a pattern."""

    pattern_name: str
    message: str
    timestamp: datetime
    severity: str
    data: dict = field(default_factory=dict)
    acknowledged: bool = False


class PatternDetector:
    """Detects patterns in observations and triggers alerts."""

    def __init__(
        self,
        window_size: int = 3600,  # Look at last hour by default
        check_interval: int = 60,  # Check every minute
    ):
        """Initialize pattern detector.

        Args:
            window_size: Time window for pattern detection (seconds)
            check_interval: How often to check for patterns (seconds)
        """
        self.window_size = window_size
        self.check_interval = check_interval
        self.patterns: dict[str, Pattern] = {}
        self.alerts: list[Alert] = []
        self.last_alert_time: dict[str, datetime] = {}

        # Observation buffer for pattern detection
        self.observation_buffer: deque[Observation] = deque(maxlen=1000)

        # Statistics tracking
        self.stats = defaultdict(int)

        # Register default patterns
        self._register_default_patterns()

    def add_pattern(self, pattern: Pattern) -> None:
        """Register a pattern for detection.

        Args:
            pattern: Pattern to register
        """
        self.patterns[pattern.name] = pattern

    def remove_pattern(self, name: str) -> None:
        """Remove a pattern.

        Args:
            name: Pattern name
        """
        if name in self.patterns:
            del self.patterns[name]

    def enable_pattern(self, name: str) -> None:
        """Enable a pattern.

        Args:
            name: Pattern name
        """
        if name in self.patterns:
            self.patterns[name].enabled = True

    def disable_pattern(self, name: str) -> None:
        """Disable a pattern.

        Args:
            name: Pattern name
        """
        if name in self.patterns:
            self.patterns[name].enabled = False

    def add_observation(self, obs: Observation) -> list[Alert]:
        """Add an observation and check for patterns.

        Args:
            obs: New observation

        Returns:
            List of alerts triggered
        """
        self.observation_buffer.append(obs)
        return self.check_patterns()

    def check_patterns(self) -> list[Alert]:
        """Check all enabled patterns and trigger alerts.

        Returns:
            List of new alerts triggered
        """
        new_alerts = []

        # Get recent observations within window
        cutoff_time = datetime.now() - timedelta(seconds=self.window_size)
        recent_obs = [obs for obs in self.observation_buffer if obs.timestamp >= cutoff_time]

        # Check each pattern
        for pattern in self.patterns.values():
            if not pattern.enabled:
                continue

            # Check cooldown
            last_alert = self.last_alert_time.get(pattern.name)
            if last_alert:
                if (datetime.now() - last_alert).total_seconds() < pattern.cooldown:
                    continue

            # Check condition
            try:
                if pattern.condition(recent_obs):
                    alert = self._create_alert(pattern, recent_obs)
                    new_alerts.append(alert)
                    self.alerts.append(alert)
                    self.last_alert_time[pattern.name] = datetime.now()
                    self.stats[f'alert_{pattern.name}'] += 1
            except Exception as e:
                print(f"Error checking pattern {pattern.name}: {e}")

        return new_alerts

    def _create_alert(self, pattern: Pattern, observations: list[Observation]) -> Alert:
        """Create an alert from a pattern match.

        Args:
            pattern: Matched pattern
            observations: Observations that matched

        Returns:
            Alert instance
        """
        # Format message with observation data
        message = pattern.alert_message

        # Extract context from observations
        data = {
            'observation_count': len(observations),
            'time_window': self.window_size,
            'pattern_name': pattern.name,
        }

        return Alert(
            pattern_name=pattern.name,
            message=message,
            timestamp=datetime.now(),
            severity=pattern.severity,
            data=data,
        )

    def get_alerts(
        self,
        since: Optional[datetime] = None,
        severity: Optional[str] = None,
        acknowledged: Optional[bool] = None,
    ) -> list[Alert]:
        """Get alerts with optional filters.

        Args:
            since: Get alerts since this time
            severity: Filter by severity
            acknowledged: Filter by acknowledgement status

        Returns:
            Filtered alerts
        """
        filtered = self.alerts

        if since:
            filtered = [a for a in filtered if a.timestamp >= since]

        if severity:
            filtered = [a for a in filtered if a.severity == severity]

        if acknowledged is not None:
            filtered = [a for a in filtered if a.acknowledged == acknowledged]

        return filtered

    def acknowledge_alert(self, alert: Alert) -> None:
        """Acknowledge an alert.

        Args:
            alert: Alert to acknowledge
        """
        alert.acknowledged = True

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
        self.last_alert_time.clear()

    def _register_default_patterns(self) -> None:
        """Register default patterns."""

        # Deep focus broken pattern
        def deep_focus_broken(obs: list[Observation]) -> bool:
            """Detect frequent app switching (broken focus)."""
            app_switches = [o for o in obs if o.event_type == 'app_switch']
            if len(app_switches) >= 10:  # 10+ switches in window
                return True
            return False

        self.add_pattern(
            Pattern(
                name='deep_focus_broken',
                description='Frequent application switching detected',
                condition=deep_focus_broken,
                alert_message='You switched apps frequently ({count} times). Try to focus on one task.',
                severity='warning',
                cooldown=1800,  # 30 minutes
            )
        )

        # Excessive browsing pattern
        def excessive_browsing(obs: list[Observation]) -> bool:
            """Detect excessive web browsing."""
            visits = [o for o in obs if o.event_type == 'browser_visit']
            if len(visits) >= 30:  # 30+ page visits in window
                return True
            return False

        self.add_pattern(
            Pattern(
                name='excessive_browsing',
                description='Excessive web browsing detected',
                condition=excessive_browsing,
                alert_message='You visited {count} web pages. Consider taking a break or refocusing.',
                severity='info',
                cooldown=3600,
            )
        )

        # Productivity milestone pattern
        def productivity_milestone(obs: list[Observation]) -> bool:
            """Detect productivity milestones (commits, completed tasks)."""
            commits = [o for o in obs if o.event_type in ['git_commit', 'github_push']]
            if len(commits) >= 5:
                return True
            return False

        self.add_pattern(
            Pattern(
                name='productivity_milestone',
                description='Productivity milestone reached',
                condition=productivity_milestone,
                alert_message='Great work! You made {count} commits recently.',
                severity='info',
                cooldown=7200,  # 2 hours
            )
        )

        # Late night work pattern
        def late_night_work(obs: list[Observation]) -> bool:
            """Detect late-night work sessions."""
            now = datetime.now()
            if now.hour >= 23 or now.hour <= 5:
                # Check if there's recent activity
                recent_activity = [
                    o
                    for o in obs
                    if (now - o.timestamp).total_seconds() < 300
                ]  # Last 5 minutes
                if len(recent_activity) >= 3:
                    return True
            return False

        self.add_pattern(
            Pattern(
                name='late_night_work',
                description='Late night work session detected',
                condition=late_night_work,
                alert_message="It's late! Consider taking a break and getting rest.",
                severity='warning',
                cooldown=3600,
            )
        )

        # Idle detection pattern
        def idle_detected(obs: list[Observation]) -> bool:
            """Detect extended idle periods."""
            if not obs:
                return False

            now = datetime.now()
            last_activity = max(o.timestamp for o in obs)
            idle_time = (now - last_activity).total_seconds()

            # If idle for more than 30 minutes
            if idle_time >= 1800:
                return True
            return False

        self.add_pattern(
            Pattern(
                name='idle_detected',
                description='Extended idle period detected',
                condition=idle_detected,
                alert_message='No activity detected for a while. Taking a break?',
                severity='info',
                cooldown=3600,
                enabled=False,  # Disabled by default
            )
        )

        # Context switch detection
        def context_switch_detected(obs: list[Observation]) -> bool:
            """Detect major context switches between projects."""
            # Look for rapid changes in repos/projects
            recent = obs[-10:] if len(obs) >= 10 else obs

            repos = set()
            for o in recent:
                if o.event_type in ['git_commit', 'github_push']:
                    repo = o.data.get('repo')
                    if repo:
                        repos.add(repo)

            # If working on 3+ different repos in short time
            if len(repos) >= 3:
                return True

            return False

        self.add_pattern(
            Pattern(
                name='context_switch',
                description='Major context switch detected',
                condition=context_switch_detected,
                alert_message='You switched between multiple projects. Consider batching similar work.',
                severity='info',
                cooldown=3600,
            )
        )

    def get_pattern_stats(self) -> dict:
        """Get statistics about pattern detection.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_patterns': len(self.patterns),
            'enabled_patterns': sum(1 for p in self.patterns.values() if p.enabled),
            'total_alerts': len(self.alerts),
            'unacknowledged_alerts': sum(1 for a in self.alerts if not a.acknowledged),
            'alert_counts': dict(self.stats),
            'patterns': {
                name: {
                    'enabled': p.enabled,
                    'description': p.description,
                    'severity': p.severity,
                }
                for name, p in self.patterns.items()
            },
        }


class AlertHandler:
    """Handler for alerts with various notification methods."""

    def __init__(self):
        """Initialize alert handler."""
        self.handlers: list[Callable[[Alert], None]] = []

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler.

        Args:
            handler: Function that takes an Alert
        """
        self.handlers.append(handler)

    def handle(self, alert: Alert) -> None:
        """Handle an alert by calling all handlers.

        Args:
            alert: Alert to handle
        """
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")

    @staticmethod
    def console_handler(alert: Alert) -> None:
        """Print alert to console.

        Args:
            alert: Alert to print
        """
        severity_icons = {
            'info': 'ℹ️ ',
            'warning': '⚠️ ',
            'critical': '🚨',
        }
        icon = severity_icons.get(alert.severity, '')
        print(f"\n{icon} ALERT [{alert.severity.upper()}]: {alert.message}")
        print(f"   Pattern: {alert.pattern_name}")
        print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")

    @staticmethod
    def log_handler(alert: Alert, log_file: str = 'og_alerts.log') -> None:
        """Log alert to file.

        Args:
            alert: Alert to log
            log_file: Path to log file
        """
        with open(log_file, 'a') as f:
            f.write(
                f"[{alert.timestamp.isoformat()}] {alert.severity}: "
                f"{alert.pattern_name} - {alert.message}\n"
            )

    @staticmethod
    def desktop_notification_handler(alert: Alert) -> None:
        """Show desktop notification (platform-specific).

        Args:
            alert: Alert to show
        """
        try:
            import platform

            system = platform.system()

            if system == 'Darwin':  # macOS
                import subprocess

                subprocess.run(
                    [
                        'osascript',
                        '-e',
                        f'display notification "{alert.message}" '
                        f'with title "OG Alert" subtitle "{alert.pattern_name}"',
                    ]
                )
            elif system == 'Linux':
                import subprocess

                subprocess.run(['notify-send', 'OG Alert', alert.message])
            elif system == 'Windows':
                try:
                    from win10toast import ToastNotifier

                    toaster = ToastNotifier()
                    toaster.show_toast(
                        'OG Alert', alert.message, duration=10, threaded=True
                    )
                except ImportError:
                    pass
        except Exception as e:
            print(f"Could not show desktop notification: {e}")
