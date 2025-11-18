"""Proactive insights and smart notifications.

This module provides anomaly detection, context-aware reminders,
and proactive suggestions based on patterns and deviations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import statistics


@dataclass
class Insight:
    """A proactive insight or notification."""

    type: str  # 'anomaly', 'reminder', 'suggestion', 'warning'
    title: str
    message: str
    timestamp: datetime
    priority: str = 'normal'  # 'low', 'normal', 'high', 'urgent'
    context: Optional[str] = None
    action: Optional[str] = None  # Suggested action
    confidence: float = 1.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_notification(self, format: str = 'simple') -> str:
        """Format as a notification."""
        if format == 'rich':
            priority_emoji = {
                'low': 'ℹ️',
                'normal': '💡',
                'high': '⚠️',
                'urgent': '🚨',
            }
            emoji = priority_emoji.get(self.priority, '💡')
            return f"{emoji} {self.title}\n{self.message}" + (
                f"\n→ {self.action}" if self.action else ""
            )
        else:
            return f"[{self.type.upper()}] {self.title}: {self.message}"


@dataclass
class ProactiveReport:
    """Report of proactive insights."""

    period_start: datetime
    period_end: datetime
    insights: List[Insight]
    anomalies: List[Insight]
    reminders: List[Insight]
    suggestions: List[Insight]

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = ["# Proactive Insights Report\n"]

        if self.anomalies:
            lines.append("## 🔔 Anomalies Detected\n")
            for insight in self.anomalies:
                lines.append(f"### {insight.title}")
                lines.append(f"{insight.message}\n")
                if insight.action:
                    lines.append(f"**Suggested action:** {insight.action}\n")

        if self.reminders:
            lines.append("## 📝 Reminders\n")
            for insight in self.reminders:
                lines.append(f"- {insight.message}")
            lines.append("")

        if self.suggestions:
            lines.append("## 💡 Suggestions\n")
            for insight in self.suggestions:
                lines.append(f"- {insight.message}")
            lines.append("")

        return "\n".join(lines)


class ProactiveInsightEngine:
    """Generates proactive insights and smart notifications."""

    def __init__(self, og_instance=None):
        """Initialize the insight engine.

        Args:
            og_instance: Optional OG instance for accessing observations
        """
        self.og = og_instance
        self._patterns = {}  # Historical patterns
        self._baseline_metrics = {}  # Baseline behavior metrics

    def analyze(
        self,
        check_anomalies: bool = True,
        check_reminders: bool = True,
        check_suggestions: bool = True,
    ) -> ProactiveReport:
        """Generate proactive insights.

        Args:
            check_anomalies: Check for anomalous behavior
            check_reminders: Generate context-aware reminders
            check_suggestions: Generate optimization suggestions

        Returns:
            ProactiveReport with insights
        """
        now = datetime.now()
        insights = []

        if check_anomalies:
            insights.extend(self._detect_anomalies(now))

        if check_reminders:
            insights.extend(self._generate_reminders(now))

        if check_suggestions:
            insights.extend(self._generate_suggestions(now))

        # Categorize insights
        anomalies = [i for i in insights if i.type == 'anomaly']
        reminders = [i for i in insights if i.type == 'reminder']
        suggestions = [i for i in insights if i.type == 'suggestion']

        return ProactiveReport(
            period_start=now - timedelta(days=1),
            period_end=now,
            insights=insights,
            anomalies=anomalies,
            reminders=reminders,
            suggestions=suggestions,
        )

    def _detect_anomalies(self, now: datetime) -> List[Insight]:
        """Detect anomalous behavior patterns."""
        if not self.og:
            return []

        insights = []

        # Build baseline if not exists
        if not self._baseline_metrics:
            self._build_baseline()

        # Check for timing anomalies
        insights.extend(self._check_timing_anomalies(now))

        # Check for activity anomalies
        insights.extend(self._check_activity_anomalies(now))

        # Check for productivity anomalies
        insights.extend(self._check_productivity_anomalies(now))

        return insights

    def _build_baseline(self):
        """Build baseline behavior metrics from historical data."""
        if not self.og:
            return

        # Get last 30 days of data
        end = datetime.now()
        start = end - timedelta(days=30)
        observations = self.og.recent_activity(start_date=start, end_date=end)

        # Calculate typical commit time
        commits = [o for o in observations if o.event_type == 'git_commit']
        if commits:
            commit_hours = [o.timestamp.hour for o in commits]
            self._baseline_metrics['avg_commit_hour'] = statistics.mean(commit_hours)
            self._baseline_metrics['std_commit_hour'] = statistics.stdev(commit_hours) if len(commit_hours) > 1 else 0

        # Calculate typical start time
        daily_starts = defaultdict(list)
        for obs in observations:
            date = obs.timestamp.date()
            daily_starts[date].append(obs.timestamp)

        start_hours = []
        for date, times in daily_starts.items():
            if times:
                first_activity = min(times)
                start_hours.append(first_activity.hour + first_activity.minute / 60)

        if start_hours:
            self._baseline_metrics['avg_start_hour'] = statistics.mean(start_hours)
            self._baseline_metrics['std_start_hour'] = statistics.stdev(start_hours) if len(start_hours) > 1 else 0

        # Calculate typical activity count
        daily_activity = [len(times) for times in daily_starts.values()]
        if daily_activity:
            self._baseline_metrics['avg_daily_activity'] = statistics.mean(daily_activity)
            self._baseline_metrics['std_daily_activity'] = statistics.stdev(daily_activity) if len(daily_activity) > 1 else 0

    def _check_timing_anomalies(self, now: datetime) -> List[Insight]:
        """Check for timing-related anomalies."""
        insights = []

        if not self.og or not self._baseline_metrics:
            return insights

        # Check if no commit by usual time
        avg_commit_hour = self._baseline_metrics.get('avg_commit_hour')
        if avg_commit_hour:
            current_hour = now.hour + now.minute / 60

            if current_hour > avg_commit_hour + 2:  # 2 hours past usual
                # Check if any commits today
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                commits = self.og.recent_activity(
                    start_date=today_start,
                    end_date=now,
                    event_type='git_commit'
                )

                if not commits:
                    insights.append(Insight(
                        type='anomaly',
                        title='Delayed Commit Pattern',
                        message=f"You usually commit by {int(avg_commit_hour):02d}:00 but haven't yet today.",
                        timestamp=now,
                        priority='normal',
                        action="Consider committing your work in progress",
                        confidence=0.7,
                    ))

        # Check for unusual start time
        avg_start = self._baseline_metrics.get('avg_start_hour')
        if avg_start and now.hour > 8:  # Only check after 8am
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_activity = self.og.recent_activity(
                start_date=today_start,
                end_date=now
            )

            if today_activity:
                first_activity_hour = min(o.timestamp for o in today_activity).hour
                if abs(first_activity_hour - avg_start) > 2:
                    insights.append(Insight(
                        type='anomaly',
                        title='Unusual Start Time',
                        message=f"You started {'earlier' if first_activity_hour < avg_start else 'later'} than usual today.",
                        timestamp=now,
                        priority='low',
                        confidence=0.6,
                    ))

        return insights

    def _check_activity_anomalies(self, now: datetime) -> List[Insight]:
        """Check for activity level anomalies."""
        insights = []

        if not self.og or not self._baseline_metrics:
            return insights

        # Check if activity is unusually low
        avg_daily = self._baseline_metrics.get('avg_daily_activity')
        std_daily = self._baseline_metrics.get('std_daily_activity', 0)

        if avg_daily:
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            today_activity = self.og.recent_activity(
                start_date=today_start,
                end_date=now
            )

            # Only check after half the typical workday
            if now.hour >= 14:
                if len(today_activity) < avg_daily - 2 * std_daily:
                    insights.append(Insight(
                        type='anomaly',
                        title='Low Activity Day',
                        message=f"Activity is unusually low today ({len(today_activity)} vs typical {avg_daily:.0f}).",
                        timestamp=now,
                        priority='normal',
                        confidence=0.7,
                    ))

        return insights

    def _check_productivity_anomalies(self, now: datetime) -> List[Insight]:
        """Check for productivity-related anomalies."""
        insights = []

        if not self.og:
            return insights

        # Check for working late (after typical hours)
        if now.hour >= 22:  # After 10 PM
            today_start = now.replace(hour=20, minute=0, second=0, microsecond=0)
            recent_activity = self.og.recent_activity(
                start_date=today_start,
                end_date=now
            )

            if len(recent_activity) > 10:
                insights.append(Insight(
                    type='warning',
                    title='Late Night Work',
                    message="You're working late. Consider taking a break to avoid burnout.",
                    timestamp=now,
                    priority='high',
                    action="Save your work and continue tomorrow",
                    confidence=1.0,
                ))

        return insights

    def _generate_reminders(self, now: datetime) -> List[Insight]:
        """Generate context-aware reminders."""
        insights = []

        if not self.og:
            return insights

        # Remind about work in progress
        insights.extend(self._remind_work_in_progress(now))

        # Remind about uncommitted changes
        insights.extend(self._remind_uncommitted_changes(now))

        # Remind about unanswered messages
        insights.extend(self._remind_pending_items(now))

        return insights

    def _remind_work_in_progress(self, now: datetime) -> List[Insight]:
        """Remind about work in progress from yesterday."""
        insights = []

        if not self.og:
            return insights

        # Get yesterday's last activity
        yesterday = now - timedelta(days=1)
        yesterday_start = yesterday.replace(hour=16, minute=0, second=0, microsecond=0)
        yesterday_end = yesterday.replace(hour=23, minute=59, second=59)

        recent_work = self.og.recent_activity(
            start_date=yesterday_start,
            end_date=yesterday_end,
            event_type='file_modify'
        )

        if recent_work:
            # Get the files/projects worked on
            files = set()
            for obs in recent_work[-5:]:  # Last 5 files
                path = obs.data.get('path', '')
                if path:
                    files.add(path)

            if files:
                insights.append(Insight(
                    type='reminder',
                    title='Resume Yesterday\'s Work',
                    message=f"You were working on {list(files)[0]} yesterday. Ready to continue?",
                    timestamp=now,
                    priority='normal',
                    action="Open recent files",
                    confidence=0.8,
                ))

        # Check if there's an active context
        if hasattr(self.og, 'context_manager') and self.og.context_manager:
            current = self.og.context_manager.current_context
            if current:
                insights.append(Insight(
                    type='reminder',
                    title=f'Continue {current.name}',
                    message=f"You're currently working on {current.name}.",
                    timestamp=now,
                    priority='normal',
                    context=current.name,
                    confidence=0.9,
                ))

        return insights

    def _remind_uncommitted_changes(self, now: datetime) -> List[Insight]:
        """Remind about uncommitted changes."""
        insights = []

        if not self.og:
            return insights

        # Look for recent file modifications without subsequent commits
        last_hour = now - timedelta(hours=1)
        modifications = self.og.recent_activity(
            start_date=last_hour,
            end_date=now,
            event_type='file_modify'
        )

        commits = self.og.recent_activity(
            start_date=last_hour,
            end_date=now,
            event_type='git_commit'
        )

        if len(modifications) > 5 and len(commits) == 0:
            insights.append(Insight(
                type='reminder',
                title='Uncommitted Changes',
                message=f"You've modified {len(modifications)} files but haven't committed in the last hour.",
                timestamp=now,
                priority='normal',
                action="Consider committing your work",
                confidence=0.8,
            ))

        return insights

    def _remind_pending_items(self, now: datetime) -> List[Insight]:
        """Remind about pending items."""
        insights = []

        # This would integrate with task management systems
        # For now, placeholder

        return insights

    def _generate_suggestions(self, now: datetime) -> List[Insight]:
        """Generate optimization suggestions."""
        insights = []

        if not self.og:
            return insights

        # Suggest optimal work timing
        insights.extend(self._suggest_optimal_timing(now))

        # Suggest focus blocks
        insights.extend(self._suggest_focus_blocks(now))

        return insights

    def _suggest_optimal_timing(self, now: datetime) -> List[Insight]:
        """Suggest optimal timing for tasks."""
        insights = []

        if not self.og:
            return insights

        # Analyze historical productivity patterns
        if hasattr(self.og, 'insight_engine') and self.og.insight_engine:
            patterns = self.og.insight_engine.detect_productivity_patterns()

            peak_hours = patterns.get('peak_hours', [])
            if peak_hours and now.hour in peak_hours:
                insights.append(Insight(
                    type='suggestion',
                    title='Peak Productivity Time',
                    message=f"This is one of your peak productivity hours ({now.hour}:00). Great time for deep work!",
                    timestamp=now,
                    priority='high',
                    action="Block this time for focused work",
                    confidence=0.9,
                ))

        return insights

    def _suggest_focus_blocks(self, now: datetime) -> List[Insight]:
        """Suggest focus time blocks."""
        insights = []

        if not self.og:
            return insights

        # Check if switching too much
        last_hour = now - timedelta(hours=1)
        app_switches = self.og.recent_activity(
            start_date=last_hour,
            end_date=now,
            event_type='app_switch'
        )

        if len(app_switches) > 15:
            insights.append(Insight(
                type='suggestion',
                title='High Context Switching',
                message=f"You've switched apps {len(app_switches)} times in the last hour.",
                timestamp=now,
                priority='high',
                action="Consider enabling focus mode to reduce distractions",
                confidence=0.8,
            ))

        return insights

    def subscribe(
        self,
        insight_type: str,
        callback: Callable[[Insight], None],
        priority_filter: Optional[List[str]] = None,
    ):
        """Subscribe to insights.

        Args:
            insight_type: Type of insight to subscribe to
            callback: Function to call when insight is generated
            priority_filter: Optional list of priorities to filter by
        """
        # This would be implemented with a pub/sub pattern
        # For now, placeholder
        pass

    def get_insights_for_time(
        self, target_time: datetime, lookahead_hours: int = 4
    ) -> List[Insight]:
        """Get insights relevant for a specific time.

        Args:
            target_time: Time to get insights for
            lookahead_hours: Hours to look ahead

        Returns:
            List of relevant insights
        """
        insights = []

        # Check calendar for upcoming meetings
        if self.og:
            upcoming = self.og.recent_activity(
                start_date=target_time,
                end_date=target_time + timedelta(hours=lookahead_hours),
                event_type='calendar_event'
            )

            if upcoming:
                next_meeting = min(upcoming, key=lambda x: x.timestamp)
                time_until = (next_meeting.timestamp - target_time).total_seconds() / 60

                if time_until < 30:
                    insights.append(Insight(
                        type='reminder',
                        title='Upcoming Meeting',
                        message=f"Meeting in {int(time_until)} minutes: {next_meeting.data.get('title')}",
                        timestamp=target_time,
                        priority='high',
                        confidence=1.0,
                    ))

        return insights
