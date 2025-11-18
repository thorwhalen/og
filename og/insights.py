"""Advanced AI-powered insights for OG.

This module provides sophisticated analysis of activity patterns,
productivity insights, and personalized recommendations.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Optional

from og.base import Observation
from og.storage import ObservationMall


class InsightEngine:
    """Advanced AI-powered insights and analytics."""

    def __init__(self, mall: ObservationMall, ai_agent=None):
        """Initialize insight engine.

        Args:
            mall: ObservationMall for querying observations
            ai_agent: Optional OGAgent for AI-powered insights
        """
        self.mall = mall
        self.ai_agent = ai_agent

    def detect_productivity_patterns(self, days: int = 30) -> dict:
        """Analyze when you're most productive.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with productivity patterns
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        observations = self.mall.query_by_timerange(start_time, end_time)

        # Analyze by time of day
        by_hour = defaultdict(int)
        by_day_of_week = defaultdict(int)

        # Count productive activities
        productive_types = {
            'git_commit',
            'github_push',
            'keyboard_activity',
            'terminal_command',
        }

        for obs in observations:
            if obs.event_type in productive_types:
                hour = obs.timestamp.hour
                day_of_week = obs.timestamp.strftime('%A')

                by_hour[hour] += 1
                by_day_of_week[day_of_week] += 1

        # Find peak hours and days
        if by_hour:
            peak_hour = max(by_hour.items(), key=lambda x: x[1])
            peak_day = max(by_day_of_week.items(), key=lambda x: x[1])
        else:
            peak_hour = (None, 0)
            peak_day = (None, 0)

        return {
            'analysis_period_days': days,
            'total_productive_observations': sum(by_hour.values()),
            'by_hour': dict(by_hour),
            'by_day_of_week': dict(by_day_of_week),
            'peak_hour': peak_hour[0],
            'peak_day': peak_day[0],
            'recommendation': self._generate_productivity_recommendation(
                peak_hour, peak_day
            ),
        }

    def identify_deep_work_sessions(
        self, days: int = 7, min_duration_minutes: int = 30
    ) -> list[dict]:
        """Identify periods of sustained focus (deep work).

        Args:
            days: Number of days to analyze
            min_duration_minutes: Minimum session duration

        Returns:
            List of deep work sessions
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        observations = self.mall.query_by_timerange(start_time, end_time)

        # Sort by time
        observations.sort(key=lambda x: x.timestamp)

        # Identify continuous work sessions
        sessions = []
        current_session = None

        focus_types = {'keyboard_activity', 'git_commit', 'terminal_command'}

        for obs in observations:
            if obs.event_type in focus_types:
                if current_session is None:
                    # Start new session
                    current_session = {
                        'start': obs.timestamp,
                        'end': obs.timestamp,
                        'observations': [obs],
                    }
                else:
                    # Check if observation is close enough to be part of session
                    time_gap = (obs.timestamp - current_session['end']).total_seconds()

                    if time_gap <= 300:  # Less than 5 minutes gap
                        # Continue session
                        current_session['end'] = obs.timestamp
                        current_session['observations'].append(obs)
                    else:
                        # End current session, start new one
                        duration = (
                            current_session['end'] - current_session['start']
                        ).total_seconds() / 60

                        if duration >= min_duration_minutes:
                            sessions.append(
                                {
                                    'start': current_session['start'].isoformat(),
                                    'end': current_session['end'].isoformat(),
                                    'duration_minutes': duration,
                                    'observation_count': len(
                                        current_session['observations']
                                    ),
                                }
                            )

                        current_session = {
                            'start': obs.timestamp,
                            'end': obs.timestamp,
                            'observations': [obs],
                        }

        # Add final session if exists
        if current_session:
            duration = (
                current_session['end'] - current_session['start']
            ).total_seconds() / 60
            if duration >= min_duration_minutes:
                sessions.append(
                    {
                        'start': current_session['start'].isoformat(),
                        'end': current_session['end'].isoformat(),
                        'duration_minutes': duration,
                        'observation_count': len(current_session['observations']),
                    }
                )

        return sessions

    def suggest_optimizations(self, days: int = 14) -> list[str]:
        """Generate personalized optimization suggestions.

        Args:
            days: Number of days to analyze

        Returns:
            List of suggestions
        """
        suggestions = []

        # Get productivity patterns
        patterns = self.detect_productivity_patterns(days)

        # Suggestion 1: Time blocking
        if patterns['peak_hour'] is not None:
            suggestions.append(
                f"🎯 Your peak productivity hour is {patterns['peak_hour']}:00. "
                f"Consider blocking this time for deep work."
            )

        if patterns['peak_day'] is not None:
            suggestions.append(
                f"📅 You're most productive on {patterns['peak_day']}s. "
                f"Schedule important tasks for this day."
            )

        # Analyze context switches
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        observations = self.mall.query_by_timerange(start_time, end_time)

        app_switches = [obs for obs in observations if obs.event_type == 'app_switch']

        if len(app_switches) > 50:
            avg_per_day = len(app_switches) / days
            suggestions.append(
                f"⚠️  You switch apps {avg_per_day:.1f} times per day on average. "
                f"Consider batching similar tasks to reduce context switching."
            )

        # Analyze deep work
        deep_work = self.identify_deep_work_sessions(days)
        if deep_work:
            avg_session = sum(s['duration_minutes'] for s in deep_work) / len(deep_work)
            suggestions.append(
                f"✅ Your average deep work session is {avg_session:.0f} minutes. "
                f"Aim to increase this gradually."
            )
        else:
            suggestions.append(
                "💡 No sustained focus sessions detected. Try the Pomodoro technique "
                "(25 min focus, 5 min break)."
            )

        # Browser usage analysis
        browser_visits = [
            obs for obs in observations if obs.event_type == 'browser_visit'
        ]
        if len(browser_visits) > 100:
            suggestions.append(
                f"🌐 You visited {len(browser_visits)} web pages. "
                f"Consider using website blockers during focus time."
            )

        return suggestions

    def track_goal_progress(
        self, goal: str, keywords: list[str], days: int = 7
    ) -> dict:
        """Track progress toward a specific goal.

        Args:
            goal: Goal description
            keywords: Keywords related to the goal
            days: Number of days to track

        Returns:
            Progress dictionary
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        observations = self.mall.query_by_timerange(start_time, end_time)

        # Find observations related to goal
        related_obs = []
        for obs in observations:
            obs_text = str(obs.data).lower()
            if any(keyword.lower() in obs_text for keyword in keywords):
                related_obs.append(obs)

        # Calculate time spent
        total_time = len(related_obs) * 5  # Rough estimate: 5 min per observation

        # Group by day
        by_day = defaultdict(int)
        for obs in related_obs:
            day = obs.timestamp.strftime('%Y-%m-%d')
            by_day[day] += 1

        return {
            'goal': goal,
            'days_tracked': days,
            'total_observations': len(related_obs),
            'estimated_time_minutes': total_time,
            'daily_activity': dict(by_day),
            'average_per_day': len(related_obs) / days,
        }

    def compare_periods(
        self, period1_days: int, period2_days: int, offset_days: int = 0
    ) -> dict:
        """Compare two time periods.

        Args:
            period1_days: Length of first period (recent)
            period2_days: Length of second period
            offset_days: Days to offset second period

        Returns:
            Comparison dictionary
        """
        # Period 1 (recent)
        end1 = datetime.now() - timedelta(days=offset_days)
        start1 = end1 - timedelta(days=period1_days)
        obs1 = self.mall.query_by_timerange(start1, end1)

        # Period 2 (older)
        end2 = start1
        start2 = end2 - timedelta(days=period2_days)
        obs2 = self.mall.query_by_timerange(start2, end2)

        # Compare statistics
        def analyze_period(observations):
            event_types = Counter(obs.event_type for obs in observations)
            commits = sum(
                1
                for obs in observations
                if obs.event_type in ['git_commit', 'github_push']
            )
            browser = sum(
                1 for obs in observations if obs.event_type == 'browser_visit'
            )
            return {
                'total': len(observations),
                'commits': commits,
                'browser_visits': browser,
                'by_type': dict(event_types),
            }

        stats1 = analyze_period(obs1)
        stats2 = analyze_period(obs2)

        # Calculate changes
        changes = {}
        for key in ['total', 'commits', 'browser_visits']:
            if stats2[key] > 0:
                pct_change = ((stats1[key] - stats2[key]) / stats2[key]) * 100
                changes[key] = {
                    'period1': stats1[key],
                    'period2': stats2[key],
                    'change_pct': round(pct_change, 1),
                }
            else:
                changes[key] = {
                    'period1': stats1[key],
                    'period2': stats2[key],
                    'change_pct': 0,
                }

        return {
            'period1': {
                'start': start1.isoformat(),
                'end': end1.isoformat(),
                'stats': stats1,
            },
            'period2': {
                'start': start2.isoformat(),
                'end': end2.isoformat(),
                'stats': stats2,
            },
            'changes': changes,
            'summary': self._generate_comparison_summary(changes),
        }

    def predict_next_activity(self, context: Optional[str] = None) -> dict:
        """Predict what activity is likely next.

        Simple prediction based on historical patterns.

        Args:
            context: Optional context to filter predictions

        Returns:
            Prediction dictionary
        """
        # Get recent patterns (last 7 days)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        observations = self.mall.query_by_timerange(start_time, end_time)

        if not observations:
            return {'prediction': 'No data', 'confidence': 0}

        # Analyze current time patterns
        current_hour = datetime.now().hour
        current_day = datetime.now().strftime('%A')

        # Find similar times
        similar_time_obs = [
            obs
            for obs in observations
            if obs.timestamp.hour == current_hour
            and obs.timestamp.strftime('%A') == current_day
        ]

        if similar_time_obs:
            # Most common activity at this time
            activity_counts = Counter(obs.event_type for obs in similar_time_obs)
            most_common = activity_counts.most_common(1)[0]

            confidence = most_common[1] / len(similar_time_obs)

            return {
                'prediction': most_common[0],
                'confidence': round(confidence, 2),
                'similar_instances': most_common[1],
            }

        return {'prediction': 'No pattern found', 'confidence': 0}

    def _generate_productivity_recommendation(
        self, peak_hour: tuple, peak_day: tuple
    ) -> str:
        """Generate recommendation based on productivity patterns."""
        if peak_hour[0] is not None and peak_day[0] is not None:
            return (
                f"Schedule your most important work for {peak_day[0]}s "
                f"around {peak_hour[0]}:00 when you're most productive."
            )
        return "Continue tracking to identify your productivity patterns."

    def _generate_comparison_summary(self, changes: dict) -> str:
        """Generate summary of period comparison."""
        total_change = changes['total']['change_pct']
        commit_change = changes['commits']['change_pct']

        if total_change > 10:
            activity = "increased significantly"
        elif total_change > 0:
            activity = "increased slightly"
        elif total_change < -10:
            activity = "decreased significantly"
        else:
            activity = "remained stable"

        return f"Overall activity {activity} ({total_change:+.1f}%). " f"Commits {commit_change:+.1f}%."
