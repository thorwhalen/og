"""Predictive scheduling and calendar optimization.

Learns optimal times for different work types, predicts task duration,
auto-schedules work blocks based on patterns.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics


@dataclass
class ScheduleBlock:
    """A scheduled block of time."""

    start: datetime
    end: datetime
    activity_type: str  # 'deep_work', 'meetings', 'admin', 'learning'
    description: str
    priority: int = 1
    predicted_productivity: float = 1.0


@dataclass
class ScheduleRecommendation:
    """A recommended schedule optimization."""

    recommendation_type: str
    message: str
    suggested_block: Optional[ScheduleBlock] = None
    confidence: float = 1.0


class PredictiveScheduler:
    """Predicts and optimizes schedule."""

    def __init__(self, og_instance=None):
        self.og = og_instance
        self._productivity_by_hour: Dict[int, List[float]] = defaultdict(list)
        self._duration_history: Dict[str, List[float]] = defaultdict(list)

    def learn_patterns(self):
        """Learn productivity patterns from history."""
        if not self.og:
            return

        # Analyze productivity by hour
        observations = self.og.recent_activity(days=30)

        by_hour = defaultdict(int)
        for obs in observations:
            if obs.event_type in ['git_commit', 'file_modify']:
                by_hour[obs.timestamp.hour] += 1

        # Store productivity scores
        for hour, count in by_hour.items():
            self._productivity_by_hour[hour].append(count)

    def predict_optimal_time(self, activity_type: str, duration_hours: float) -> List[datetime]:
        """Predict optimal time for an activity."""
        # Get productivity by hour
        if not self._productivity_by_hour:
            self.learn_patterns()

        avg_productivity = {
            hour: statistics.mean(scores) if scores else 0
            for hour, scores in self._productivity_by_hour.items()
        }

        # Find best hours
        sorted_hours = sorted(avg_productivity.items(), key=lambda x: x[1], reverse=True)

        # Suggest times in next 7 days
        suggestions = []
        now = datetime.now()

        for day_offset in range(7):
            date = now + timedelta(days=day_offset)

            for hour, productivity in sorted_hours[:3]:  # Top 3 hours
                block_start = date.replace(hour=hour, minute=0, second=0, microsecond=0)

                if block_start > now:
                    suggestions.append(block_start)

                    if len(suggestions) >= 5:
                        return suggestions

        return suggestions

    def predict_duration(self, task_description: str) -> float:
        """Predict task duration based on historical data."""
        # Simple keyword-based prediction
        # In production, would use ML model

        if 'bug' in task_description.lower() or 'fix' in task_description.lower():
            return 2.0  # 2 hours for bug fixes

        if 'feature' in task_description.lower():
            return 8.0  # 8 hours for features

        return 4.0  # Default 4 hours

    def optimize_schedule(
        self, start_date: datetime, end_date: datetime
    ) -> List[ScheduleRecommendation]:
        """Generate schedule optimization recommendations."""
        recommendations = []

        # Learn patterns first
        self.learn_patterns()

        if not self._productivity_by_hour:
            return recommendations

        # Find peak productivity hours
        avg_productivity = {
            hour: statistics.mean(scores) if scores else 0
            for hour, scores in self._productivity_by_hour.items()
        }

        peak_hours = sorted(avg_productivity.items(), key=lambda x: x[1], reverse=True)[:2]

        if peak_hours:
            best_hour = peak_hours[0][0]

            recommendations.append(ScheduleRecommendation(
                recommendation_type='deep_work',
                message=f"Schedule deep work sessions at {best_hour}:00 - your most productive time",
                suggested_block=ScheduleBlock(
                    start=datetime.now().replace(hour=best_hour, minute=0),
                    end=datetime.now().replace(hour=best_hour+2, minute=0),
                    activity_type='deep_work',
                    description='Focus time',
                    priority=5,
                    predicted_productivity=1.0,
                ),
                confidence=0.8,
            ))

        # Suggest batching meetings
        recommendations.append(ScheduleRecommendation(
            recommendation_type='meetings',
            message="Batch meetings in the afternoon to preserve morning focus time",
            confidence=0.7,
        ))

        return recommendations

    def auto_schedule_tasks(
        self, tasks: List[Dict[str, Any]], start_date: datetime
    ) -> List[ScheduleBlock]:
        """Auto-schedule tasks based on predicted optimal times."""
        schedule = []

        for task in tasks:
            title = task.get('title', '')
            duration = self.predict_duration(title)

            # Find optimal time
            optimal_times = self.predict_optimal_time('work', duration)

            if optimal_times:
                start = optimal_times[0]
                end = start + timedelta(hours=duration)

                schedule.append(ScheduleBlock(
                    start=start,
                    end=end,
                    activity_type='work',
                    description=title,
                    priority=task.get('priority', 1),
                    predicted_productivity=0.8,
                ))

        return schedule

    def suggest_breaks(self, schedule: List[ScheduleBlock]) -> List[ScheduleBlock]:
        """Suggest break times in schedule."""
        breaks = []

        for i, block in enumerate(schedule):
            # Suggest break after 2+ hour blocks
            if (block.end - block.start).total_seconds() / 3600 >= 2:
                breaks.append(ScheduleBlock(
                    start=block.end,
                    end=block.end + timedelta(minutes=15),
                    activity_type='break',
                    description='Break',
                    priority=3,
                ))

        return breaks
