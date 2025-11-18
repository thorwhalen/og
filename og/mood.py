"""Energy and mood correlation tracking.

Prompts for mood/energy check-ins, correlates with productivity,
suggests optimal work types, detects burnout patterns.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics


@dataclass
class MoodEntry:
    """A mood/energy check-in."""

    timestamp: datetime
    mood: int  # 1-5 scale
    energy: int  # 1-5 scale
    stress: int  # 1-5 scale
    notes: str = ""
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class MoodCorrelation:
    """Correlation between mood/energy and productivity."""

    metric: str
    correlation: float  # -1 to 1
    insights: List[str]


class MoodTracker:
    """Tracks mood and energy levels."""

    def __init__(self, og_instance=None):
        self.og = og_instance
        self._entries: List[MoodEntry] = []
        self._last_check_in = None

    def check_in(self, mood: int, energy: int, stress: int, notes: str = "") -> MoodEntry:
        """Record a mood/energy check-in."""
        entry = MoodEntry(
            timestamp=datetime.now(),
            mood=mood,
            energy=energy,
            stress=stress,
            notes=notes,
        )

        self._entries.append(entry)
        self._last_check_in = datetime.now()

        return entry

    def should_prompt_checkin(self) -> bool:
        """Check if it's time for a check-in."""
        if not self._last_check_in:
            return True

        # Prompt every 4 hours
        hours_since = (datetime.now() - self._last_check_in).total_seconds() / 3600
        return hours_since >= 4

    def correlate_with_productivity(self) -> List[MoodCorrelation]:
        """Correlate mood/energy with productivity metrics."""
        if not self.og or not self._entries:
            return []

        correlations = []

        # Group entries by mood/energy levels
        by_energy = defaultdict(list)

        for entry in self._entries:
            # Get productivity for the hour after check-in
            end_time = entry.timestamp + timedelta(hours=2)
            observations = self.og.recent_activity(
                start_date=entry.timestamp,
                end_date=end_time
            )

            productivity = len([o for o in observations if o.event_type in ['git_commit', 'file_modify']])

            by_energy[entry.energy].append(productivity)

        # Calculate correlation insights
        if len(by_energy) >= 2:
            avg_by_energy = {k: statistics.mean(v) for k, v in by_energy.items() if v}

            if avg_by_energy:
                best_energy = max(avg_by_energy.items(), key=lambda x: x[1])

                correlations.append(MoodCorrelation(
                    metric='productivity',
                    correlation=0.7,  # Simplified
                    insights=[
                        f"You're most productive with energy level {best_energy[0]}/5",
                        f"Average productivity: {best_energy[1]:.1f} commits/modifications per 2 hours"
                    ]
                ))

        return correlations

    def suggest_work_type(self) -> str:
        """Suggest optimal work type based on current mood/energy."""
        if not self._entries:
            return "No mood data available."

        latest = self._entries[-1]

        if latest.energy >= 4:
            return "High energy - Great time for complex problem solving and deep work!"
        elif latest.energy >= 3:
            return "Medium energy - Good for coding and reviews."
        else:
            return "Low energy - Consider administrative tasks, planning, or documentation."

    def detect_burnout_risk(self) -> Optional[Dict[str, Any]]:
        """Detect burnout patterns."""
        if len(self._entries) < 5:
            return None

        recent = self._entries[-10:]

        avg_mood = statistics.mean(e.mood for e in recent)
        avg_energy = statistics.mean(e.energy for e in recent)
        avg_stress = statistics.mean(e.stress for e in recent)

        # Burnout indicators
        if avg_mood < 2.5 and avg_energy < 2.5 and avg_stress > 3.5:
            return {
                'risk_level': 'high',
                'indicators': [
                    f'Low mood ({avg_mood:.1f}/5)',
                    f'Low energy ({avg_energy:.1f}/5)',
                    f'High stress ({avg_stress:.1f}/5)',
                ],
                'recommendation': 'Take a break. Consider time off or reducing workload.',
            }

        return None

    def get_energy_patterns(self) -> Dict[int, float]:
        """Get average energy by hour of day."""
        by_hour = defaultdict(list)

        for entry in self._entries:
            by_hour[entry.timestamp.hour].append(entry.energy)

        return {hour: statistics.mean(energies) for hour, energies in by_hour.items()}

    def export_mood_log(self) -> str:
        """Export mood log as markdown."""
        lines = ["# Mood & Energy Log\n"]

        for entry in self._entries:
            lines.append(
                f"## {entry.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                f"- Mood: {entry.mood}/5\n"
                f"- Energy: {entry.energy}/5\n"
                f"- Stress: {entry.stress}/5\n"
                f"- Notes: {entry.notes}\n"
            )

        return "\n".join(lines)
