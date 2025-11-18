"""Meeting intelligence and analysis.

This module provides meeting transcription, action item extraction,
efficiency analysis, and meeting value assessment.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re


@dataclass
class ActionItem:
    """An action item extracted from a meeting."""

    description: str
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    completed: bool = False
    confidence: float = 1.0


@dataclass
class MeetingParticipant:
    """A meeting participant with stats."""

    name: str
    talk_time_seconds: float = 0
    talk_percentage: float = 0
    questions_asked: int = 0
    decisions_made: int = 0


@dataclass
class Meeting:
    """A meeting with metadata and analysis."""

    title: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    participants: List[MeetingParticipant] = field(default_factory=list)
    transcript: Optional[str] = None
    summary: Optional[str] = None
    action_items: List[ActionItem] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    efficiency_score: float = 0.0
    value_score: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate markdown summary."""
        lines = [
            f"# {self.title}\n",
            f"**Date:** {self.start_time.strftime('%Y-%m-%d %H:%M')}",
            f"**Duration:** {self.duration_minutes:.0f} minutes",
            f"**Efficiency Score:** {self.efficiency_score:.1f}/10",
            f"**Value Score:** {self.value_score:.1f}/10\n",
        ]

        if self.participants:
            lines.append("## Participants\n")
            for p in self.participants:
                lines.append(
                    f"- {p.name} ({p.talk_percentage:.0f}% talk time, "
                    f"{p.questions_asked} questions)"
                )
            lines.append("")

        if self.summary:
            lines.append("## Summary\n")
            lines.append(self.summary)
            lines.append("")

        if self.decisions:
            lines.append("## Decisions\n")
            for decision in self.decisions:
                lines.append(f"- {decision}")
            lines.append("")

        if self.action_items:
            lines.append("## Action Items\n")
            for item in self.action_items:
                assignee = f" ({item.assignee})" if item.assignee else ""
                due = f" - Due: {item.due_date.strftime('%Y-%m-%d')}" if item.due_date else ""
                checkbox = "[x]" if item.completed else "[ ]"
                lines.append(f"- {checkbox} {item.description}{assignee}{due}")
            lines.append("")

        if self.topics:
            lines.append("## Topics\n")
            for topic in self.topics:
                lines.append(f"- {topic}")
            lines.append("")

        return "\n".join(lines)


@dataclass
class MeetingInsights:
    """Insights about meeting patterns."""

    total_meetings: int
    total_meeting_time_hours: float
    avg_meeting_duration_minutes: float
    recurring_meetings: List[str]
    low_value_meetings: List[Meeting]
    high_efficiency_meetings: List[Meeting]
    recommendations: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Meeting Intelligence Report\n",
            "## Overview\n",
            f"- **Total meetings:** {self.total_meetings}",
            f"- **Total time in meetings:** {self.total_meeting_time_hours:.1f} hours",
            f"- **Average duration:** {self.avg_meeting_duration_minutes:.0f} minutes\n",
        ]

        if self.recurring_meetings:
            lines.append("## Recurring Meetings\n")
            for meeting_title in self.recurring_meetings[:10]:
                lines.append(f"- {meeting_title}")
            lines.append("")

        if self.low_value_meetings:
            lines.append("## Low Value Meetings (Consider Canceling)\n")
            for meeting in self.low_value_meetings[:5]:
                lines.append(
                    f"- {meeting.title} "
                    f"(Value: {meeting.value_score:.1f}/10, "
                    f"Duration: {meeting.duration_minutes:.0f} min)"
                )
            lines.append("")

        if self.high_efficiency_meetings:
            lines.append("## High Efficiency Meetings (Good Examples)\n")
            for meeting in self.high_efficiency_meetings[:5]:
                lines.append(
                    f"- {meeting.title} "
                    f"(Efficiency: {meeting.efficiency_score:.1f}/10)"
                )
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations\n")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        if self.metrics:
            lines.append("## Detailed Metrics\n")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.2f}")
                else:
                    lines.append(f"- **{key}:** {value}")

        return "\n".join(lines)


class MeetingAnalyzer:
    """Analyzes meetings for efficiency and value."""

    # Patterns for extracting action items
    ACTION_PATTERNS = [
        r'(?:TODO|Action item|Action|Task):\s*(.+)',
        r'(?:^|\s)(?:will|should|need to|must)\s+(.+?)(?:\.|$)',
        r'@(\w+)\s+(?:please|can you|could you)\s+(.+?)(?:\.|$)',
    ]

    # Patterns for extracting decisions
    DECISION_PATTERNS = [
        r'(?:decided|agreed|decision):\s*(.+)',
        r'(?:we will|we\'ll|let\'s)\s+(.+?)(?:\.|$)',
    ]

    def __init__(self, og_instance=None):
        """Initialize the analyzer.

        Args:
            og_instance: Optional OG instance for accessing observations
        """
        self.og = og_instance

    def analyze_meeting(
        self,
        title: str,
        start_time: datetime,
        end_time: datetime,
        transcript: Optional[str] = None,
        participants: Optional[List[str]] = None,
    ) -> Meeting:
        """Analyze a single meeting.

        Args:
            title: Meeting title
            start_time: Start time
            end_time: End time
            transcript: Optional meeting transcript
            participants: Optional list of participant names

        Returns:
            Meeting object with analysis
        """
        duration = (end_time - start_time).total_seconds() / 60

        meeting = Meeting(
            title=title,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration,
        )

        # Parse transcript if available
        if transcript:
            meeting.transcript = transcript
            meeting.action_items = self._extract_action_items(transcript)
            meeting.decisions = self._extract_decisions(transcript)
            meeting.topics = self._extract_topics(transcript)
            meeting.summary = self._generate_summary(transcript)

            # Analyze participants
            if participants:
                meeting.participants = self._analyze_participants(
                    transcript, participants
                )

        # Calculate efficiency score
        meeting.efficiency_score = self._calculate_efficiency_score(meeting)

        # Calculate value score
        meeting.value_score = self._calculate_value_score(meeting)

        return meeting

    def analyze_meetings(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
    ) -> MeetingInsights:
        """Analyze all meetings in a period.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            days: Number of days to analyze (if start_date not provided)

        Returns:
            MeetingInsights with analysis results
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Get calendar observations
        meetings = self._get_meetings(start_date, end_date)

        if not meetings:
            return MeetingInsights(
                total_meetings=0,
                total_meeting_time_hours=0,
                avg_meeting_duration_minutes=0,
                recurring_meetings=[],
                low_value_meetings=[],
                high_efficiency_meetings=[],
                recommendations=[],
            )

        # Calculate metrics
        total_time = sum(m.duration_minutes for m in meetings)
        avg_duration = total_time / len(meetings) if meetings else 0

        # Identify recurring meetings
        recurring = self._identify_recurring_meetings(meetings)

        # Find low value meetings
        low_value = sorted(meetings, key=lambda m: m.value_score)[:5]

        # Find high efficiency meetings
        high_efficiency = sorted(
            meetings, key=lambda m: m.efficiency_score, reverse=True
        )[:5]

        # Generate recommendations
        recommendations = self._generate_meeting_recommendations(
            meetings, recurring, low_value
        )

        # Detailed metrics
        metrics = self._calculate_meeting_metrics(meetings)

        return MeetingInsights(
            total_meetings=len(meetings),
            total_meeting_time_hours=total_time / 60,
            avg_meeting_duration_minutes=avg_duration,
            recurring_meetings=recurring,
            low_value_meetings=low_value,
            high_efficiency_meetings=high_efficiency,
            recommendations=recommendations,
            metrics=metrics,
        )

    def _get_meetings(
        self, start_date: datetime, end_date: datetime
    ) -> List[Meeting]:
        """Get meetings from observations."""
        if not self.og:
            return []

        observations = self.og.recent_activity(
            start_date=start_date,
            end_date=end_date,
            event_type='calendar_event',
        )

        meetings = []

        for obs in observations:
            title = obs.data.get('title', 'Untitled')
            start = obs.data.get('start_time')
            end = obs.data.get('end_time')
            attendees = obs.data.get('attendees', [])

            if start and end:
                meeting = self.analyze_meeting(
                    title=title,
                    start_time=start,
                    end_time=end,
                    participants=attendees,
                )
                meetings.append(meeting)

        return meetings

    def _extract_action_items(self, transcript: str) -> List[ActionItem]:
        """Extract action items from transcript."""
        action_items = []

        for pattern in self.ACTION_PATTERNS:
            matches = re.finditer(pattern, transcript, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                description = match.group(1) if match.lastindex >= 1 else match.group(0)
                assignee = match.group(1) if len(match.groups()) > 1 else None

                action_items.append(ActionItem(
                    description=description.strip(),
                    assignee=assignee,
                    confidence=0.8,
                ))

        return action_items

    def _extract_decisions(self, transcript: str) -> List[str]:
        """Extract decisions from transcript."""
        decisions = []

        for pattern in self.DECISION_PATTERNS:
            matches = re.finditer(pattern, transcript, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                decision = match.group(1).strip()
                decisions.append(decision)

        return decisions

    def _extract_topics(self, transcript: str) -> List[str]:
        """Extract main topics from transcript."""
        # Simple topic extraction - could be enhanced with NLP
        # For now, look for phrases that indicate topic changes
        topics = []

        topic_indicators = [
            r'(?:let\'s talk about|discuss|next topic is|moving on to)\s+(.+?)(?:\.|$)',
            r'(?:regarding|about)\s+(.+?)(?:\.|,|$)',
        ]

        for pattern in topic_indicators:
            matches = re.finditer(pattern, transcript, re.IGNORECASE)
            for match in matches:
                topic = match.group(1).strip()
                topics.append(topic)

        return topics[:5]  # Return top 5 topics

    def _generate_summary(self, transcript: str) -> str:
        """Generate a brief summary of the meeting."""
        # Simple summary - first few sentences
        # In production, would use AI for better summarization
        sentences = re.split(r'[.!?]\s+', transcript)
        summary = '. '.join(sentences[:3])
        return summary + '.'

    def _analyze_participants(
        self, transcript: str, participant_names: List[str]
    ) -> List[MeetingParticipant]:
        """Analyze participant engagement."""
        participants = []

        for name in participant_names:
            # Count occurrences of name speaking
            # Simplified - in production would parse speaker labels
            pattern = rf'{name}:\s*([^:]+?)(?:{"|".join(participant_names)}:|$)'
            matches = re.findall(pattern, transcript, re.IGNORECASE)

            talk_time = len(matches)  # Simplified metric
            questions = len(re.findall(rf'{name}:.*?\?', transcript, re.IGNORECASE))

            participant = MeetingParticipant(
                name=name,
                talk_time_seconds=talk_time * 30,  # Rough estimate
                questions_asked=questions,
            )

            participants.append(participant)

        # Calculate talk percentages
        total_talk = sum(p.talk_time_seconds for p in participants)
        if total_talk > 0:
            for p in participants:
                p.talk_percentage = (p.talk_time_seconds / total_talk) * 100

        return participants

    def _calculate_efficiency_score(self, meeting: Meeting) -> float:
        """Calculate meeting efficiency score (0-10)."""
        score = 5.0  # Base score

        # Bonus for having action items
        if meeting.action_items:
            score += min(2.0, len(meeting.action_items) * 0.5)

        # Bonus for having decisions
        if meeting.decisions:
            score += min(1.5, len(meeting.decisions) * 0.5)

        # Penalty for long duration without outcomes
        if meeting.duration_minutes > 60 and not meeting.action_items:
            score -= 2.0

        # Bonus for balanced participation
        if meeting.participants:
            talk_times = [p.talk_percentage for p in meeting.participants]
            if talk_times:
                # Check if participation is balanced (no one dominates)
                max_talk = max(talk_times)
                if max_talk < 60:  # No one speaks more than 60%
                    score += 1.5

        return max(0, min(10, score))

    def _calculate_value_score(self, meeting: Meeting) -> float:
        """Calculate meeting value score (0-10)."""
        score = 5.0  # Base score

        # Value increases with outcomes
        if meeting.action_items:
            score += min(3.0, len(meeting.action_items) * 0.7)

        if meeting.decisions:
            score += min(2.0, len(meeting.decisions) * 0.7)

        # Penalty for meetings with no clear outcomes
        if not meeting.action_items and not meeting.decisions:
            score -= 3.0

        # Penalty for very long meetings
        if meeting.duration_minutes > 90:
            score -= 1.5

        # Bonus for shorter focused meetings with outcomes
        if meeting.duration_minutes <= 30 and (meeting.action_items or meeting.decisions):
            score += 1.5

        return max(0, min(10, score))

    def _identify_recurring_meetings(self, meetings: List[Meeting]) -> List[str]:
        """Identify recurring meetings."""
        title_counts = defaultdict(int)

        for meeting in meetings:
            # Normalize title (remove dates, numbers)
            normalized = re.sub(r'\d{4}-\d{2}-\d{2}', '', meeting.title)
            normalized = re.sub(r'\d+', '', normalized).strip()
            title_counts[normalized] += 1

        # Return meetings that occur more than once
        recurring = [
            title for title, count in title_counts.items() if count > 1
        ]

        return sorted(recurring, key=lambda t: title_counts[t], reverse=True)

    def _generate_meeting_recommendations(
        self,
        meetings: List[Meeting],
        recurring: List[str],
        low_value: List[Meeting],
    ) -> List[str]:
        """Generate recommendations for improving meeting culture."""
        recommendations = []

        # Too many meetings
        if len(meetings) > 20:
            recommendations.append(
                f"You had {len(meetings)} meetings this period. "
                "Consider declining non-essential meetings to preserve focus time."
            )

        # Long average duration
        avg_duration = sum(m.duration_minutes for m in meetings) / len(meetings)
        if avg_duration > 45:
            recommendations.append(
                f"Average meeting duration is {avg_duration:.0f} minutes. "
                "Try setting 25 or 45-minute defaults instead of 30/60."
            )

        # Low value recurring meetings
        low_value_recurring = [m for m in low_value if m.title in recurring]
        if low_value_recurring:
            recommendations.append(
                f"Found {len(low_value_recurring)} recurring meetings with low value scores. "
                "Consider canceling or restructuring these."
            )

        # Meetings without outcomes
        no_outcomes = [
            m for m in meetings
            if not m.action_items and not m.decisions
        ]
        if len(no_outcomes) > len(meetings) * 0.3:
            recommendations.append(
                f"{len(no_outcomes)} meetings had no clear action items or decisions. "
                "Always end meetings with clear next steps."
            )

        # Unbalanced participation
        unbalanced = [
            m for m in meetings
            if m.participants and any(p.talk_percentage > 70 for p in m.participants)
        ]
        if unbalanced:
            recommendations.append(
                f"{len(unbalanced)} meetings had unbalanced participation. "
                "Encourage quieter participants to share their perspectives."
            )

        return recommendations

    def _calculate_meeting_metrics(self, meetings: List[Meeting]) -> Dict[str, Any]:
        """Calculate detailed meeting metrics."""
        total_time = sum(m.duration_minutes for m in meetings)

        metrics = {
            'meetings_per_week': len(meetings) * 7 / 30,  # Assuming 30-day period
            'meeting_time_percentage': total_time / (30 * 8 * 60) * 100,  # % of work hours
            'avg_efficiency_score': sum(m.efficiency_score for m in meetings) / len(meetings) if meetings else 0,
            'avg_value_score': sum(m.value_score for m in meetings) / len(meetings) if meetings else 0,
            'meetings_with_action_items': len([m for m in meetings if m.action_items]),
            'meetings_with_decisions': len([m for m in meetings if m.decisions]),
            'total_action_items': sum(len(m.action_items) for m in meetings),
        }

        return metrics
