"""Tests for meeting intelligence."""

import pytest
from datetime import datetime, timedelta
from og.meetings import (
    MeetingAnalyzer,
    Meeting,
    ActionItem,
    MeetingParticipant,
    MeetingInsights,
)


def test_action_item_creation():
    """Test ActionItem creation."""
    item = ActionItem(
        description='Update documentation',
        assignee='john',
        completed=False,
    )

    assert item.description == 'Update documentation'
    assert item.assignee == 'john'
    assert item.completed is False


def test_meeting_participant_creation():
    """Test MeetingParticipant creation."""
    participant = MeetingParticipant(
        name='Alice',
        talk_time_seconds=600,
        talk_percentage=40.0,
        questions_asked=3,
    )

    assert participant.name == 'Alice'
    assert participant.talk_percentage == 40.0


def test_meeting_creation():
    """Test Meeting creation."""
    now = datetime.now()
    meeting = Meeting(
        title='Sprint Planning',
        start_time=now,
        end_time=now + timedelta(hours=1),
        duration_minutes=60,
    )

    assert meeting.title == 'Sprint Planning'
    assert meeting.duration_minutes == 60


def test_meeting_to_markdown():
    """Test meeting markdown generation."""
    now = datetime.now()
    meeting = Meeting(
        title='Daily Standup',
        start_time=now,
        end_time=now + timedelta(minutes=15),
        duration_minutes=15,
        action_items=[
            ActionItem('Review PR #123', 'bob'),
        ],
        decisions=['Use FastAPI for new service'],
        efficiency_score=8.5,
        value_score=7.0,
    )

    md = meeting.to_markdown()

    assert '# Daily Standup' in md
    assert 'Review PR #123' in md
    assert 'Use FastAPI for new service' in md
    assert '8.5' in md


def test_analyzer_creation():
    """Test MeetingAnalyzer creation."""
    analyzer = MeetingAnalyzer()

    assert analyzer is not None
    assert len(analyzer.ACTION_PATTERNS) > 0


def test_meeting_insights_to_markdown():
    """Test insights markdown generation."""
    insights = MeetingInsights(
        total_meetings=25,
        total_meeting_time_hours=12.5,
        avg_meeting_duration_minutes=30,
        recurring_meetings=['Daily Standup', 'Sprint Planning'],
        low_value_meetings=[],
        high_efficiency_meetings=[],
        recommendations=['Reduce meeting time by 20%'],
    )

    md = insights.to_markdown()

    assert '# Meeting Intelligence Report' in md
    assert 'Total meetings: 25' in md
    assert 'Daily Standup' in md
