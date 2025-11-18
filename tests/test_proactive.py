"""Tests for proactive insights."""

import pytest
from datetime import datetime
from og.proactive import ProactiveInsightEngine, Insight, ProactiveReport


def test_insight_creation():
    """Test Insight creation."""
    insight = Insight(
        type='anomaly',
        title='Delayed Commit',
        message='You usually commit by 11am',
        timestamp=datetime.now(),
        priority='normal',
        confidence=0.8,
    )

    assert insight.type == 'anomaly'
    assert insight.priority == 'normal'
    assert insight.confidence == 0.8


def test_insight_to_notification():
    """Test notification formatting."""
    insight = Insight(
        type='suggestion',
        title='Peak Time',
        message='This is your peak productivity hour',
        timestamp=datetime.now(),
        priority='high',
        action='Block this time for deep work',
    )

    notification = insight.to_notification(format='simple')
    assert '[SUGGESTION]' in notification
    assert 'Peak Time' in notification

    rich = insight.to_notification(format='rich')
    assert '⚠️' in rich or '💡' in rich


def test_proactive_report_creation():
    """Test ProactiveReport creation."""
    now = datetime.now()
    report = ProactiveReport(
        period_start=now,
        period_end=now,
        insights=[],
        anomalies=[],
        reminders=[],
        suggestions=[],
    )

    assert report.period_start == now


def test_engine_creation():
    """Test ProactiveInsightEngine creation."""
    engine = ProactiveInsightEngine()

    assert engine is not None
    assert engine._patterns == {}
