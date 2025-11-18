"""Tests for context switching cost analysis."""

import pytest
from datetime import datetime, timedelta
from og.switching_cost import (
    SwitchingCostAnalyzer,
    Switch,
    FlowSession,
    SwitchingReport,
)


def test_switch_creation():
    """Test Switch creation."""
    switch = Switch(
        timestamp=datetime.now(),
        from_context='Chrome',
        to_context='VSCode',
        switch_type='app',
        recovery_time=5.0,
    )

    assert switch.from_context == 'Chrome'
    assert switch.to_context == 'VSCode'
    assert switch.switch_type == 'app'
    assert switch.recovery_time == 5.0


def test_flow_session_creation():
    """Test FlowSession creation."""
    now = datetime.now()
    session = FlowSession(
        start=now,
        end=now + timedelta(hours=2),
        duration_minutes=120,
        context='coding',
        interruption_count=1,
        productivity_score=0.9,
    )

    assert session.duration_minutes == 120
    assert session.interruption_count == 1
    assert session.productivity_score == 0.9


def test_switching_report_to_markdown():
    """Test report markdown generation."""
    report = SwitchingReport(
        period_start=datetime.now() - timedelta(days=7),
        period_end=datetime.now(),
        total_switches=50,
        switches_by_type={'app': 30, 'project': 20},
        estimated_cost_minutes=250,
        flow_sessions=[],
        fragmented_sessions=[],
        top_interrupters=[('Slack', 15), ('Email', 10)],
        recommendations=['Enable focus mode'],
    )

    md = report.to_markdown()

    assert '# Context Switching Cost Analysis' in md
    assert 'Total context switches: 50' in md
    assert '250.0 minutes' in md
    assert 'Slack' in md


def test_analyzer_creation():
    """Test SwitchingCostAnalyzer creation."""
    analyzer = SwitchingCostAnalyzer()

    assert analyzer is not None
    assert analyzer.FLOW_SESSION_MIN_DURATION == 30


def test_recovery_times():
    """Test recovery time constants."""
    analyzer = SwitchingCostAnalyzer()

    assert analyzer.RECOVERY_TIMES['app'] == 5.0
    assert analyzer.RECOVERY_TIMES['project'] == 15.0
    assert analyzer.RECOVERY_TIMES['browser_tab'] == 2.0
