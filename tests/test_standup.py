"""Tests for automated standup generation."""

import pytest
from datetime import datetime, timedelta
from og.standup import StandupGenerator, Standup, StandupItem


def test_standup_item_creation():
    """Test StandupItem creation."""
    item = StandupItem(
        category='completed',
        description='Fixed authentication bug',
        context='my-repo',
    )

    assert item.category == 'completed'
    assert item.description == 'Fixed authentication bug'
    assert item.context == 'my-repo'


def test_standup_to_text():
    """Test standup text formatting."""
    standup = Standup(
        date=datetime.now(),
        yesterday=[
            StandupItem('completed', 'Fixed bug', 'repo1'),
            StandupItem('completed', 'Reviewed PR', 'repo2'),
        ],
        today=[
            StandupItem('planned', 'Implement feature', 'repo1'),
        ],
        blockers=[
            StandupItem('blocker', 'Waiting on API key'),
        ],
    )

    text = standup.to_text()

    assert 'Yesterday:' in text
    assert 'Fixed bug' in text
    assert 'Today:' in text
    assert 'Implement feature' in text
    assert 'Blockers:' in text
    assert 'Waiting on API key' in text


def test_standup_to_slack():
    """Test Slack formatting."""
    standup = Standup(
        date=datetime.now(),
        yesterday=[StandupItem('completed', 'Test task')],
        today=[],
        blockers=[],
    )

    slack_text = standup.to_text(format='slack')

    assert '*Yesterday:*' in slack_text
    assert 'Test task' in slack_text


def test_standup_to_markdown():
    """Test markdown formatting."""
    standup = Standup(
        date=datetime.now(),
        yesterday=[StandupItem('completed', 'Test task')],
        today=[],
        blockers=[],
    )

    md = standup.to_markdown()

    assert '# Standup' in md
    assert '## Yesterday' in md


def test_standup_generator_creation():
    """Test StandupGenerator creation."""
    generator = StandupGenerator()

    assert generator is not None
    assert generator.og is None


def test_work_verbs_mapping():
    """Test work verb mappings."""
    generator = StandupGenerator()

    assert 'git_commit' in generator._work_verbs
    assert 'Fixed' in generator._work_verbs['git_commit']
