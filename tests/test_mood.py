"""Tests for mood tracking."""

import pytest
from datetime import datetime, timedelta
from og.mood import MoodTracker, MoodEntry


def test_mood_entry_creation():
    """Test MoodEntry creation."""
    entry = MoodEntry(
        timestamp=datetime.now(),
        mood=4,
        energy=3,
        stress=2,
        notes='Feeling good',
    )

    assert entry.mood == 4
    assert entry.energy == 3
    assert entry.stress == 2


def test_mood_tracker_creation():
    """Test MoodTracker creation."""
    tracker = MoodTracker()

    assert tracker is not None
    assert len(tracker._entries) == 0


def test_mood_checkin():
    """Test mood check-in."""
    tracker = MoodTracker()

    entry = tracker.check_in(mood=4, energy=3, stress=2, notes='Good day')

    assert len(tracker._entries) == 1
    assert tracker._last_check_in is not None


def test_should_prompt_checkin():
    """Test check-in prompting."""
    tracker = MoodTracker()

    # Should prompt initially
    assert tracker.should_prompt_checkin()

    # Check in
    tracker.check_in(4, 3, 2)

    # Should not prompt immediately after
    assert not tracker.should_prompt_checkin()


def test_suggest_work_type():
    """Test work type suggestions."""
    tracker = MoodTracker()

    tracker.check_in(mood=4, energy=5, stress=1)
    suggestion = tracker.suggest_work_type()

    assert 'High energy' in suggestion or 'complex' in suggestion.lower()
