"""Tests for focus mode."""

import pytest
from datetime import datetime
from og.focus_mode import FocusMode, FocusSession


def test_focus_session_creation():
    """Test FocusSession creation."""
    session = FocusSession(
        start=datetime.now(),
        duration_minutes=60,
    )

    assert session.duration_minutes == 60
    assert session.interruptions == 0


def test_focus_mode_creation():
    """Test FocusMode creation."""
    focus = FocusMode()

    assert focus is not None
    assert not focus._active
    assert len(focus.DEFAULT_BLOCKED_SITES) > 0


def test_focus_mode_enable_disable():
    """Test enabling and disabling focus mode."""
    focus = FocusMode()

    # Should start inactive
    assert not focus._active

    # Enable
    focus.enable(duration_minutes=30)
    assert focus._active
    assert focus._current_session is not None
    assert focus._current_session.duration_minutes == 30

    # Disable
    focus.disable()
    assert not focus._active
