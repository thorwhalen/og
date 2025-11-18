"""Tests for voice interface."""

import pytest
from datetime import datetime
from og.voice import VoiceInterface, VoiceCommand


def test_voice_command_creation():
    """Test VoiceCommand creation."""
    cmd = VoiceCommand(
        raw_text='What did I work on today?',
        intent='summary',
        action='get_summary',
        parameters={'days': 1},
        confidence=0.9,
    )

    assert cmd.intent == 'summary'
    assert cmd.confidence == 0.9


def test_voice_interface_creation():
    """Test VoiceInterface creation."""
    interface = VoiceInterface()

    assert interface is not None
    assert len(interface.PATTERNS) > 0


def test_parse_summary_command():
    """Test parsing summary commands."""
    interface = VoiceInterface()

    cmd = interface.parse_command("What did I work on today?")

    assert cmd is not None
    assert cmd.intent == 'summary'
    assert cmd.parameters.get('period') == 'today'


def test_parse_question_command():
    """Test parsing question commands."""
    interface = VoiceInterface()

    cmd = interface.parse_command("How many commits did I make?")

    assert cmd is not None
    assert cmd.intent == 'question'
