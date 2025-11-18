"""Tests for learning trajectory tracking."""

import pytest
from datetime import datetime, timedelta
from og.learning import LearningTracker, Skill, LearningSession, KnowledgeGap


def test_skill_creation():
    """Test Skill creation."""
    now = datetime.now()
    skill = Skill(
        name='python',
        category='language',
        first_seen=now - timedelta(days=30),
        last_used=now,
        usage_count=100,
        proficiency_score=0.75,
    )

    assert skill.name == 'python'
    assert skill.category == 'language'
    assert skill.proficiency_score == 0.75


def test_skill_properties():
    """Test Skill properties."""
    now = datetime.now()
    new_skill = Skill(
        name='rust',
        category='language',
        first_seen=now - timedelta(days=10),
        last_used=now - timedelta(days=2),
    )

    assert new_skill.is_new
    assert new_skill.is_active


def test_learning_session_creation():
    """Test LearningSession creation."""
    now = datetime.now()
    session = LearningSession(
        start=now,
        end=now + timedelta(hours=2),
        duration_minutes=120,
        skills=['python', 'django'],
        mode='tutorial',
    )

    assert session.duration_minutes == 120
    assert 'python' in session.skills
    assert session.mode == 'tutorial'


def test_knowledge_gap_creation():
    """Test KnowledgeGap creation."""
    gap = KnowledgeGap(
        skill='kubernetes',
        gap_type='missing',
        evidence=['No production usage'],
        severity='medium',
        recommendation='Try a tutorial project',
    )

    assert gap.skill == 'kubernetes'
    assert gap.gap_type == 'missing'


def test_tracker_creation():
    """Test LearningTracker creation."""
    tracker = LearningTracker()

    assert tracker is not None
    assert len(tracker.LANGUAGE_PATTERNS) > 0
    assert 'python' in tracker.LANGUAGE_PATTERNS
