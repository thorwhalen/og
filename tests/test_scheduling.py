"""Tests for predictive scheduling."""

import pytest
from datetime import datetime, timedelta
from og.scheduling import PredictiveScheduler, ScheduleBlock, ScheduleRecommendation


def test_schedule_block_creation():
    """Test ScheduleBlock creation."""
    now = datetime.now()
    block = ScheduleBlock(
        start=now,
        end=now + timedelta(hours=2),
        activity_type='deep_work',
        description='Code review',
        priority=3,
    )

    assert block.activity_type == 'deep_work'
    assert block.priority == 3


def test_schedule_recommendation_creation():
    """Test ScheduleRecommendation creation."""
    rec = ScheduleRecommendation(
        recommendation_type='deep_work',
        message='Schedule deep work in morning',
        confidence=0.8,
    )

    assert rec.recommendation_type == 'deep_work'
    assert rec.confidence == 0.8


def test_scheduler_creation():
    """Test PredictiveScheduler creation."""
    scheduler = PredictiveScheduler()

    assert scheduler is not None


def test_predict_duration():
    """Test duration prediction."""
    scheduler = PredictiveScheduler()

    bug_duration = scheduler.predict_duration('Fix critical bug')
    feature_duration = scheduler.predict_duration('Implement new feature')

    assert bug_duration > 0
    assert feature_duration > 0
    assert feature_duration > bug_duration  # Features typically take longer
