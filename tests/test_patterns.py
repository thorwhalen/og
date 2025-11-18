"""Tests for pattern detection and alerts."""

import pytest
from datetime import datetime

from og.base import Observation
from og.patterns import PatternDetector, Pattern, Alert


@pytest.fixture
def sample_observations():
    """Create sample observations for testing."""
    obs_list = []

    # Create many app switches (should trigger pattern)
    for i in range(15):
        obs = Observation(
            timestamp=datetime.now(),
            observer_name='test',
            event_type='app_switch',
            data={'from_app': 'App A', 'to_app': f'App {i}'},
        )
        obs_list.append(obs)

    return obs_list


def test_pattern_detector_init():
    """Test PatternDetector initialization."""
    detector = PatternDetector()

    assert detector is not None
    assert len(detector.patterns) > 0  # Should have default patterns


def test_add_pattern():
    """Test adding custom patterns."""
    detector = PatternDetector()

    # Define custom pattern
    def always_true(obs):
        return True

    pattern = Pattern(
        name='test_pattern',
        description='Test pattern',
        condition=always_true,
        alert_message='Test alert',
    )

    detector.add_pattern(pattern)

    assert 'test_pattern' in detector.patterns


def test_pattern_detection(sample_observations):
    """Test pattern detection on observations."""
    detector = PatternDetector()

    # Add observations one by one
    alerts = []
    for obs in sample_observations:
        new_alerts = detector.add_observation(obs)
        alerts.extend(new_alerts)

    # Should have triggered the deep focus broken pattern
    assert len(alerts) > 0


def test_enable_disable_pattern():
    """Test enabling/disabling patterns."""
    detector = PatternDetector()

    # Disable a pattern
    pattern_name = list(detector.patterns.keys())[0]
    detector.disable_pattern(pattern_name)

    assert not detector.patterns[pattern_name].enabled

    # Re-enable
    detector.enable_pattern(pattern_name)

    assert detector.patterns[pattern_name].enabled


def test_get_alerts():
    """Test retrieving alerts."""
    detector = PatternDetector()

    # Create a pattern that always triggers
    def always_match(obs):
        return len(obs) > 0

    pattern = Pattern(
        name='always',
        description='Always triggers',
        condition=always_match,
        alert_message='Alert!',
        cooldown=0,  # No cooldown for testing
    )

    detector.add_pattern(pattern)

    # Add observation
    obs = Observation(
        timestamp=datetime.now(),
        observer_name='test',
        event_type='test',
        data={},
    )
    detector.add_observation(obs)

    # Get alerts
    alerts = detector.get_alerts()
    assert len(alerts) > 0


def test_pattern_stats():
    """Test pattern statistics."""
    detector = PatternDetector()

    stats = detector.get_pattern_stats()

    assert 'total_patterns' in stats
    assert 'enabled_patterns' in stats
    assert 'total_alerts' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
