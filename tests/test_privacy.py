"""Tests for privacy controls."""

import pytest
from datetime import datetime

from og.privacy import PrivacyControls
from og.base import Observation


def test_privacy_controls_init():
    """Test PrivacyControls initialization."""
    privacy = PrivacyControls()

    assert privacy is not None
    assert not privacy.encryption_enabled
    assert privacy.retention_days is None


def test_redact_sensitive_data():
    """Test sensitive data redaction."""
    privacy = PrivacyControls()

    # Test email redaction
    text = "Contact me at user@example.com for details"
    redacted = privacy.redact_sensitive_data(text)

    assert '[REDACTED]' in redacted
    assert 'user@example.com' not in redacted

    # Test password redaction
    text2 = 'password: secretpass123'
    redacted2 = privacy.redact_sensitive_data(text2)

    assert '[REDACTED]' in redacted2


def test_anonymize_observation():
    """Test observation anonymization."""
    privacy = PrivacyControls()

    obs = Observation(
        timestamp=datetime.now(),
        observer_name='test',
        event_type='test',
        data={
            'message': 'Email: user@example.com',
            'user': 'john_doe',
        },
    )

    anon_obs = privacy.anonymize_observation(obs)

    # Email should be redacted
    assert '[REDACTED]' in anon_obs.data['message']

    # User should be hashed
    assert anon_obs.data['user'] != 'john_doe'


def test_exclude_patterns():
    """Test exclude patterns."""
    privacy = PrivacyControls()

    # Add exclude pattern for URLs
    privacy.add_exclude_pattern(r'bank\.com', 'url')

    # Test observation with excluded URL
    obs = Observation(
        timestamp=datetime.now(),
        observer_name='browser',
        event_type='browser_visit',
        data={'url': 'https://bank.com/login'},
    )

    assert privacy.should_exclude(obs)

    # Test observation with allowed URL
    obs2 = Observation(
        timestamp=datetime.now(),
        observer_name='browser',
        event_type='browser_visit',
        data={'url': 'https://example.com'},
    )

    assert not privacy.should_exclude(obs2)


def test_retention_policy():
    """Test retention policy."""
    privacy = PrivacyControls()

    privacy.set_retention_policy(30)

    assert privacy.retention_days == 30


def test_privacy_report():
    """Test privacy report generation."""
    privacy = PrivacyControls()

    observations = [
        Observation(
            timestamp=datetime.now(),
            observer_name='test',
            event_type='test_event',
            data={'field1': 'value1', 'email': 'test@example.com'},
        )
        for _ in range(5)
    ]

    report = privacy.generate_privacy_report(observations)

    assert report['total_observations'] == 5
    assert 'test' in report['observers']
    assert 'test_event' in report['event_types']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
