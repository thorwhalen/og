"""Tests for context management."""

import pytest
from datetime import datetime

from og.context import ContextManager, Context
from og.base import Observation


def test_context_manager_init():
    """Test ContextManager initialization."""
    manager = ContextManager()

    assert manager is not None
    assert len(manager.contexts) == 0


def test_create_context():
    """Test creating a context."""
    manager = ContextManager()

    context = manager.create_context(
        name='test_project',
        description='Test project',
        keywords=['test', 'project'],
        repos=['test-repo'],
    )

    assert context.name == 'test_project'
    assert 'test_project' in manager.contexts


def test_switch_context():
    """Test switching contexts."""
    manager = ContextManager()

    # Create two contexts
    ctx1 = manager.create_context('project1')
    ctx2 = manager.create_context('project2')

    # Switch to project1
    manager.switch_context('project1')

    assert manager.current_context == 'project1'
    assert ctx1.active
    assert not ctx2.active

    # Switch to project2
    manager.switch_context('project2')

    assert manager.current_context == 'project2'
    assert not ctx1.active
    assert ctx2.active


def test_delete_context():
    """Test deleting a context."""
    manager = ContextManager()

    manager.create_context('temp_project')
    assert 'temp_project' in manager.contexts

    result = manager.delete_context('temp_project')

    assert result
    assert 'temp_project' not in manager.contexts


def test_add_observation_to_context():
    """Test adding observations to a context."""
    manager = ContextManager()

    context = manager.create_context('test')

    obs = Observation(
        timestamp=datetime.now(),
        observer_name='test',
        event_type='test',
        data={},
    )

    manager.add_observation_to_context(obs, 'test')

    assert len(manager.context_observations['test']) == 1


def test_get_context_summary():
    """Test getting context summary."""
    manager = ContextManager()

    manager.create_context('test')

    # Add some observations
    for i in range(5):
        obs = Observation(
            timestamp=datetime.now(),
            observer_name='test',
            event_type='test_event',
            data={'index': i},
        )
        manager.add_observation_to_context(obs, 'test')

    summary = manager.get_context_summary('test')

    assert summary['context'] == 'test'
    assert summary['total_observations'] == 5


def test_merge_contexts():
    """Test merging contexts."""
    manager = ContextManager()

    ctx1 = manager.create_context(
        'proj1',
        keywords=['keyword1'],
        repos=['repo1'],
    )
    ctx2 = manager.create_context(
        'proj2',
        keywords=['keyword2'],
        repos=['repo2'],
    )

    merged = manager.merge_contexts('proj1', 'proj2', 'merged_project')

    assert 'merged_project' in manager.contexts
    assert 'keyword1' in merged.keywords
    assert 'keyword2' in merged.keywords
    assert 'repo1' in merged.repos
    assert 'repo2' in merged.repos


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
