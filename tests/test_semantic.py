"""Tests for semantic search functionality."""

import pytest
from datetime import datetime

from og.base import Observation
from og.semantic import SemanticMemory


@pytest.fixture
def sample_observations():
    """Create sample observations for testing."""
    return [
        Observation(
            timestamp=datetime.now(),
            observer_name='test',
            event_type='git_commit',
            data={'message': 'Implement machine learning model', 'repo': 'ml-project'},
            tags=['code', 'ml'],
        ),
        Observation(
            timestamp=datetime.now(),
            observer_name='test',
            event_type='git_commit',
            data={'message': 'Fix bug in data processing', 'repo': 'ml-project'},
            tags=['code', 'bugfix'],
        ),
        Observation(
            timestamp=datetime.now(),
            observer_name='test',
            event_type='browser_visit',
            data={'title': 'Python Machine Learning Tutorial', 'url': 'example.com'},
            tags=['learning', 'ml'],
        ),
    ]


def test_semantic_memory_init():
    """Test SemanticMemory initialization."""
    try:
        memory = SemanticMemory()
        assert memory is not None
        assert memory.collection is not None
    except ImportError:
        pytest.skip("chromadb not installed")


def test_add_observation(sample_observations):
    """Test adding observations to semantic memory."""
    try:
        memory = SemanticMemory()

        # Add single observation
        memory.add_observation(sample_observations[0])

        # Check it was added
        stats = memory.get_stats()
        assert stats['total_observations'] == 1

    except ImportError:
        pytest.skip("chromadb not installed")


def test_add_observations_batch(sample_observations):
    """Test adding multiple observations at once."""
    try:
        memory = SemanticMemory()

        # Add batch
        memory.add_observations(sample_observations)

        # Check all were added
        stats = memory.get_stats()
        assert stats['total_observations'] == len(sample_observations)

    except ImportError:
        pytest.skip("chromadb not installed")


def test_semantic_search(sample_observations):
    """Test semantic search functionality."""
    try:
        memory = SemanticMemory()
        memory.add_observations(sample_observations)

        # Search for machine learning content
        results = memory.search("machine learning", n_results=5)

        assert 'documents' in results
        assert len(results['documents']) > 0

        # Should find ML-related observations
        assert len(results['documents'][0]) > 0

    except ImportError:
        pytest.skip("chromadb not installed")


def test_clear(sample_observations):
    """Test clearing semantic memory."""
    try:
        memory = SemanticMemory()
        memory.add_observations(sample_observations)

        # Clear
        memory.clear()

        # Check empty
        stats = memory.get_stats()
        assert stats['total_observations'] == 0

    except ImportError:
        pytest.skip("chromadb not installed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
