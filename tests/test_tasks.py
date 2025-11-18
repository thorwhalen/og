"""Tests for task integration."""

import pytest
from datetime import datetime
from og.tasks import TaskIntegration, Task, TaskStatus


def test_task_creation():
    """Test Task creation."""
    task = Task(
        id='task-1',
        title='Implement feature X',
        description='Add new feature',
        status=TaskStatus.TODO,
        created_at=datetime.now(),
        source='linear',
    )

    assert task.id == 'task-1'
    assert task.status == TaskStatus.TODO


def test_task_integration_creation():
    """Test TaskIntegration creation."""
    integration = TaskIntegration()

    assert integration is not None
    assert len(integration._integrations) > 0
