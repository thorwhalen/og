"""Task manager integration.

Bi-directional sync with Todoist, Linear, Jira, GitHub Issues, etc.
Auto-completes tasks when work is detected.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from enum import Enum


class TaskStatus(Enum):
    """Task status."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """A task from any task manager."""

    id: str
    title: str
    description: str
    status: TaskStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    project: Optional[str] = None
    source: str = "manual"  # 'todoist', 'linear', 'jira', 'github'


class TaskIntegration:
    """Integrates with various task management systems."""

    def __init__(self, og_instance=None):
        self.og = og_instance
        self._tasks: Dict[str, Task] = {}
        self._integrations = {
            'todoist': TodoistIntegration(),
            'linear': LinearIntegration(),
            'jira': JiraIntegration(),
            'github': GitHubIssuesIntegration(),
        }

    def sync_tasks(self, sources: List[str] = None):
        """Sync tasks from external systems."""
        if sources is None:
            sources = list(self._integrations.keys())

        for source in sources:
            if source in self._integrations:
                integration = self._integrations[source]
                tasks = integration.fetch_tasks()
                for task in tasks:
                    self._tasks[task.id] = task

    def auto_complete_tasks(self):
        """Auto-complete tasks based on detected work."""
        if not self.og:
            return

        # Get recent activity
        recent = self.og.recent_activity(hours=24)

        for task in self._tasks.values():
            if task.status == TaskStatus.TODO:
                # Check if work related to task was done
                if self._is_task_completed(task, recent):
                    self.complete_task(task.id)

    def _is_task_completed(self, task: Task, observations: List[Any]) -> bool:
        """Check if task was completed based on observations."""
        # Simple heuristic: check for commits/PRs related to task
        task_keywords = set(task.title.lower().split())

        for obs in observations:
            if obs.event_type == 'git_commit':
                message = obs.data.get('message', '').lower()
                if any(keyword in message for keyword in task_keywords):
                    return True

        return False

    def complete_task(self, task_id: str):
        """Mark task as completed."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # Calculate actual time
            if task.created_at and self.og:
                actual = self._calculate_actual_time(task)
                task.actual_hours = actual

            # Sync back to source
            source = task.source
            if source in self._integrations:
                self._integrations[source].update_task(task)

    def _calculate_actual_time(self, task: Task) -> float:
        """Calculate actual time spent on task."""
        # Get observations between task creation and completion
        observations = self.og.recent_activity(
            start_date=task.created_at,
            end_date=task.completed_at or datetime.now()
        )

        # Filter relevant observations
        relevant = []
        task_keywords = set(task.title.lower().split())

        for obs in observations:
            if obs.event_type in ['git_commit', 'file_modify']:
                # Check if observation is related to task
                text = str(obs.data).lower()
                if any(keyword in text for keyword in task_keywords):
                    relevant.append(obs)

        # Estimate time (simplified)
        if relevant:
            time_span = (relevant[-1].timestamp - relevant[0].timestamp).total_seconds() / 3600
            return min(time_span, 8)  # Cap at 8 hours per task

        return 0.0

    def generate_timesheet(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate timesheet for completed tasks."""
        completed = [
            task for task in self._tasks.values()
            if task.status == TaskStatus.COMPLETED
            and task.completed_at
            and start_date <= task.completed_at <= end_date
        ]

        return {
            'period_start': start_date,
            'period_end': end_date,
            'tasks': completed,
            'total_hours': sum(t.actual_hours or 0 for t in completed),
            'estimated_hours': sum(t.estimated_hours or 0 for t in completed),
        }


# Integration stubs (would implement with actual APIs)

class TodoistIntegration:
    """Todoist API integration."""

    def fetch_tasks(self) -> List[Task]:
        # Would use Todoist API
        return []

    def update_task(self, task: Task):
        # Would update via API
        pass


class LinearIntegration:
    """Linear API integration."""

    def fetch_tasks(self) -> List[Task]:
        return []

    def update_task(self, task: Task):
        pass


class JiraIntegration:
    """Jira API integration."""

    def fetch_tasks(self) -> List[Task]:
        return []

    def update_task(self, task: Task):
        pass


class GitHubIssuesIntegration:
    """GitHub Issues integration."""

    def fetch_tasks(self) -> List[Task]:
        return []

    def update_task(self, task: Task):
        pass
