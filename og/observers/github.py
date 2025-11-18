"""GitHub activity observer for OG (Own Ghost).

This observer tracks GitHub activity using the GitHub API.
"""

import os
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Optional

from og.base import BaseObserver, Observation, PollingObserver


class GithubObserver(PollingObserver):
    """Observer for GitHub activity.

    Tracks:
    - Commits
    - Pull requests
    - Issues
    - Comments
    - Starred repositories
    """

    def __init__(
        self,
        name: str = 'github',
        poll_interval: float = 300.0,  # 5 minutes
        enabled: bool = True,
        username: Optional[str] = None,
        token: Optional[str] = None,
        **config
    ):
        """Initialize GitHub observer.

        Args:
            name: Observer name
            poll_interval: How often to poll GitHub API (seconds)
            enabled: Whether observer is enabled
            username: GitHub username (uses git config if None)
            token: GitHub API token (uses GITHUB_TOKEN env var if None)
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)
        self.username = username or self._get_git_username()
        self.token = token or os.getenv('GITHUB_TOKEN')
        self._last_check: Optional[datetime] = None

        # Lazy import to avoid requiring github package if not used
        self._github_client = None

    def _get_git_username(self) -> Optional[str]:
        """Try to get GitHub username from git config."""
        try:
            import subprocess

            result = subprocess.run(
                ['git', 'config', 'user.name'],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def _get_github_client(self):
        """Get GitHub client (lazy loading)."""
        if self._github_client is None:
            try:
                from github import Github

                if self.token:
                    self._github_client = Github(self.token)
                else:
                    self._github_client = Github()  # Anonymous access
            except ImportError:
                raise ImportError(
                    "PyGithub is required for GitHub observer. "
                    "Install it with: pip install PyGithub"
                )
        return self._github_client

    def poll(self) -> list[Observation]:
        """Poll GitHub for new activity."""
        if not self.username:
            print("GitHub username not configured, skipping poll")
            return []

        observations = []

        try:
            gh = self._get_github_client()
            user = gh.get_user(self.username)

            # Set time window for this poll
            now = datetime.now()
            since = self._last_check or (now - timedelta(hours=1))
            self._last_check = now

            # Get recent events
            events = user.get_events()

            for event in events:
                if event.created_at < since:
                    break  # Events are ordered by time, so we can stop

                obs = self._event_to_observation(event)
                if obs:
                    observations.append(obs)

        except Exception as e:
            print(f"Error polling GitHub: {e}")

        return observations

    def _event_to_observation(self, event) -> Optional[Observation]:
        """Convert a GitHub event to an observation.

        Args:
            event: GitHub event object

        Returns:
            Observation or None if event type not tracked
        """
        event_type_map = {
            'PushEvent': self._handle_push_event,
            'PullRequestEvent': self._handle_pr_event,
            'IssuesEvent': self._handle_issue_event,
            'IssueCommentEvent': self._handle_comment_event,
            'WatchEvent': self._handle_star_event,
        }

        handler = event_type_map.get(event.type)
        if handler:
            return handler(event)

        return None

    def _handle_push_event(self, event) -> Observation:
        """Handle push event (commits)."""
        commits = event.payload.get('commits', [])
        return self.create_observation(
            event_type='github_push',
            data={
                'repo': event.repo.name,
                'ref': event.payload.get('ref'),
                'commits': [
                    {
                        'sha': c.get('sha'),
                        'message': c.get('message'),
                        'author': c.get('author', {}).get('name'),
                    }
                    for c in commits
                ],
                'commit_count': len(commits),
            },
            metadata={
                'event_id': event.id,
                'created_at': event.created_at.isoformat(),
            },
            tags=['github', 'commit', 'push'],
        )

    def _handle_pr_event(self, event) -> Observation:
        """Handle pull request event."""
        pr = event.payload.get('pull_request', {})
        return self.create_observation(
            event_type='github_pull_request',
            data={
                'repo': event.repo.name,
                'action': event.payload.get('action'),
                'pr_number': pr.get('number'),
                'pr_title': pr.get('title'),
                'pr_url': pr.get('html_url'),
            },
            metadata={
                'event_id': event.id,
                'created_at': event.created_at.isoformat(),
            },
            tags=['github', 'pr', event.payload.get('action')],
        )

    def _handle_issue_event(self, event) -> Observation:
        """Handle issue event."""
        issue = event.payload.get('issue', {})
        return self.create_observation(
            event_type='github_issue',
            data={
                'repo': event.repo.name,
                'action': event.payload.get('action'),
                'issue_number': issue.get('number'),
                'issue_title': issue.get('title'),
                'issue_url': issue.get('html_url'),
            },
            metadata={
                'event_id': event.id,
                'created_at': event.created_at.isoformat(),
            },
            tags=['github', 'issue', event.payload.get('action')],
        )

    def _handle_comment_event(self, event) -> Observation:
        """Handle comment event."""
        comment = event.payload.get('comment', {})
        issue = event.payload.get('issue', {})

        return self.create_observation(
            event_type='github_comment',
            data={
                'repo': event.repo.name,
                'action': event.payload.get('action'),
                'comment_body': comment.get('body'),
                'issue_number': issue.get('number'),
                'issue_title': issue.get('title'),
            },
            metadata={
                'event_id': event.id,
                'created_at': event.created_at.isoformat(),
            },
            tags=['github', 'comment'],
        )

    def _handle_star_event(self, event) -> Observation:
        """Handle star/watch event."""
        return self.create_observation(
            event_type='github_star',
            data={
                'repo': event.repo.name,
                'action': event.payload.get('action'),
            },
            metadata={
                'event_id': event.id,
                'created_at': event.created_at.isoformat(),
            },
            tags=['github', 'star'],
        )
