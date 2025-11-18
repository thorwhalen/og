"""Git commit observer for OG (Own Ghost).

This observer tracks git commits in local repositories.
"""

import subprocess
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Optional

from og.base import Observation, PollingObserver


class GitCommitObserver(PollingObserver):
    """Observer for git commits.

    Tracks:
    - Commits in local repositories
    - Commit messages
    - Changed files
    - Branch information
    """

    def __init__(
        self,
        name: str = 'git',
        poll_interval: float = 300.0,  # 5 minutes
        enabled: bool = True,
        repo_paths: Optional[list[str]] = None,
        **config
    ):
        """Initialize git observer.

        Args:
            name: Observer name
            poll_interval: How often to check for commits (seconds)
            enabled: Whether observer is enabled
            repo_paths: Paths to git repositories to monitor
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)
        self.repo_paths = repo_paths or []
        self._last_commits: dict[str, str] = {}  # repo -> last commit hash

        # Auto-discover git repos if none specified
        if not self.repo_paths:
            self._discover_repos()

    def _discover_repos(self):
        """Auto-discover git repositories."""
        # Look for git repos in common locations
        search_paths = [
            Path.home() / 'code',
            Path.home() / 'projects',
            Path.home() / 'dev',
            Path.home() / 'src',
        ]

        for search_path in search_paths:
            if search_path.exists():
                # Find .git directories
                for git_dir in search_path.rglob('.git'):
                    if git_dir.is_dir():
                        repo_path = git_dir.parent
                        self.repo_paths.append(str(repo_path))

    def poll(self) -> list[Observation]:
        """Poll git repositories for new commits."""
        observations = []

        for repo_path in self.repo_paths:
            try:
                repo_obs = self._poll_repo(repo_path)
                observations.extend(repo_obs)
            except Exception as e:
                print(f"Error polling repo {repo_path}: {e}")

        return observations

    def _poll_repo(self, repo_path: str) -> list[Observation]:
        """Poll a specific repository for new commits.

        Args:
            repo_path: Path to git repository

        Returns:
            List of observations for new commits
        """
        observations = []

        try:
            # Get current HEAD commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return observations

            current_commit = result.stdout.strip()

            # Check if this is a new commit
            last_commit = self._last_commits.get(repo_path)

            if last_commit and last_commit != current_commit:
                # Get commits between last and current
                result = subprocess.run(
                    ['git', 'log', f'{last_commit}..{current_commit}', '--format=%H|%s|%an|%ae|%at'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            parts = line.split('|')
                            if len(parts) >= 5:
                                commit_hash, message, author_name, author_email, timestamp = parts[:5]

                                obs = self.create_observation(
                                    event_type='git_commit',
                                    data={
                                        'repo': repo_path,
                                        'commit_hash': commit_hash,
                                        'message': message,
                                        'author_name': author_name,
                                        'author_email': author_email,
                                    },
                                    metadata={
                                        'commit_time': datetime.fromtimestamp(int(timestamp)).isoformat(),
                                    },
                                    tags=['git', 'commit', 'code'],
                                )
                                observations.append(obs)

            # Update last commit
            self._last_commits[repo_path] = current_commit

        except Exception as e:
            print(f"Error checking repo {repo_path}: {e}")

        return observations
