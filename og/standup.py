"""Automated standup/status generation from observations.

This module provides automatic generation of daily standup updates, status reports,
and progress summaries based on observed activity.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re


@dataclass
class StandupItem:
    """A single standup item (completed work, planned work, or blocker)."""

    category: str  # 'completed', 'planned', 'blocker'
    description: str
    context: Optional[str] = None
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)


@dataclass
class Standup:
    """A complete standup report."""

    date: datetime
    yesterday: List[StandupItem]
    today: List[StandupItem]
    blockers: List[StandupItem]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_text(self, format: str = 'simple') -> str:
        """Generate text representation."""
        if format == 'slack':
            return self._to_slack()
        elif format == 'markdown':
            return self._to_markdown()
        else:
            return self._to_simple()

    def _to_simple(self) -> str:
        """Simple text format."""
        lines = []

        if self.yesterday:
            lines.append("Yesterday:")
            for item in self.yesterday:
                context = f" ({item.context})" if item.context else ""
                lines.append(f"  • {item.description}{context}")

        if self.today:
            lines.append("\nToday:")
            for item in self.today:
                context = f" ({item.context})" if item.context else ""
                lines.append(f"  • {item.description}{context}")

        if self.blockers:
            lines.append("\nBlockers:")
            for item in self.blockers:
                lines.append(f"  • {item.description}")

        if self.metrics:
            lines.append("\nMetrics:")
            for key, value in self.metrics.items():
                lines.append(f"  • {key}: {value}")

        return "\n".join(lines)

    def _to_slack(self) -> str:
        """Slack-formatted text."""
        lines = []

        if self.yesterday:
            lines.append("*Yesterday:*")
            for item in self.yesterday:
                context = f" _{item.context}_" if item.context else ""
                lines.append(f"• {item.description}{context}")

        if self.today:
            lines.append("\n*Today:*")
            for item in self.today:
                context = f" _{item.context}_" if item.context else ""
                lines.append(f"• {item.description}{context}")

        if self.blockers:
            lines.append("\n*Blockers:*")
            for item in self.blockers:
                lines.append(f"• {item.description}")

        return "\n".join(lines)

    def _to_markdown(self) -> str:
        """Markdown format."""
        lines = [f"# Standup - {self.date.strftime('%Y-%m-%d')}\n"]

        if self.yesterday:
            lines.append("## Yesterday")
            for item in self.yesterday:
                context = f" *({item.context})*" if item.context else ""
                lines.append(f"- {item.description}{context}")
            lines.append("")

        if self.today:
            lines.append("## Today")
            for item in self.today:
                context = f" *({item.context})*" if item.context else ""
                lines.append(f"- {item.description}{context}")
            lines.append("")

        if self.blockers:
            lines.append("## Blockers")
            for item in self.blockers:
                lines.append(f"- {item.description}")
            lines.append("")

        if self.metrics:
            lines.append("## Metrics")
            for key, value in self.metrics.items():
                lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)


class StandupGenerator:
    """Generates automated standup reports from observations."""

    def __init__(self, og_instance=None):
        """Initialize the standup generator.

        Args:
            og_instance: Optional OG instance for accessing observations
        """
        self.og = og_instance
        self._work_verbs = {
            'git_commit': ['Fixed', 'Implemented', 'Added', 'Updated', 'Refactored'],
            'github_pr': ['Reviewed', 'Opened', 'Merged'],
            'github_issue': ['Closed', 'Opened', 'Commented on'],
            'file_create': ['Created'],
            'file_modify': ['Modified', 'Updated'],
        }

    def generate(
        self,
        date: Optional[datetime] = None,
        include_metrics: bool = True,
        predict_today: bool = True,
    ) -> Standup:
        """Generate a standup report.

        Args:
            date: Date for the standup (default: today)
            include_metrics: Include productivity metrics
            predict_today: Predict today's work from patterns

        Returns:
            Standup object
        """
        if date is None:
            date = datetime.now()

        # Get yesterday's observations
        yesterday_start = (date - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        yesterday_end = yesterday_start + timedelta(days=1)

        yesterday_items = self._extract_work_items(
            yesterday_start, yesterday_end, 'completed'
        )

        # Get today's observations (partial day)
        today_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = date

        today_items = self._extract_work_items(today_start, today_end, 'completed')

        # Predict remaining work for today
        if predict_today:
            predicted_items = self._predict_today_work(date)
            today_items.extend(predicted_items)

        # Extract blockers
        blockers = self._extract_blockers(yesterday_start, date)

        # Calculate metrics
        metrics = {}
        if include_metrics:
            metrics = self._calculate_metrics(yesterday_start, yesterday_end)

        return Standup(
            date=date,
            yesterday=yesterday_items,
            today=today_items,
            blockers=blockers,
            metrics=metrics,
        )

    def _extract_work_items(
        self, start: datetime, end: datetime, category: str
    ) -> List[StandupItem]:
        """Extract work items from observations."""
        if not self.og:
            return []

        observations = self.og.recent_activity(
            start_date=start, end_date=end
        )

        items = []

        # Group by event type
        by_type = defaultdict(list)
        for obs in observations:
            by_type[obs.event_type].append(obs)

        # Process git commits
        if 'git_commit' in by_type:
            commits = by_type['git_commit']
            # Group by repo
            by_repo = defaultdict(list)
            for commit in commits:
                repo = commit.data.get('repo', 'unknown')
                by_repo[repo].append(commit)

            for repo, repo_commits in by_repo.items():
                if len(repo_commits) == 1:
                    msg = repo_commits[0].data.get('message', '').split('\n')[0]
                    items.append(StandupItem(
                        category=category,
                        description=msg,
                        context=repo,
                        evidence=[f"commit: {repo_commits[0].data.get('hash', 'unknown')[:7]}"]
                    ))
                else:
                    # Summarize multiple commits
                    items.append(StandupItem(
                        category=category,
                        description=f"Made {len(repo_commits)} commits",
                        context=repo,
                        evidence=[f"{len(repo_commits)} commits"]
                    ))

        # Process PRs
        if 'github_pr_opened' in by_type:
            for pr in by_type['github_pr_opened']:
                items.append(StandupItem(
                    category=category,
                    description=f"Opened PR: {pr.data.get('title', 'untitled')}",
                    context=pr.data.get('repo'),
                    evidence=[f"PR #{pr.data.get('number')}"]
                ))

        if 'github_pr_merged' in by_type:
            for pr in by_type['github_pr_merged']:
                items.append(StandupItem(
                    category=category,
                    description=f"Merged PR: {pr.data.get('title', 'untitled')}",
                    context=pr.data.get('repo'),
                    evidence=[f"PR #{pr.data.get('number')}"]
                ))

        # Process issues
        if 'github_issue_closed' in by_type:
            for issue in by_type['github_issue_closed']:
                items.append(StandupItem(
                    category=category,
                    description=f"Closed issue: {issue.data.get('title', 'untitled')}",
                    context=issue.data.get('repo'),
                    evidence=[f"Issue #{issue.data.get('number')}"]
                ))

        return items

    def _predict_today_work(self, date: datetime) -> List[StandupItem]:
        """Predict remaining work for today based on patterns."""
        if not self.og:
            return []

        items = []

        # Look for active contexts
        if hasattr(self.og, 'context_manager') and self.og.context_manager:
            current_context = self.og.context_manager.current_context
            if current_context:
                # Infer work from context
                items.append(StandupItem(
                    category='planned',
                    description=f"Continue work on {current_context.name}",
                    context=current_context.name,
                    confidence=0.7,
                ))

        # Look for recent file modifications (work in progress)
        recent = self.og.recent_activity(
            hours=4, event_type='file_modify'
        )

        if recent:
            # Group by directory/project
            files = [obs.data.get('path', '') for obs in recent[:5]]
            items.append(StandupItem(
                category='planned',
                description=f"Continue editing files in current project",
                confidence=0.6,
                evidence=[f"Recent edits: {len(recent)} files"]
            ))

        return items

    def _extract_blockers(
        self, start: datetime, end: datetime
    ) -> List[StandupItem]:
        """Extract potential blockers from observations."""
        if not self.og:
            return []

        blockers = []

        # Look for patterns that indicate blockers
        observations = self.og.recent_activity(start_date=start, end_date=end)

        # Check for excessive browser searches (confusion/blocked)
        searches = [
            obs for obs in observations
            if obs.event_type == 'browser_visit'
            and any(
                term in obs.data.get('url', '').lower()
                for term in ['stackoverflow', 'google.com/search', 'github.com/issues']
            )
        ]

        if len(searches) > 20:
            blockers.append(StandupItem(
                category='blocker',
                description="May be blocked - high volume of searches for solutions",
                confidence=0.6,
                evidence=[f"{len(searches)} search-related visits"]
            ))

        # Check for comments on issues (waiting for response)
        comments = [
            obs for obs in observations
            if obs.event_type == 'github_comment'
        ]

        if comments:
            for comment in comments[-3:]:  # Last 3 comments
                if any(
                    word in comment.data.get('body', '').lower()
                    for word in ['waiting', 'blocked', 'need', 'help']
                ):
                    blockers.append(StandupItem(
                        category='blocker',
                        description=f"Waiting on: {comment.data.get('title', 'response')}",
                        evidence=[f"Comment on {comment.data.get('repo')}"]
                    ))

        return blockers

    def _calculate_metrics(
        self, start: datetime, end: datetime
    ) -> Dict[str, Any]:
        """Calculate productivity metrics for the period."""
        if not self.og:
            return {}

        observations = self.og.recent_activity(start_date=start, end_date=end)

        metrics = {
            'total_observations': len(observations),
            'commits': len([o for o in observations if o.event_type == 'git_commit']),
            'files_modified': len([o for o in observations if o.event_type == 'file_modify']),
            'prs': len([o for o in observations if 'pr' in o.event_type]),
        }

        return metrics

    def generate_weekly(self, date: Optional[datetime] = None) -> str:
        """Generate a weekly status report.

        Args:
            date: End date for the week (default: today)

        Returns:
            Markdown formatted weekly report
        """
        if date is None:
            date = datetime.now()

        # Get Monday of the week
        week_start = date - timedelta(days=date.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        lines = [f"# Weekly Report - Week of {week_start.strftime('%Y-%m-%d')}\n"]

        # Generate standup for each day
        for i in range(5):  # Mon-Fri
            day = week_start + timedelta(days=i)
            standup = self.generate(date=day, predict_today=False)

            lines.append(f"## {day.strftime('%A, %B %d')}")
            lines.append("")

            if standup.yesterday:
                for item in standup.yesterday:
                    context = f" *({item.context})*" if item.context else ""
                    lines.append(f"- {item.description}{context}")
            else:
                lines.append("- No activity recorded")

            lines.append("")

        return "\n".join(lines)
