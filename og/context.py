"""Context and project management for OG.

This module provides automatic context detection and tracking,
allowing OG to understand which projects or work contexts you're in.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from og.base import Observation


@dataclass
class Context:
    """A work context or project."""

    name: str
    description: str
    created_at: datetime
    keywords: list[str] = field(default_factory=list)
    repos: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    auto_detected: bool = False
    active: bool = False


class ContextManager:
    """Manages work contexts and projects.

    Automatically detects contexts based on activity patterns
    and allows explicit context switching.
    """

    def __init__(self):
        """Initialize context manager."""
        self.contexts: dict[str, Context] = {}
        self.current_context: Optional[str] = None
        self.context_history: list[tuple[datetime, str]] = []

        # Observations by context
        self.context_observations: dict[str, list[Observation]] = defaultdict(list)

        # Auto-detection settings
        self.auto_detect_enabled = True
        self.detection_threshold = 5  # Observations needed to detect context

    def create_context(
        self,
        name: str,
        description: str = '',
        keywords: Optional[list[str]] = None,
        repos: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
    ) -> Context:
        """Create a new context.

        Args:
            name: Context name
            description: Context description
            keywords: Keywords associated with this context
            repos: Repository paths or names
            tags: Tags for categorization

        Returns:
            Created context

        Example:
            >>> manager = ContextManager()
            >>> ctx = manager.create_context(
            ...     name='og_project',
            ...     description='Working on Own Ghost',
            ...     repos=['og'],
            ...     keywords=['og', 'observer', 'ai']
            ... )
        """
        context = Context(
            name=name,
            description=description,
            created_at=datetime.now(),
            keywords=keywords or [],
            repos=repos or [],
            tags=tags or [],
        )

        self.contexts[name] = context
        return context

    def delete_context(self, name: str) -> bool:
        """Delete a context.

        Args:
            name: Context name

        Returns:
            True if deleted, False if not found
        """
        if name in self.contexts:
            del self.contexts[name]
            if self.current_context == name:
                self.current_context = None
            return True
        return False

    def switch_context(self, name: str) -> None:
        """Switch to a specific context.

        Args:
            name: Context to switch to

        Raises:
            ValueError: If context doesn't exist
        """
        if name not in self.contexts:
            raise ValueError(f"Context '{name}' does not exist")

        # Deactivate current context
        if self.current_context:
            self.contexts[self.current_context].active = False

        # Activate new context
        self.current_context = name
        self.contexts[name].active = True

        # Record in history
        self.context_history.append((datetime.now(), name))

    def get_current_context(self) -> Optional[Context]:
        """Get the current active context.

        Returns:
            Current context or None
        """
        if self.current_context:
            return self.contexts[self.current_context]
        return None

    def add_observation_to_context(self, obs: Observation, context_name: str) -> None:
        """Add an observation to a specific context.

        Args:
            obs: Observation to add
            context_name: Context to add to
        """
        if context_name in self.contexts:
            self.context_observations[context_name].append(obs)

    def auto_detect_context(self, observations: list[Observation]) -> Optional[str]:
        """Automatically detect context from observations.

        Args:
            observations: Recent observations to analyze

        Returns:
            Detected context name or None
        """
        if not self.auto_detect_enabled or not observations:
            return None

        # Score each context based on observation match
        scores = defaultdict(int)

        for obs in observations:
            for context_name, context in self.contexts.items():
                score = self._match_observation_to_context(obs, context)
                scores[context_name] += score

        # Get highest scoring context
        if scores:
            best_context = max(scores.items(), key=lambda x: x[1])
            if best_context[1] >= self.detection_threshold:
                return best_context[0]

        return None

    def cluster_into_contexts(
        self, observations: list[Observation], min_cluster_size: int = 10
    ) -> dict[str, list[Observation]]:
        """Automatically cluster observations into contexts.

        Uses simple keyword and repository-based clustering to
        identify potential contexts.

        Args:
            observations: Observations to cluster
            min_cluster_size: Minimum observations for a context

        Returns:
            Dictionary mapping detected context names to observations
        """
        # Group by repositories
        by_repo = defaultdict(list)
        by_keywords = defaultdict(list)

        for obs in observations:
            # Group by repository
            if obs.event_type in ['git_commit', 'github_push']:
                repo = obs.data.get('repo', 'unknown')
                by_repo[repo].append(obs)

            # Group by keywords in data
            for value in obs.data.values():
                if isinstance(value, str):
                    # Simple keyword extraction (could be improved with NLP)
                    words = value.lower().split()
                    for word in words[:5]:  # Top 5 words
                        if len(word) > 4:  # Skip short words
                            by_keywords[word].append(obs)

        # Create auto-detected contexts from clusters
        clusters = {}

        # From repos
        for repo, obs_list in by_repo.items():
            if len(obs_list) >= min_cluster_size:
                context_name = f"auto_{repo}"
                if context_name not in self.contexts:
                    self.create_context(
                        name=context_name,
                        description=f"Auto-detected context for {repo}",
                        repos=[repo],
                    )
                    self.contexts[context_name].auto_detected = True

                clusters[context_name] = obs_list

        return clusters

    def get_context_summary(
        self, context_name: str, days: Optional[int] = None
    ) -> dict:
        """Get a summary of activity in a specific context.

        Args:
            context_name: Context to summarize
            days: Optional number of days to include

        Returns:
            Summary dictionary
        """
        if context_name not in self.contexts:
            raise ValueError(f"Context '{context_name}' not found")

        observations = self.context_observations.get(context_name, [])

        # Filter by days if specified
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            observations = [obs for obs in observations if obs.timestamp >= cutoff]

        # Compute statistics
        event_types = Counter(obs.event_type for obs in observations)
        observers = Counter(obs.observer_name for obs in observations)

        # Time range
        if observations:
            start_time = min(obs.timestamp for obs in observations)
            end_time = max(obs.timestamp for obs in observations)
            duration = end_time - start_time
        else:
            start_time = end_time = None
            duration = timedelta(0)

        return {
            'context': context_name,
            'total_observations': len(observations),
            'event_types': dict(event_types),
            'observers': dict(observers),
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'duration_hours': duration.total_seconds() / 3600,
        }

    def get_all_contexts(self) -> list[Context]:
        """Get all contexts.

        Returns:
            List of all contexts
        """
        return list(self.contexts.values())

    def get_active_contexts(self) -> list[Context]:
        """Get all active contexts.

        Returns:
            List of active contexts
        """
        return [ctx for ctx in self.contexts.values() if ctx.active]

    def get_context_timeline(self, days: int = 7) -> list[dict]:
        """Get timeline of context switches.

        Args:
            days: Number of days to include

        Returns:
            List of context switch events
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent_history = [
            (timestamp, ctx_name)
            for timestamp, ctx_name in self.context_history
            if timestamp >= cutoff
        ]

        timeline = []
        for i, (timestamp, ctx_name) in enumerate(recent_history):
            # Calculate duration in this context
            if i < len(recent_history) - 1:
                next_timestamp = recent_history[i + 1][0]
                duration = (next_timestamp - timestamp).total_seconds()
            else:
                duration = (datetime.now() - timestamp).total_seconds()

            timeline.append(
                {
                    'context': ctx_name,
                    'start': timestamp.isoformat(),
                    'duration_minutes': duration / 60,
                }
            )

        return timeline

    def suggest_contexts(self, observations: list[Observation]) -> list[dict]:
        """Suggest new contexts based on observation patterns.

        Args:
            observations: Observations to analyze

        Returns:
            List of suggested contexts with scores
        """
        suggestions = []

        # Analyze repositories
        repos = Counter()
        for obs in observations:
            if obs.event_type in ['git_commit', 'github_push']:
                repo = obs.data.get('repo')
                if repo:
                    repos[repo] += 1

        # Suggest contexts for frequently used repos
        for repo, count in repos.most_common(5):
            if count >= 10:  # At least 10 observations
                # Check if context already exists
                existing = any(
                    repo in ctx.repos for ctx in self.contexts.values()
                )
                if not existing:
                    suggestions.append(
                        {
                            'name': f'context_{repo.split("/")[-1]}',
                            'description': f'Work on {repo}',
                            'repos': [repo],
                            'confidence': min(count / 100, 1.0),
                        }
                    )

        return suggestions

    def _match_observation_to_context(
        self, obs: Observation, context: Context
    ) -> int:
        """Score how well an observation matches a context.

        Args:
            obs: Observation to match
            context: Context to match against

        Returns:
            Match score (higher is better)
        """
        score = 0

        # Check repository match
        if obs.event_type in ['git_commit', 'github_push']:
            repo = obs.data.get('repo', '')
            if any(ctx_repo in repo for ctx_repo in context.repos):
                score += 5

        # Check keyword match
        obs_text = str(obs.data).lower()
        for keyword in context.keywords:
            if keyword.lower() in obs_text:
                score += 2

        # Check tag match
        for tag in context.tags:
            if tag in obs.tags:
                score += 1

        return score

    def merge_contexts(self, context1: str, context2: str, new_name: str) -> Context:
        """Merge two contexts into one.

        Args:
            context1: First context name
            context2: Second context name
            new_name: Name for merged context

        Returns:
            New merged context
        """
        if context1 not in self.contexts or context2 not in self.contexts:
            raise ValueError("One or both contexts not found")

        ctx1 = self.contexts[context1]
        ctx2 = self.contexts[context2]

        # Create merged context
        merged = Context(
            name=new_name,
            description=f"Merged: {ctx1.description} + {ctx2.description}",
            created_at=datetime.now(),
            keywords=list(set(ctx1.keywords + ctx2.keywords)),
            repos=list(set(ctx1.repos + ctx2.repos)),
            tags=list(set(ctx1.tags + ctx2.tags)),
        )

        self.contexts[new_name] = merged

        # Merge observations
        self.context_observations[new_name] = (
            self.context_observations.get(context1, [])
            + self.context_observations.get(context2, [])
        )

        return merged
