"""Cross-project intelligence and code reuse detection.

Identifies reusable patterns, similar problems, and suggests code/approaches
from past projects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re


@dataclass
class CodePattern:
    """A reusable code pattern."""

    pattern_type: str
    code_snippet: str
    projects: List[str]
    usage_count: int
    last_used: datetime
    similarity_score: float = 1.0


@dataclass
class SimilarProblem:
    """A similar problem solved in another project."""

    problem_description: str
    solution_project: str
    solution_file: str
    solution_date: datetime
    similarity_score: float
    tags: List[str] = field(default_factory=list)


class CrossProjectAnalyzer:
    """Analyzes patterns across projects."""

    def __init__(self, og_instance=None):
        self.og = og_instance
        self._pattern_database: Dict[str, CodePattern] = {}

    def find_similar_solutions(self, current_problem: str) -> List[SimilarProblem]:
        """Find similar problems from past projects."""
        if not self.og:
            return []

        similar = []

        # Get all git commits
        commits = self.og.recent_activity(days=365, event_type='git_commit')

        # Extract keywords from current problem
        keywords = set(re.findall(r'\w+', current_problem.lower()))

        for commit in commits:
            message = commit.data.get('message', '').lower()
            commit_keywords = set(re.findall(r'\w+', message))

            # Calculate similarity
            common_keywords = keywords & commit_keywords
            if common_keywords:
                similarity = len(common_keywords) / len(keywords | commit_keywords)

                if similarity > 0.3:
                    similar.append(SimilarProblem(
                        problem_description=commit.data.get('message', ''),
                        solution_project=commit.data.get('repo', 'unknown'),
                        solution_file=commit.data.get('files', [''])[0] if commit.data.get('files') else '',
                        solution_date=commit.timestamp,
                        similarity_score=similarity,
                        tags=list(common_keywords),
                    ))

        return sorted(similar, key=lambda x: x.similarity_score, reverse=True)[:10]

    def detect_reusable_patterns(self) -> List[CodePattern]:
        """Detect reusable code patterns across projects."""
        patterns = []

        # Analyze file modifications for common patterns
        # Simplified implementation - in production would use AST analysis

        return patterns

    def suggest_from_past_work(self, context: str) -> List[str]:
        """Suggest approaches from past work."""
        similar = self.find_similar_solutions(context)

        suggestions = []
        for problem in similar[:3]:
            suggestions.append(
                f"You solved a similar problem in {problem.solution_project} "
                f"on {problem.solution_date.strftime('%Y-%m-%d')}"
            )

        return suggestions
