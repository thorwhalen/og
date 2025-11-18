"""Learning trajectory tracking and skill development analysis.

This module tracks skill development over time, detects learning modes,
identifies knowledge gaps, and provides career development insights.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, Counter
import re


@dataclass
class Skill:
    """A tracked skill with proficiency and history."""

    name: str
    category: str  # 'language', 'framework', 'tool', 'concept'
    first_seen: datetime
    last_used: datetime
    usage_count: int = 0
    proficiency_score: float = 0.0  # 0-1 scale
    learning_velocity: float = 0.0  # Rate of improvement
    evidence: List[str] = field(default_factory=list)

    @property
    def days_since_first(self) -> int:
        """Days since first encountering this skill."""
        return (datetime.now() - self.first_seen).days

    @property
    def is_new(self) -> bool:
        """Check if skill is recently learned (< 30 days)."""
        return self.days_since_first < 30

    @property
    def is_active(self) -> bool:
        """Check if skill is actively being used (< 7 days)."""
        return (datetime.now() - self.last_used).days < 7


@dataclass
class LearningSession:
    """A detected learning session."""

    start: datetime
    end: datetime
    duration_minutes: float
    skills: List[str]
    mode: str  # 'exploration', 'tutorial', 'practice', 'production'
    resources: List[str] = field(default_factory=list)  # URLs, docs visited
    confidence: float = 1.0


@dataclass
class KnowledgeGap:
    """An identified knowledge gap."""

    skill: str
    gap_type: str  # 'missing', 'weak', 'outdated'
    evidence: List[str]
    severity: str = 'medium'  # 'low', 'medium', 'high'
    recommendation: Optional[str] = None


@dataclass
class LearningReport:
    """Comprehensive learning trajectory report."""

    period_start: datetime
    period_end: datetime
    skills: List[Skill]
    new_skills: List[Skill]
    improving_skills: List[Skill]
    declining_skills: List[Skill]
    learning_sessions: List[LearningSession]
    knowledge_gaps: List[KnowledgeGap]
    recommendations: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Learning Trajectory Report\n",
            f"**Period:** {self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}\n",
            "## Summary\n",
            f"- **Total skills tracked:** {len(self.skills)}",
            f"- **New skills learned:** {len(self.new_skills)}",
            f"- **Skills improving:** {len(self.improving_skills)}",
            f"- **Learning sessions:** {len(self.learning_sessions)}",
            "",
        ]

        # New skills
        if self.new_skills:
            lines.append("## 🎓 New Skills Learned\n")
            for skill in self.new_skills:
                lines.append(
                    f"- **{skill.name}** ({skill.category}) - "
                    f"Used {skill.usage_count} times, "
                    f"Proficiency: {skill.proficiency_score:.0%}"
                )
            lines.append("")

        # Improving skills
        if self.improving_skills:
            lines.append("## 📈 Skills Improving\n")
            for skill in self.improving_skills:
                lines.append(
                    f"- **{skill.name}** - "
                    f"Proficiency: {skill.proficiency_score:.0%} "
                    f"(+{skill.learning_velocity:.1%} velocity)"
                )
            lines.append("")

        # Declining skills
        if self.declining_skills:
            lines.append("## ⚠️ Skills Declining (Not Recently Used)\n")
            for skill in self.declining_skills:
                days_since = (datetime.now() - skill.last_used).days
                lines.append(
                    f"- **{skill.name}** - Last used {days_since} days ago"
                )
            lines.append("")

        # Knowledge gaps
        if self.knowledge_gaps:
            lines.append("## 🔍 Knowledge Gaps Identified\n")
            for gap in self.knowledge_gaps:
                lines.append(f"### {gap.skill}")
                lines.append(f"- **Type:** {gap.gap_type}")
                lines.append(f"- **Severity:** {gap.severity}")
                if gap.recommendation:
                    lines.append(f"- **Recommendation:** {gap.recommendation}")
                lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("## 💡 Recommendations\n")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Metrics
        if self.metrics:
            lines.append("## 📊 Detailed Metrics\n")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.2f}")
                else:
                    lines.append(f"- **{key}:** {value}")

        return "\n".join(lines)


class LearningTracker:
    """Tracks learning trajectory and skill development."""

    # Skill detection patterns
    LANGUAGE_PATTERNS = {
        'python': r'\b(?:python|\.py|pip|pytest|django|flask)\b',
        'javascript': r'\b(?:javascript|\.js|npm|node|react|vue)\b',
        'typescript': r'\b(?:typescript|\.ts|tsx)\b',
        'rust': r'\b(?:rust|\.rs|cargo)\b',
        'go': r'\b(?:golang|\.go)\b',
        'java': r'\b(?:java|\.java|maven|gradle)\b',
    }

    FRAMEWORK_PATTERNS = {
        'react': r'\b(?:react|jsx|create-react-app)\b',
        'vue': r'\b(?:vue|vuex|nuxt)\b',
        'django': r'\b(?:django|\.models\.py|manage\.py)\b',
        'flask': r'\b(?:flask|@app\.route)\b',
        'fastapi': r'\b(?:fastapi|pydantic)\b',
        'pytorch': r'\b(?:pytorch|torch|\.pth)\b',
        'tensorflow': r'\b(?:tensorflow|keras)\b',
    }

    TOOL_PATTERNS = {
        'git': r'\b(?:git|github|commit|push|pull|merge)\b',
        'docker': r'\b(?:docker|dockerfile|container)\b',
        'kubernetes': r'\b(?:kubernetes|k8s|kubectl)\b',
        'aws': r'\b(?:aws|ec2|s3|lambda)\b',
        'postgres': r'\b(?:postgres|postgresql|psql)\b',
    }

    # Learning mode indicators
    TUTORIAL_INDICATORS = [
        'tutorial', 'getting started', 'quickstart', 'introduction',
        'beginner', 'basics', 'learn', 'course'
    ]

    DOCUMENTATION_INDICATORS = [
        'docs', 'documentation', 'api reference', 'guide', 'manual'
    ]

    def __init__(self, og_instance=None):
        """Initialize the learning tracker.

        Args:
            og_instance: Optional OG instance for accessing observations
        """
        self.og = og_instance
        self._skill_database: Dict[str, Skill] = {}

    def track(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 30,
    ) -> LearningReport:
        """Track learning trajectory over a period.

        Args:
            start_date: Start of tracking period
            end_date: End of tracking period
            days: Number of days to track (if start_date not provided)

        Returns:
            LearningReport with analysis
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Get observations
        observations = self._get_observations(start_date, end_date)

        # Extract skills from observations
        self._extract_skills(observations)

        # Detect learning sessions
        learning_sessions = self._detect_learning_sessions(observations)

        # Identify knowledge gaps
        knowledge_gaps = self._identify_knowledge_gaps(observations)

        # Categorize skills
        new_skills = [s for s in self._skill_database.values() if s.is_new]
        improving_skills = [s for s in self._skill_database.values() if s.learning_velocity > 0.1]
        declining_skills = [
            s for s in self._skill_database.values()
            if (datetime.now() - s.last_used).days > 30
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(
            new_skills, improving_skills, declining_skills, knowledge_gaps
        )

        # Calculate metrics
        metrics = self._calculate_metrics(
            list(self._skill_database.values()),
            learning_sessions
        )

        return LearningReport(
            period_start=start_date,
            period_end=end_date,
            skills=list(self._skill_database.values()),
            new_skills=new_skills,
            improving_skills=improving_skills,
            declining_skills=declining_skills,
            learning_sessions=learning_sessions,
            knowledge_gaps=knowledge_gaps,
            recommendations=recommendations,
            metrics=metrics,
        )

    def _get_observations(
        self, start_date: datetime, end_date: datetime
    ) -> List[Any]:
        """Get observations for the period."""
        if not self.og:
            return []

        return self.og.recent_activity(start_date=start_date, end_date=end_date)

    def _extract_skills(self, observations: List[Any]):
        """Extract and track skills from observations."""
        for obs in observations:
            # Extract from commits
            if obs.event_type == 'git_commit':
                message = obs.data.get('message', '')
                files = obs.data.get('files', [])

                # Detect skills from commit message and files
                skills = self._detect_skills_in_text(message)
                skills.update(self._detect_skills_in_files(files))

                for skill_name, category in skills:
                    self._record_skill_usage(skill_name, category, obs.timestamp)

            # Extract from browser history (docs, tutorials)
            elif obs.event_type == 'browser_visit':
                url = obs.data.get('url', '')
                title = obs.data.get('title', '')

                skills = self._detect_skills_in_text(url + ' ' + title)

                for skill_name, category in skills:
                    self._record_skill_usage(skill_name, category, obs.timestamp)

            # Extract from file modifications
            elif obs.event_type in ['file_modify', 'file_create']:
                path = obs.data.get('path', '')
                skills = self._detect_skills_in_files([path])

                for skill_name, category in skills:
                    self._record_skill_usage(skill_name, category, obs.timestamp)

    def _detect_skills_in_text(self, text: str) -> Set[tuple]:
        """Detect skills mentioned in text."""
        skills = set()
        text_lower = text.lower()

        # Check languages
        for lang, pattern in self.LANGUAGE_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                skills.add((lang, 'language'))

        # Check frameworks
        for framework, pattern in self.FRAMEWORK_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                skills.add((framework, 'framework'))

        # Check tools
        for tool, pattern in self.TOOL_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                skills.add((tool, 'tool'))

        return skills

    def _detect_skills_in_files(self, files: List[str]) -> Set[tuple]:
        """Detect skills from file extensions and names."""
        skills = set()

        for file in files:
            file_lower = file.lower()

            # Language detection by extension
            if file_lower.endswith('.py'):
                skills.add(('python', 'language'))
            elif file_lower.endswith(('.js', '.jsx')):
                skills.add(('javascript', 'language'))
            elif file_lower.endswith(('.ts', '.tsx')):
                skills.add(('typescript', 'language'))
            elif file_lower.endswith('.rs'):
                skills.add(('rust', 'language'))
            elif file_lower.endswith('.go'):
                skills.add(('go', 'language'))
            elif file_lower.endswith('.java'):
                skills.add(('java', 'language'))

            # Framework detection
            if 'requirements.txt' in file_lower or 'setup.py' in file_lower:
                skills.add(('python', 'language'))
            if 'package.json' in file_lower:
                skills.add(('javascript', 'language'))
            if 'dockerfile' in file_lower:
                skills.add(('docker', 'tool'))

        return skills

    def _record_skill_usage(self, skill_name: str, category: str, timestamp: datetime):
        """Record usage of a skill."""
        if skill_name not in self._skill_database:
            self._skill_database[skill_name] = Skill(
                name=skill_name,
                category=category,
                first_seen=timestamp,
                last_used=timestamp,
                usage_count=1,
            )
        else:
            skill = self._skill_database[skill_name]
            skill.last_used = timestamp
            skill.usage_count += 1

        # Update proficiency score (simple heuristic)
        skill = self._skill_database[skill_name]
        days_experience = (timestamp - skill.first_seen).days + 1
        # Proficiency grows logarithmically with usage and time
        skill.proficiency_score = min(
            1.0,
            (skill.usage_count ** 0.5) / 20 + (days_experience ** 0.5) / 100
        )

        # Calculate learning velocity (rate of change in proficiency)
        if days_experience > 0:
            skill.learning_velocity = skill.proficiency_score / days_experience

    def _detect_learning_sessions(
        self, observations: List[Any]
    ) -> List[LearningSession]:
        """Detect learning sessions from observations."""
        sessions = []

        # Look for documentation/tutorial browsing patterns
        browser_obs = [o for o in observations if o.event_type == 'browser_visit']

        current_session = None

        for obs in sorted(browser_obs, key=lambda x: x.timestamp):
            url = obs.data.get('url', '').lower()
            title = obs.data.get('title', '').lower()

            # Check if this is a learning-related visit
            is_learning = any(
                indicator in url + title
                for indicator in self.TUTORIAL_INDICATORS + self.DOCUMENTATION_INDICATORS
            )

            if not is_learning:
                continue

            # Detect mode
            mode = 'exploration'
            if any(ind in url + title for ind in self.TUTORIAL_INDICATORS):
                mode = 'tutorial'
            elif any(ind in url + title for ind in self.DOCUMENTATION_INDICATORS):
                mode = 'practice'

            # Detect skills being learned
            skills_detected = self._detect_skills_in_text(url + ' ' + title)

            if current_session is None:
                current_session = {
                    'start': obs.timestamp,
                    'last': obs.timestamp,
                    'mode': mode,
                    'skills': set(s[0] for s in skills_detected),
                    'resources': [url],
                }
            else:
                time_gap = (obs.timestamp - current_session['last']).total_seconds() / 60

                # Same session if < 15 minutes gap
                if time_gap < 15:
                    current_session['last'] = obs.timestamp
                    current_session['skills'].update(s[0] for s in skills_detected)
                    current_session['resources'].append(url)
                else:
                    # Save previous session
                    if current_session['skills']:
                        duration = (current_session['last'] - current_session['start']).total_seconds() / 60
                        sessions.append(LearningSession(
                            start=current_session['start'],
                            end=current_session['last'],
                            duration_minutes=duration,
                            skills=list(current_session['skills']),
                            mode=current_session['mode'],
                            resources=current_session['resources'],
                            confidence=0.8,
                        ))

                    # Start new session
                    current_session = {
                        'start': obs.timestamp,
                        'last': obs.timestamp,
                        'mode': mode,
                        'skills': set(s[0] for s in skills_detected),
                        'resources': [url],
                    }

        # Save final session
        if current_session and current_session['skills']:
            duration = (current_session['last'] - current_session['start']).total_seconds() / 60
            sessions.append(LearningSession(
                start=current_session['start'],
                end=current_session['last'],
                duration_minutes=duration,
                skills=list(current_session['skills']),
                mode=current_session['mode'],
                resources=current_session['resources'],
                confidence=0.8,
            ))

        return sessions

    def _identify_knowledge_gaps(
        self, observations: List[Any]
    ) -> List[KnowledgeGap]:
        """Identify knowledge gaps based on activity patterns."""
        gaps = []

        # Look for repeated searches/documentation visits (indicates struggle)
        browser_obs = [o for o in observations if o.event_type == 'browser_visit']

        # Count documentation visits by skill
        doc_visits = defaultdict(list)

        for obs in browser_obs:
            url = obs.data.get('url', '').lower()
            title = obs.data.get('title', '').lower()

            if any(ind in url + title for ind in self.DOCUMENTATION_INDICATORS):
                skills = self._detect_skills_in_text(url + title)
                for skill_name, _ in skills:
                    doc_visits[skill_name].append(url)

        # Frequent doc visits indicate weak knowledge
        for skill, visits in doc_visits.items():
            if len(visits) > 10:  # Many documentation visits
                gaps.append(KnowledgeGap(
                    skill=skill,
                    gap_type='weak',
                    evidence=[f"{len(visits)} documentation visits"],
                    severity='medium',
                    recommendation=f"Consider a structured course or tutorial for {skill}",
                ))

        # Skills that haven't been used in production (only tutorials)
        learning_skills = set()
        for session in self._detect_learning_sessions(observations):
            learning_skills.update(session.skills)

        production_skills = set()
        for obs in observations:
            if obs.event_type == 'git_commit':
                skills = self._detect_skills_in_text(obs.data.get('message', ''))
                production_skills.update(s[0] for s in skills)

        # Skills learned but not used
        learned_not_used = learning_skills - production_skills
        for skill in learned_not_used:
            gaps.append(KnowledgeGap(
                skill=skill,
                gap_type='missing',
                evidence=["Tutorials viewed but no production usage"],
                severity='low',
                recommendation=f"Try building a small project with {skill}",
            ))

        return gaps

    def _generate_recommendations(
        self,
        new_skills: List[Skill],
        improving_skills: List[Skill],
        declining_skills: List[Skill],
        knowledge_gaps: List[KnowledgeGap],
    ) -> List[str]:
        """Generate learning recommendations."""
        recommendations = []

        # Encourage practicing new skills
        if new_skills:
            recommendations.append(
                f"You're learning {len(new_skills)} new skills. "
                "Practice them daily to build proficiency."
            )

        # Warn about declining skills
        if declining_skills:
            top_declining = declining_skills[0]
            recommendations.append(
                f"You haven't used {top_declining.name} in {(datetime.now() - top_declining.last_used).days} days. "
                "Consider a refresher project to maintain proficiency."
            )

        # Address knowledge gaps
        high_severity_gaps = [g for g in knowledge_gaps if g.severity == 'high']
        if high_severity_gaps:
            gap = high_severity_gaps[0]
            recommendations.append(gap.recommendation)

        # Suggest complementary skills
        if improving_skills:
            # Simple logic: if learning a language, suggest a framework
            for skill in improving_skills:
                if skill.category == 'language' and skill.name == 'python':
                    if 'django' not in self._skill_database and 'flask' not in self._skill_database:
                        recommendations.append(
                            "You're improving in Python. Consider learning a web framework like FastAPI or Flask."
                        )
                    break

        return recommendations

    def _calculate_metrics(
        self, skills: List[Skill], learning_sessions: List[LearningSession]
    ) -> Dict[str, Any]:
        """Calculate learning metrics."""
        if not skills:
            return {}

        total_learning_time = sum(s.duration_minutes for s in learning_sessions)

        metrics = {
            'total_learning_time_hours': total_learning_time / 60,
            'avg_proficiency': sum(s.proficiency_score for s in skills) / len(skills),
            'skills_by_category': Counter(s.category for s in skills),
            'most_used_skill': max(skills, key=lambda s: s.usage_count).name if skills else None,
            'fastest_learning_skill': max(skills, key=lambda s: s.learning_velocity).name if skills else None,
        }

        return metrics
