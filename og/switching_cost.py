"""Context switching cost analysis.

This module analyzes the cost of context switching, measures time lost to
interruptions, quantifies flow state vs fragmented work, and identifies
the biggest interrupters.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import statistics


@dataclass
class Switch:
    """A single context switch event."""

    timestamp: datetime
    from_context: str
    to_context: str
    switch_type: str  # 'app', 'project', 'repo', 'browser_tab'
    recovery_time: Optional[float] = None  # Estimated time to regain focus (minutes)


@dataclass
class FlowSession:
    """A period of focused work without interruptions."""

    start: datetime
    end: datetime
    duration_minutes: float
    context: str
    interruption_count: int = 0
    productivity_score: float = 1.0


@dataclass
class SwitchingReport:
    """Context switching analysis report."""

    period_start: datetime
    period_end: datetime
    total_switches: int
    switches_by_type: Dict[str, int]
    estimated_cost_minutes: float
    flow_sessions: List[FlowSession]
    fragmented_sessions: List[FlowSession]
    top_interrupters: List[Tuple[str, int]]
    recommendations: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Context Switching Cost Analysis\n",
            f"**Period:** {self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}\n",
            "## Summary\n",
            f"- **Total context switches:** {self.total_switches}",
            f"- **Estimated time lost:** {self.estimated_cost_minutes:.1f} minutes ({self.estimated_cost_minutes/60:.1f} hours)",
            f"- **Flow sessions:** {len(self.flow_sessions)}",
            f"- **Fragmented sessions:** {len(self.fragmented_sessions)}",
            "",
        ]

        # Switches by type
        lines.append("## Switches by Type\n")
        for switch_type, count in sorted(
            self.switches_by_type.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"- **{switch_type}:** {count}")
        lines.append("")

        # Top interrupters
        if self.top_interrupters:
            lines.append("## Top Interrupters\n")
            for interrupter, count in self.top_interrupters[:10]:
                lines.append(f"- **{interrupter}:** {count} times")
            lines.append("")

        # Flow sessions
        if self.flow_sessions:
            lines.append("## Flow Sessions (Deep Work)\n")
            total_flow = sum(s.duration_minutes for s in self.flow_sessions)
            lines.append(f"Total flow time: {total_flow:.1f} minutes ({total_flow/60:.1f} hours)\n")
            for session in self.flow_sessions[:5]:
                lines.append(
                    f"- {session.start.strftime('%Y-%m-%d %H:%M')} - "
                    f"{session.duration_minutes:.0f} min - {session.context}"
                )
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.append("## Recommendations\n")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Metrics
        if self.metrics:
            lines.append("## Detailed Metrics\n")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    lines.append(f"- **{key}:** {value:.2f}")
                else:
                    lines.append(f"- **{key}:** {value}")

        return "\n".join(lines)


class SwitchingCostAnalyzer:
    """Analyzes context switching costs and flow states."""

    # Recovery time estimates (minutes) for different switch types
    RECOVERY_TIMES = {
        'app': 5.0,  # Switching between applications
        'project': 15.0,  # Switching between projects/repos
        'browser_tab': 2.0,  # Browser tab switches
        'notification': 7.0,  # Interruption from notification
    }

    # Minimum duration for a flow session (minutes)
    FLOW_SESSION_MIN_DURATION = 30

    # Maximum gap between activities in same session (minutes)
    SESSION_GAP_THRESHOLD = 5

    def __init__(self, og_instance=None):
        """Initialize the analyzer.

        Args:
            og_instance: Optional OG instance for accessing observations
        """
        self.og = og_instance

    def analyze(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 7,
    ) -> SwitchingReport:
        """Analyze context switching costs.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            days: Number of days to analyze (if start_date not provided)

        Returns:
            SwitchingReport with analysis results
        """
        if end_date is None:
            end_date = datetime.now()

        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Get observations
        observations = self._get_observations(start_date, end_date)

        # Detect switches
        switches = self._detect_switches(observations)

        # Identify flow sessions
        flow_sessions, fragmented_sessions = self._identify_flow_sessions(observations)

        # Calculate costs
        total_cost = sum(
            self.RECOVERY_TIMES.get(s.switch_type, 5.0) for s in switches
        )

        # Analyze interrupters
        top_interrupters = self._identify_interrupters(observations, switches)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            switches, flow_sessions, fragmented_sessions, top_interrupters
        )

        # Calculate detailed metrics
        metrics = self._calculate_metrics(
            switches, flow_sessions, fragmented_sessions, observations
        )

        return SwitchingReport(
            period_start=start_date,
            period_end=end_date,
            total_switches=len(switches),
            switches_by_type=self._count_by_type(switches),
            estimated_cost_minutes=total_cost,
            flow_sessions=flow_sessions,
            fragmented_sessions=fragmented_sessions,
            top_interrupters=top_interrupters,
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

    def _detect_switches(self, observations: List[Any]) -> List[Switch]:
        """Detect context switches from observations."""
        switches = []

        # Track current context by type
        current_app = None
        current_repo = None
        current_tab = None

        for obs in sorted(observations, key=lambda x: x.timestamp):
            # App switches
            if obs.event_type == 'app_switch':
                new_app = obs.data.get('app')
                if current_app and new_app and new_app != current_app:
                    switches.append(Switch(
                        timestamp=obs.timestamp,
                        from_context=current_app,
                        to_context=new_app,
                        switch_type='app',
                        recovery_time=self.RECOVERY_TIMES['app']
                    ))
                current_app = new_app

            # Project/repo switches
            elif obs.event_type == 'git_commit':
                repo = obs.data.get('repo')
                if current_repo and repo and repo != current_repo:
                    switches.append(Switch(
                        timestamp=obs.timestamp,
                        from_context=current_repo,
                        to_context=repo,
                        switch_type='project',
                        recovery_time=self.RECOVERY_TIMES['project']
                    ))
                current_repo = repo

            # Browser tab switches (inferred from rapid page visits)
            elif obs.event_type == 'browser_visit':
                url = obs.data.get('url', '')
                if current_tab and url != current_tab:
                    switches.append(Switch(
                        timestamp=obs.timestamp,
                        from_context=current_tab[:50],
                        to_context=url[:50],
                        switch_type='browser_tab',
                        recovery_time=self.RECOVERY_TIMES['browser_tab']
                    ))
                current_tab = url

        return switches

    def _identify_flow_sessions(
        self, observations: List[Any]
    ) -> Tuple[List[FlowSession], List[FlowSession]]:
        """Identify flow sessions and fragmented sessions."""
        flow_sessions = []
        fragmented_sessions = []

        # Group observations by app/context
        sessions = []
        current_session = None

        for obs in sorted(observations, key=lambda x: x.timestamp):
            # Get context from observation
            context = self._get_context(obs)

            if current_session is None:
                current_session = {
                    'start': obs.timestamp,
                    'last': obs.timestamp,
                    'context': context,
                    'interruptions': 0,
                    'observations': [obs],
                }
            else:
                time_gap = (obs.timestamp - current_session['last']).total_seconds() / 60

                # Check if same session
                if time_gap < self.SESSION_GAP_THRESHOLD and context == current_session['context']:
                    current_session['last'] = obs.timestamp
                    current_session['observations'].append(obs)
                else:
                    # End current session, start new one
                    if time_gap < self.SESSION_GAP_THRESHOLD:
                        current_session['interruptions'] += 1

                    sessions.append(current_session)
                    current_session = {
                        'start': obs.timestamp,
                        'last': obs.timestamp,
                        'context': context,
                        'interruptions': 0,
                        'observations': [obs],
                    }

        if current_session:
            sessions.append(current_session)

        # Classify sessions as flow or fragmented
        for session in sessions:
            duration = (session['last'] - session['start']).total_seconds() / 60

            if duration < 1:  # Skip very short sessions
                continue

            flow_session = FlowSession(
                start=session['start'],
                end=session['last'],
                duration_minutes=duration,
                context=session['context'],
                interruption_count=session['interruptions'],
                productivity_score=max(0, 1 - session['interruptions'] * 0.1)
            )

            if duration >= self.FLOW_SESSION_MIN_DURATION and session['interruptions'] <= 2:
                flow_sessions.append(flow_session)
            else:
                fragmented_sessions.append(flow_session)

        return flow_sessions, fragmented_sessions

    def _get_context(self, obs: Any) -> str:
        """Extract context from observation."""
        if obs.event_type == 'app_switch':
            return obs.data.get('app', 'unknown')
        elif obs.event_type == 'git_commit':
            return obs.data.get('repo', 'unknown')
        elif obs.event_type == 'browser_visit':
            url = obs.data.get('url', '')
            # Extract domain
            if '//' in url:
                domain = url.split('//')[1].split('/')[0]
                return domain
            return 'browser'
        elif obs.event_type in ['file_modify', 'file_create']:
            path = obs.data.get('path', '')
            # Get top-level directory
            parts = path.split('/')
            if len(parts) > 1:
                return parts[0]
            return 'files'
        else:
            return obs.event_type

    def _identify_interrupters(
        self, observations: List[Any], switches: List[Switch]
    ) -> List[Tuple[str, int]]:
        """Identify top interrupters."""
        interrupter_counts = defaultdict(int)

        for switch in switches:
            # The "to" context is what interrupted
            interrupter_counts[switch.to_context] += 1

        # Sort by frequency
        sorted_interrupters = sorted(
            interrupter_counts.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_interrupters

    def _generate_recommendations(
        self,
        switches: List[Switch],
        flow_sessions: List[FlowSession],
        fragmented_sessions: List[FlowSession],
        top_interrupters: List[Tuple[str, int]],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Too many switches
        if len(switches) > 50:
            recommendations.append(
                f"You had {len(switches)} context switches. "
                "Consider batching similar tasks and using focus mode."
            )

        # Low flow time
        total_flow = sum(s.duration_minutes for s in flow_sessions)
        total_fragmented = sum(s.duration_minutes for s in fragmented_sessions)

        if total_flow < total_fragmented:
            recommendations.append(
                "You had more fragmented time than flow time. "
                "Try blocking larger chunks of time for deep work."
            )

        # Specific interrupters
        if top_interrupters:
            top_app = top_interrupters[0][0]
            top_count = top_interrupters[0][1]

            if top_count > 10:
                if 'slack' in top_app.lower():
                    recommendations.append(
                        f"Slack interrupted you {top_count} times. "
                        "Consider setting 'Do Not Disturb' during focus sessions."
                    )
                elif 'chrome' in top_app.lower() or 'browser' in top_app.lower():
                    recommendations.append(
                        f"Browser activity interrupted you {top_count} times. "
                        "Consider blocking distracting websites during deep work."
                    )
                else:
                    recommendations.append(
                        f"{top_app} interrupted you {top_count} times. "
                        "Consider disabling notifications from this app during focus time."
                    )

        # App switching recommendations
        app_switches = [s for s in switches if s.switch_type == 'app']
        if len(app_switches) > 30:
            recommendations.append(
                f"You switched apps {len(app_switches)} times. "
                "Try working in full-screen mode to reduce distractions."
            )

        return recommendations

    def _calculate_metrics(
        self,
        switches: List[Switch],
        flow_sessions: List[FlowSession],
        fragmented_sessions: List[FlowSession],
        observations: List[Any],
    ) -> Dict[str, Any]:
        """Calculate detailed metrics."""
        total_flow = sum(s.duration_minutes for s in flow_sessions)
        total_fragmented = sum(s.duration_minutes for s in fragmented_sessions)
        total_time = total_flow + total_fragmented

        metrics = {
            'flow_time_minutes': total_flow,
            'fragmented_time_minutes': total_fragmented,
            'flow_percentage': (total_flow / total_time * 100) if total_time > 0 else 0,
            'avg_flow_session_minutes': statistics.mean([s.duration_minutes for s in flow_sessions]) if flow_sessions else 0,
            'avg_fragmented_session_minutes': statistics.mean([s.duration_minutes for s in fragmented_sessions]) if fragmented_sessions else 0,
            'switches_per_hour': len(switches) / (total_time / 60) if total_time > 0 else 0,
        }

        # Recovery time by hour of day
        switches_by_hour = defaultdict(int)
        for switch in switches:
            switches_by_hour[switch.timestamp.hour] += 1

        if switches_by_hour:
            peak_hour = max(switches_by_hour.items(), key=lambda x: x[1])
            metrics['peak_switching_hour'] = f"{peak_hour[0]}:00 ({peak_hour[1]} switches)"

        return metrics

    def _count_by_type(self, switches: List[Switch]) -> Dict[str, int]:
        """Count switches by type."""
        counts = defaultdict(int)
        for switch in switches:
            counts[switch.switch_type] += 1
        return dict(counts)
