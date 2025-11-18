"""Freelancer time tracking and invoicing.

Auto-generates timesheets, correlates work with clients/projects,
generates invoices with detailed activity backup.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal


@dataclass
class TimeEntry:
    """A time entry for billing."""

    start: datetime
    end: datetime
    duration_hours: float
    project: str
    client: str
    description: str
    billable: bool = True
    hourly_rate: Optional[Decimal] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class Invoice:
    """An invoice for a client."""

    invoice_number: str
    client: str
    issue_date: datetime
    due_date: datetime
    entries: List[TimeEntry]
    subtotal: Decimal
    tax_rate: Decimal = Decimal('0.0')
    total: Decimal = Decimal('0.0')

    def calculate_total(self):
        """Calculate invoice total."""
        self.subtotal = sum(
            Decimal(str(e.duration_hours)) * (e.hourly_rate or Decimal('0'))
            for e in self.entries
            if e.billable
        )
        self.total = self.subtotal * (1 + self.tax_rate)


class FreelancerTimeTracker:
    """Time tracking for freelancers."""

    def __init__(self, og_instance=None):
        self.og = og_instance
        self._project_mapping: Dict[str, tuple] = {}  # repo -> (project, client, rate)
        self._entries: List[TimeEntry] = []

    def map_project(self, repo: str, project: str, client: str, hourly_rate: Decimal):
        """Map a repository to a client project."""
        self._project_mapping[repo] = (project, client, hourly_rate)

    def generate_timesheet(
        self, start_date: datetime, end_date: datetime
    ) -> List[TimeEntry]:
        """Generate timesheet from observations."""
        if not self.og:
            return []

        observations = self.og.recent_activity(start_date=start_date, end_date=end_date)

        # Group by project and day
        sessions = self._detect_work_sessions(observations)

        entries = []
        for session in sessions:
            repo = session.get('repo')
            if repo in self._project_mapping:
                project, client, rate = self._project_mapping[repo]

                entry = TimeEntry(
                    start=session['start'],
                    end=session['end'],
                    duration_hours=session['duration_hours'],
                    project=project,
                    client=client,
                    description=self._generate_description(session['observations']),
                    hourly_rate=rate,
                )

                entries.append(entry)
                self._entries.append(entry)

        return entries

    def _detect_work_sessions(self, observations: List[Any]) -> List[Dict[str, Any]]:
        """Detect work sessions from observations."""
        sessions = []
        current_session = None

        for obs in sorted(observations, key=lambda x: x.timestamp):
            if obs.event_type in ['git_commit', 'file_modify']:
                repo = obs.data.get('repo', 'unknown')

                if current_session is None:
                    current_session = {
                        'start': obs.timestamp,
                        'end': obs.timestamp,
                        'repo': repo,
                        'observations': [obs],
                    }
                else:
                    time_gap = (obs.timestamp - current_session['end']).total_seconds() / 60

                    if time_gap < 15 and repo == current_session['repo']:
                        # Same session
                        current_session['end'] = obs.timestamp
                        current_session['observations'].append(obs)
                    else:
                        # End session
                        duration = (current_session['end'] - current_session['start']).total_seconds() / 3600
                        current_session['duration_hours'] = duration
                        sessions.append(current_session)

                        # Start new session
                        current_session = {
                            'start': obs.timestamp,
                            'end': obs.timestamp,
                            'repo': repo,
                            'observations': [obs],
                        }

        if current_session:
            duration = (current_session['end'] - current_session['start']).total_seconds() / 3600
            current_session['duration_hours'] = duration
            sessions.append(current_session)

        return sessions

    def _generate_description(self, observations: List[Any]) -> str:
        """Generate description from observations."""
        # Extract commit messages
        commits = [
            obs.data.get('message', '').split('\n')[0]
            for obs in observations
            if obs.event_type == 'git_commit'
        ]

        if commits:
            return commits[0]  # Use first commit message
        else:
            return "Development work"

    def generate_invoice(
        self,
        client: str,
        start_date: datetime,
        end_date: datetime,
        invoice_number: str,
        tax_rate: Decimal = Decimal('0.0'),
    ) -> Invoice:
        """Generate invoice for a client."""
        # Filter entries for this client
        client_entries = [
            e for e in self._entries
            if e.client == client
            and start_date <= e.start <= end_date
        ]

        invoice = Invoice(
            invoice_number=invoice_number,
            client=client,
            issue_date=datetime.now(),
            due_date=datetime.now() + timedelta(days=30),
            entries=client_entries,
            subtotal=Decimal('0'),
            tax_rate=tax_rate,
        )

        invoice.calculate_total()

        return invoice

    def export_invoice_markdown(self, invoice: Invoice) -> str:
        """Export invoice as markdown."""
        lines = [
            f"# Invoice {invoice.invoice_number}\n",
            f"**Client:** {invoice.client}",
            f"**Issue Date:** {invoice.issue_date.strftime('%Y-%m-%d')}",
            f"**Due Date:** {invoice.due_date.strftime('%Y-%m-%d')}\n",
            "## Time Entries\n",
        ]

        for entry in invoice.entries:
            lines.append(
                f"- {entry.start.strftime('%Y-%m-%d')} - "
                f"{entry.duration_hours:.2f}h - "
                f"{entry.description} - "
                f"${entry.hourly_rate * Decimal(str(entry.duration_hours)):.2f}"
            )

        lines.append(f"\n**Subtotal:** ${invoice.subtotal:.2f}")
        if invoice.tax_rate > 0:
            lines.append(f"**Tax ({invoice.tax_rate*100}%):** ${invoice.subtotal * invoice.tax_rate:.2f}")
        lines.append(f"**Total:** ${invoice.total:.2f}")

        return "\n".join(lines)
