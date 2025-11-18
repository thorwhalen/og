"""Export and integration features for OG.

This module provides export capabilities to various formats
and integrations with popular productivity tools.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from og.base import Observation
from og.storage import ObservationMall


class Exporter:
    """Export observations to various formats."""

    def __init__(self, mall: ObservationMall):
        """Initialize exporter.

        Args:
            mall: ObservationMall to export from
        """
        self.mall = mall

    def to_json(
        self,
        output_file: str,
        days: Optional[int] = None,
        observer: Optional[str] = None,
    ) -> int:
        """Export observations to JSON.

        Args:
            output_file: Path to output file
            days: Optional number of days to export
            observer: Optional observer filter

        Returns:
            Number of observations exported
        """
        observations = self._get_observations(days, observer)

        data = [obs.to_dict() for obs in observations]

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        return len(data)

    def to_csv(
        self,
        output_file: str,
        days: Optional[int] = None,
        observer: Optional[str] = None,
    ) -> int:
        """Export observations to CSV.

        Args:
            output_file: Path to output file
            days: Optional number of days to export
            observer: Optional observer filter

        Returns:
            Number of observations exported
        """
        observations = self._get_observations(days, observer)

        if not observations:
            return 0

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                ['timestamp', 'observer', 'event_type', 'data', 'tags']
            )

            # Data
            for obs in observations:
                writer.writerow(
                    [
                        obs.timestamp.isoformat(),
                        obs.observer_name,
                        obs.event_type,
                        json.dumps(obs.data),
                        ','.join(obs.tags),
                    ]
                )

        return len(observations)

    def to_markdown(
        self,
        output_file: str,
        days: Optional[int] = None,
        group_by: str = 'day',
    ) -> int:
        """Export observations to Markdown format.

        Useful for daily notes, documentation, etc.

        Args:
            output_file: Path to output file
            days: Optional number of days to export
            group_by: How to group observations ('day', 'type', 'observer')

        Returns:
            Number of observations exported
        """
        observations = self._get_observations(days)

        if not observations:
            return 0

        with open(output_file, 'w') as f:
            f.write(f"# Activity Log\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

            if group_by == 'day':
                # Group by day
                by_day = {}
                for obs in observations:
                    day = obs.timestamp.strftime('%Y-%m-%d')
                    if day not in by_day:
                        by_day[day] = []
                    by_day[day].append(obs)

                for day in sorted(by_day.keys(), reverse=True):
                    f.write(f"## {day}\n\n")
                    for obs in by_day[day]:
                        f.write(
                            f"- **{obs.timestamp.strftime('%H:%M')}** "
                            f"[{obs.event_type}] {self._format_obs_data(obs)}\n"
                        )
                    f.write("\n")

            elif group_by == 'type':
                # Group by event type
                by_type = {}
                for obs in observations:
                    if obs.event_type not in by_type:
                        by_type[obs.event_type] = []
                    by_type[obs.event_type].append(obs)

                for event_type in sorted(by_type.keys()):
                    f.write(f"## {event_type}\n\n")
                    for obs in by_type[event_type]:
                        f.write(
                            f"- {obs.timestamp.strftime('%Y-%m-%d %H:%M')} "
                            f"{self._format_obs_data(obs)}\n"
                        )
                    f.write("\n")

        return len(observations)

    def to_notion(
        self,
        days: Optional[int] = None,
        database_id: Optional[str] = None,
    ) -> int:
        """Export observations to Notion.

        Args:
            days: Optional number of days to export
            database_id: Notion database ID

        Returns:
            Number of observations exported

        Note:
            Requires Notion API key in NOTION_TOKEN environment variable
        """
        import os

        token = os.getenv('NOTION_TOKEN')
        if not token:
            raise ValueError("NOTION_TOKEN environment variable not set")

        try:
            from notion_client import Client

            notion = Client(auth=token)

            observations = self._get_observations(days)

            count = 0
            for obs in observations:
                # Create page in database
                properties = {
                    'Title': {'title': [{'text': {'content': obs.event_type}}]},
                    'Date': {'date': {'start': obs.timestamp.isoformat()}},
                    'Observer': {'select': {'name': obs.observer_name}},
                    'Data': {'rich_text': [{'text': {'content': json.dumps(obs.data)}}]},
                }

                if database_id:
                    notion.pages.create(parent={'database_id': database_id}, properties=properties)
                    count += 1

            return count

        except ImportError:
            raise ImportError("notion-client required. Install with: pip install notion-client")

    def to_obsidian(
        self,
        vault_path: str,
        days: Optional[int] = None,
        template: str = 'daily',
    ) -> int:
        """Export observations to Obsidian daily notes.

        Args:
            vault_path: Path to Obsidian vault
            days: Optional number of days to export
            template: Template to use ('daily', 'weekly')

        Returns:
            Number of notes created
        """
        vault = Path(vault_path)
        if not vault.exists():
            raise ValueError(f"Vault path does not exist: {vault_path}")

        observations = self._get_observations(days)

        # Group by day
        by_day = {}
        for obs in observations:
            day = obs.timestamp.strftime('%Y-%m-%d')
            if day not in by_day:
                by_day[day] = []
            by_day[day].append(obs)

        # Create daily notes
        daily_notes_dir = vault / 'Daily Notes'
        daily_notes_dir.mkdir(exist_ok=True)

        for day, day_obs in by_day.items():
            note_path = daily_notes_dir / f"{day}.md"

            with open(note_path, 'w') as f:
                f.write(f"# {day}\n\n")
                f.write(f"## Activity Log\n\n")

                for obs in sorted(day_obs, key=lambda x: x.timestamp):
                    f.write(
                        f"- {obs.timestamp.strftime('%H:%M')} "
                        f"**{obs.event_type}** - {self._format_obs_data(obs)}\n"
                    )

                f.write(f"\n---\n")
                f.write(f"Total observations: {len(day_obs)}\n")

        return len(by_day)

    def to_roam(
        self,
        output_file: str,
        days: Optional[int] = None,
    ) -> int:
        """Export observations to Roam Research format (JSON).

        Args:
            output_file: Path to output file
            days: Optional number of days to export

        Returns:
            Number of observations exported
        """
        observations = self._get_observations(days)

        # Group by day for Roam daily pages
        by_day = {}
        for obs in observations:
            day = obs.timestamp.strftime('%B %dth, %Y')  # Roam format
            if day not in by_day:
                by_day[day] = []
            by_day[day].append(obs)

        # Create Roam JSON structure
        roam_pages = []

        for day, day_obs in by_day.items():
            page = {
                'title': day,
                'children': [
                    {
                        'string': 'Activity Log',
                        'children': [
                            {
                                'string': f"{obs.timestamp.strftime('%H:%M')} "
                                f"{obs.event_type} - {self._format_obs_data(obs)}"
                            }
                            for obs in day_obs
                        ],
                    }
                ],
            }
            roam_pages.append(page)

        with open(output_file, 'w') as f:
            json.dump(roam_pages, f, indent=2)

        return len(observations)

    def _get_observations(
        self, days: Optional[int] = None, observer: Optional[str] = None
    ) -> list[Observation]:
        """Helper to get observations with filters.

        Args:
            days: Optional number of days
            observer: Optional observer filter

        Returns:
            Filtered observations
        """
        if days:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            if observer:
                observations = self.mall.query_by_timerange(
                    start_time, end_time, observer_name=observer
                )
            else:
                observations = self.mall.query_by_timerange(start_time, end_time)
        else:
            # Get all observations
            observations = []
            for store_name in self.mall.list_stores():
                store = self.mall.get_store(store_name)
                for key in store:
                    observations.append(store[key])

        return sorted(observations, key=lambda x: x.timestamp)

    def _format_obs_data(self, obs: Observation) -> str:
        """Format observation data for display.

        Args:
            obs: Observation to format

        Returns:
            Formatted string
        """
        # Format based on event type
        if obs.event_type == 'git_commit':
            return obs.data.get('message', '')

        elif obs.event_type == 'browser_visit':
            return obs.data.get('title', obs.data.get('url', ''))

        elif obs.event_type == 'terminal_command':
            return obs.data.get('command', '')

        # Default: show key fields
        key_fields = []
        for key, value in obs.data.items():
            if isinstance(value, (str, int, float)):
                key_fields.append(f"{key}: {value}")

        return ', '.join(key_fields[:3]) if key_fields else str(obs.data)


class Integrations:
    """Integrations with external services."""

    def __init__(self, mall: ObservationMall):
        """Initialize integrations.

        Args:
            mall: ObservationMall to integrate
        """
        self.mall = mall

    def sync_rescuetime(self, days: int = 7) -> dict:
        """Sync with RescueTime.

        Args:
            days: Number of days to sync

        Returns:
            Sync results
        """
        import os
        import requests

        api_key = os.getenv('RESCUETIME_API_KEY')
        if not api_key:
            raise ValueError("RESCUETIME_API_KEY not set")

        # Get data from RescueTime
        url = 'https://www.rescuetime.com/anapi/data'
        params = {
            'key': api_key,
            'perspective': 'interval',
            'restrict_kind': 'activity',
            'interval': 'hour',
            'format': 'json',
        }

        response = requests.get(url, params=params)
        data = response.json()

        # TODO: Convert RescueTime data to OG observations
        # This would create observations for app usage time

        return {'status': 'not_implemented', 'data': data}

    def sync_toggl(self, days: int = 7) -> dict:
        """Sync with Toggl time tracking.

        Args:
            days: Number of days to sync

        Returns:
            Sync results
        """
        import os
        import requests
        from datetime import timezone

        api_token = os.getenv('TOGGL_API_TOKEN')
        if not api_token:
            raise ValueError("TOGGL_API_TOKEN not set")

        # Get time entries
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        url = 'https://api.track.toggl.com/api/v9/me/time_entries'
        params = {
            'start_date': start_time.isoformat(),
            'end_date': end_time.isoformat(),
        }

        response = requests.get(url, params=params, auth=(api_token, 'api_token'))

        if response.status_code == 200:
            entries = response.json()
            # TODO: Convert to OG observations
            return {'status': 'not_implemented', 'entries': len(entries)}

        return {'status': 'error', 'message': response.text}
