"""Calendar observer for OG (Own Ghost).

This observer tracks calendar events and meetings.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from og.base import Observation, PollingObserver


class CalendarObserver(PollingObserver):
    """Observer for calendar events and meetings.

    Tracks:
    - Upcoming meetings
    - Meeting duration
    - Meeting attendees
    - Calendar availability

    Supports Google Calendar and Outlook.
    """

    def __init__(
        self,
        name: str = 'calendar',
        poll_interval: float = 600.0,  # 10 minutes
        enabled: bool = True,
        provider: str = 'google',  # 'google' or 'outlook'
        lookahead_hours: int = 24,  # How far ahead to look
        **config
    ):
        """Initialize calendar observer.

        Args:
            name: Observer name
            poll_interval: How often to check calendar (seconds)
            enabled: Whether observer is enabled
            provider: Calendar provider ('google' or 'outlook')
            lookahead_hours: Hours ahead to fetch events
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)
        self.provider = provider
        self.lookahead_hours = lookahead_hours
        self._service = None
        self._seen_events: set[str] = set()

    def poll(self) -> list[Observation]:
        """Poll calendar for events."""
        observations = []

        try:
            if self.provider == 'google':
                observations = self._poll_google_calendar()
            elif self.provider == 'outlook':
                observations = self._poll_outlook()
            else:
                print(f"Unknown calendar provider: {self.provider}")

        except Exception as e:
            print(f"Error polling calendar: {e}")

        return observations

    def _poll_google_calendar(self) -> list[Observation]:
        """Poll Google Calendar."""
        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            import pickle
            from pathlib import Path

            observations = []

            SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

            creds = None
            token_file = Path.home() / '.og' / 'calendar_token.pickle'

            if token_file.exists():
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    credentials_file = Path.home() / '.og' / 'calendar_credentials.json'
                    if not credentials_file.exists():
                        print("Calendar credentials not found")
                        return []

                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_file), SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                token_file.parent.mkdir(parents=True, exist_ok=True)
                with open(token_file, 'wb') as token:
                    pickle.dump(creds, token)

            if self._service is None:
                self._service = build('calendar', 'v3', credentials=creds)

            # Get upcoming events
            now = datetime.utcnow().isoformat() + 'Z'
            end_time = (
                datetime.utcnow() + timedelta(hours=self.lookahead_hours)
            ).isoformat() + 'Z'

            events_result = (
                self._service.events()
                .list(
                    calendarId='primary',
                    timeMin=now,
                    timeMax=end_time,
                    maxResults=50,
                    singleEvents=True,
                    orderBy='startTime',
                )
                .execute()
            )

            events = events_result.get('items', [])

            for event in events:
                event_id = event['id']

                # Skip if already seen
                if event_id in self._seen_events:
                    continue
                self._seen_events.add(event_id)

                # Extract event details
                summary = event.get('summary', 'No title')
                start = event['start'].get('dateTime', event['start'].get('date'))
                end = event['end'].get('dateTime', event['end'].get('date'))

                # Parse times
                if 'T' in start:  # DateTime format
                    start_time = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end.replace('Z', '+00:00'))
                    duration = (end_time - start_time).total_seconds() / 60
                else:  # Date-only (all-day event)
                    start_time = datetime.fromisoformat(start)
                    duration = None

                attendees = event.get('attendees', [])
                attendee_count = len(attendees)

                # Create observation
                obs = self.create_observation(
                    event_type='calendar_event',
                    data={
                        'summary': summary,
                        'start': start,
                        'duration_minutes': duration,
                        'attendee_count': attendee_count,
                        'all_day': duration is None,
                    },
                    metadata={
                        'event_id': event_id,
                        'location': event.get('location', ''),
                    },
                    tags=['calendar', 'meeting', 'schedule'],
                )
                observations.append(obs)

            return observations

        except ImportError:
            print("Google Calendar API libraries required")
            return []

    def _poll_outlook(self) -> list[Observation]:
        """Poll Outlook Calendar using Microsoft Graph API."""
        # This would require Microsoft Graph API
        # Similar implementation to Google Calendar
        print("Outlook calendar support not yet implemented")
        return []
