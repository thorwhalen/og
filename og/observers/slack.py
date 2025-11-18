"""Slack observer for OG (Own Ghost).

This observer tracks Slack activity.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from og.base import Observation, PollingObserver


class SlackObserver(PollingObserver):
    """Observer for Slack activity.

    Tracks:
    - Messages sent
    - Channels active in
    - Direct messages
    - Reactions given/received

    Requires Slack API token.
    """

    def __init__(
        self,
        name: str = 'slack',
        poll_interval: float = 300.0,  # 5 minutes
        enabled: bool = True,
        token: Optional[str] = None,
        include_message_content: bool = False,  # Privacy
        **config
    ):
        """Initialize Slack observer.

        Args:
            name: Observer name
            poll_interval: How often to poll Slack (seconds)
            enabled: Whether observer is enabled
            token: Slack API token
            include_message_content: Whether to include actual messages (privacy)
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)
        self.token = token or os.getenv('SLACK_TOKEN')
        self.include_message_content = include_message_content
        self._slack_client = None
        self._last_check: Optional[datetime] = None
        self._user_id: Optional[str] = None

    def poll(self) -> list[Observation]:
        """Poll Slack for activity."""
        if not self.token:
            print("Slack token not configured")
            return []

        observations = []

        try:
            from slack_sdk import WebClient

            if self._slack_client is None:
                self._slack_client = WebClient(token=self.token)

                # Get user ID
                auth_response = self._slack_client.auth_test()
                self._user_id = auth_response['user_id']

            # Determine time range
            now = datetime.now()
            since = self._last_check or (now - timedelta(hours=1))
            since_ts = since.timestamp()

            # Get conversations (channels)
            conversations = self._slack_client.conversations_list()

            for channel in conversations['channels']:
                channel_id = channel['id']
                channel_name = channel['name']

                # Get recent messages in this channel
                try:
                    history = self._slack_client.conversations_history(
                        channel=channel_id, oldest=str(since_ts), limit=100
                    )

                    # Count our messages
                    our_messages = [
                        msg
                        for msg in history.get('messages', [])
                        if msg.get('user') == self._user_id
                    ]

                    if our_messages:
                        obs = self.create_observation(
                            event_type='slack_messages',
                            data={
                                'channel': channel_name,
                                'message_count': len(our_messages),
                                'include_content': self.include_message_content,
                            },
                            tags=['slack', 'communication', 'team'],
                        )
                        observations.append(obs)

                except Exception as e:
                    # Channel might be private or archived
                    continue

            self._last_check = now

        except ImportError:
            print("slack-sdk required. Install with: pip install slack-sdk")
            return []
        except Exception as e:
            print(f"Error polling Slack: {e}")

        return observations
