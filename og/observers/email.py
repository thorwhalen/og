"""Email observer for OG (Own Ghost).

This observer tracks email activity via IMAP or Gmail API.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from og.base import Observation, PollingObserver


class EmailObserver(PollingObserver):
    """Observer for email activity.

    Tracks:
    - Emails received
    - Emails sent
    - Email subjects and senders (configurable privacy)
    - Email labels/folders

    Supports IMAP and Gmail API.
    """

    def __init__(
        self,
        name: str = 'email',
        poll_interval: float = 300.0,  # 5 minutes
        enabled: bool = True,
        provider: str = 'gmail',  # 'gmail' or 'imap'
        email_address: Optional[str] = None,
        include_content: bool = False,  # Privacy: don't include email content by default
        **config
    ):
        """Initialize email observer.

        Args:
            name: Observer name
            poll_interval: How often to check email (seconds)
            enabled: Whether observer is enabled
            provider: Email provider ('gmail' or 'imap')
            email_address: Your email address
            include_content: Whether to include email content (privacy consideration)
            **config: Additional configuration (credentials, etc.)
        """
        super().__init__(name, poll_interval, enabled, **config)
        self.provider = provider
        self.email_address = email_address or os.getenv('OG_EMAIL')
        self.include_content = include_content

        self._last_check_time: Optional[datetime] = None
        self._gmail_service = None
        self._imap_connection = None

    def poll(self) -> list[Observation]:
        """Poll email for new messages."""
        if not self.email_address:
            print("Email address not configured")
            return []

        observations = []

        try:
            if self.provider == 'gmail':
                observations = self._poll_gmail()
            elif self.provider == 'imap':
                observations = self._poll_imap()
            else:
                print(f"Unknown email provider: {self.provider}")

        except Exception as e:
            print(f"Error polling email: {e}")

        # Update last check time
        self._last_check_time = datetime.now()

        return observations

    def _poll_gmail(self) -> list[Observation]:
        """Poll Gmail using Gmail API."""
        try:
            from googleapiclient.discovery import build
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            import pickle
            from pathlib import Path

            observations = []

            # Gmail API scope
            SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

            creds = None
            token_file = Path.home() / '.og' / 'gmail_token.pickle'

            # Load credentials
            if token_file.exists():
                with open(token_file, 'rb') as token:
                    creds = pickle.load(token)

            # Refresh if needed
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    # Need user authorization
                    credentials_file = Path.home() / '.og' / 'gmail_credentials.json'
                    if not credentials_file.exists():
                        print(
                            "Gmail credentials not found. "
                            "Please download credentials.json from Google Cloud Console."
                        )
                        return []

                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_file), SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save credentials
                token_file.parent.mkdir(parents=True, exist_ok=True)
                with open(token_file, 'wb') as token:
                    pickle.dump(creds, token)

            # Build service
            if self._gmail_service is None:
                self._gmail_service = build('gmail', 'v1', credentials=creds)

            # Query for recent messages
            query = 'is:inbox'
            if self._last_check_time:
                # Gmail doesn't support datetime filters directly
                # Use "after" with date
                date_str = self._last_check_time.strftime('%Y/%m/%d')
                query += f' after:{date_str}'

            # Get messages
            results = (
                self._gmail_service.users()
                .messages()
                .list(userId='me', q=query, maxResults=50)
                .execute()
            )

            messages = results.get('messages', [])

            for msg_info in messages:
                msg_id = msg_info['id']

                # Get full message
                message = (
                    self._gmail_service.users()
                    .messages()
                    .get(userId='me', id=msg_id, format='metadata')
                    .execute()
                )

                # Extract metadata
                headers = {h['name']: h['value'] for h in message.get('payload', {}).get('headers', [])}

                subject = headers.get('Subject', 'No subject')
                sender = headers.get('From', 'Unknown')
                date_str = headers.get('Date', '')

                # Parse date
                from email.utils import parsedate_to_datetime

                try:
                    email_date = parsedate_to_datetime(date_str)
                except Exception:
                    email_date = datetime.now()

                # Only include if after last check
                if self._last_check_time and email_date <= self._last_check_time:
                    continue

                # Create observation
                obs = self.create_observation(
                    event_type='email_received',
                    data={
                        'subject': subject,
                        'from': sender,
                        'labels': message.get('labelIds', []),
                    },
                    metadata={
                        'message_id': msg_id,
                        'received_at': email_date.isoformat(),
                    },
                    tags=['email', 'communication'],
                )
                observations.append(obs)

            return observations

        except ImportError:
            print(
                "Gmail API libraries required. "
                "Install with: pip install google-auth google-auth-oauthlib google-api-python-client"
            )
            return []

    def _poll_imap(self) -> list[Observation]:
        """Poll email using IMAP."""
        import imaplib
        import email
        from email.header import decode_header

        observations = []

        try:
            # IMAP settings from config
            imap_server = self.config.get('imap_server', 'imap.gmail.com')
            imap_port = self.config.get('imap_port', 993)
            password = self.config.get('password') or os.getenv('OG_EMAIL_PASSWORD')

            if not password:
                print("Email password not configured")
                return []

            # Connect to IMAP server
            if self._imap_connection is None:
                self._imap_connection = imaplib.IMAP4_SSL(imap_server, imap_port)
                self._imap_connection.login(self.email_address, password)

            # Select mailbox
            self._imap_connection.select('INBOX')

            # Search for recent emails
            if self._last_check_time:
                # Search since last check
                date_str = self._last_check_time.strftime('%d-%b-%Y')
                _, message_numbers = self._imap_connection.search(
                    None, f'(SINCE {date_str})'
                )
            else:
                # Get last 20 emails
                _, message_numbers = self._imap_connection.search(None, 'ALL')

            # Get message IDs
            msg_ids = message_numbers[0].split()[-20:]  # Last 20

            for msg_id in msg_ids:
                # Fetch message
                _, msg_data = self._imap_connection.fetch(msg_id, '(RFC822)')

                # Parse email
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)

                # Extract headers
                subject = email_message.get('Subject', '')
                sender = email_message.get('From', '')
                date_str = email_message.get('Date', '')

                # Decode subject if needed
                if subject:
                    decoded = decode_header(subject)[0]
                    if isinstance(decoded[0], bytes):
                        subject = decoded[0].decode(decoded[1] or 'utf-8')
                    else:
                        subject = decoded[0]

                # Create observation
                obs = self.create_observation(
                    event_type='email_received',
                    data={'subject': subject, 'from': sender},
                    metadata={'date': date_str},
                    tags=['email', 'communication'],
                )
                observations.append(obs)

            return observations

        except Exception as e:
            print(f"Error with IMAP: {e}")
            self._imap_connection = None
            return []

    def stop(self):
        """Stop observer and cleanup connections."""
        super().stop()

        # Close IMAP connection
        if self._imap_connection:
            try:
                self._imap_connection.close()
                self._imap_connection.logout()
            except Exception:
                pass
            self._imap_connection = None
