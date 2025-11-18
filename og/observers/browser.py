"""Browser activity observer for OG (Own Ghost).

This observer tracks browser activity including visited pages,
and optionally saves local copies of pages.
"""

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from og.base import Observation, PollingObserver


class BrowserObserver(PollingObserver):
    """Observer for browser activity.

    Tracks:
    - Pages visited (URL, title, timestamp)
    - Domain frequency
    - Optionally: local copies of visited pages

    Supports Chrome, Firefox, Safari (by reading their history databases)
    """

    def __init__(
        self,
        name: str = 'browser',
        poll_interval: float = 60.0,
        enabled: bool = True,
        save_pages: bool = False,
        save_dir: Optional[str] = None,
        browsers: Optional[list[str]] = None,
        **config
    ):
        """Initialize browser observer.

        Args:
            name: Observer name
            poll_interval: How often to check browser history (seconds)
            enabled: Whether observer is enabled
            save_pages: Whether to save local copies of visited pages
            save_dir: Directory to save pages (uses ~/.og/browser_cache if None)
            browsers: List of browsers to track ['chrome', 'firefox', 'safari']
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)
        self.save_pages = save_pages
        self.save_dir = Path(save_dir or Path.home() / '.og' / 'browser_cache')
        if save_pages:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.browsers = browsers or ['chrome', 'firefox', 'safari']
        self._last_check_time: Optional[datetime] = None
        self._seen_urls: set = set()

    def _get_browser_history_path(self, browser: str) -> Optional[Path]:
        """Get the path to a browser's history database.

        Args:
            browser: Browser name (chrome, firefox, safari)

        Returns:
            Path to history database or None if not found
        """
        import platform

        system = platform.system()
        home = Path.home()

        paths = {
            'chrome': {
                'Darwin': home
                / 'Library/Application Support/Google/Chrome/Default/History',
                'Linux': home / '.config/google-chrome/Default/History',
                'Windows': home
                / 'AppData/Local/Google/Chrome/User Data/Default/History',
            },
            'firefox': {
                'Darwin': home / 'Library/Application Support/Firefox/Profiles',
                'Linux': home / '.mozilla/firefox',
                'Windows': home / 'AppData/Roaming/Mozilla/Firefox/Profiles',
            },
            'safari': {
                'Darwin': home / 'Library/Safari/History.db',
            },
        }

        browser_paths = paths.get(browser, {})
        path = browser_paths.get(system)

        if path and path.exists():
            # For Firefox, we need to find the actual profile
            if browser == 'firefox' and path.is_dir():
                for profile_dir in path.iterdir():
                    if profile_dir.is_dir():
                        history_db = profile_dir / 'places.sqlite'
                        if history_db.exists():
                            return history_db
                return None
            return path

        return None

    def poll(self) -> list[Observation]:
        """Poll browser histories for new visits."""
        observations = []
        now = datetime.now()

        # Set time window
        if self._last_check_time is None:
            # First run, look back 1 hour
            self._last_check_time = now

        for browser in self.browsers:
            try:
                browser_obs = self._poll_browser(browser, self._last_check_time)
                observations.extend(browser_obs)
            except Exception as e:
                print(f"Error polling {browser}: {e}")

        self._last_check_time = now
        return observations

    def _poll_browser(
        self, browser: str, since: datetime
    ) -> list[Observation]:
        """Poll a specific browser's history.

        Args:
            browser: Browser name
            since: Only get visits since this time

        Returns:
            List of observations
        """
        history_path = self._get_browser_history_path(browser)
        if not history_path:
            return []

        observations = []

        try:
            # Copy database to avoid locking issues
            import tempfile
            import shutil

            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                tmp_path = tmp.name

            shutil.copy2(history_path, tmp_path)

            try:
                conn = sqlite3.connect(tmp_path)
                cursor = conn.cursor()

                visits = self._query_history(cursor, browser, since)

                for visit in visits:
                    url, title, visit_time = visit

                    # Skip if we've already seen this exact visit
                    visit_hash = hashlib.md5(
                        f"{url}:{visit_time}".encode()
                    ).hexdigest()
                    if visit_hash in self._seen_urls:
                        continue
                    self._seen_urls.add(visit_hash)

                    # Create observation
                    obs = self._create_visit_observation(
                        browser, url, title, visit_time
                    )
                    observations.append(obs)

                    # Optionally save page
                    if self.save_pages:
                        self._save_page(url, title, visit_time)

                conn.close()
            finally:
                # Clean up temp file
                os.unlink(tmp_path)

        except Exception as e:
            print(f"Error reading {browser} history: {e}")

        return observations

    def _query_history(
        self, cursor, browser: str, since: datetime
    ) -> list[tuple]:
        """Query browser history database.

        Args:
            cursor: Database cursor
            browser: Browser name
            since: Only get visits since this time

        Returns:
            List of (url, title, timestamp) tuples
        """
        # Convert datetime to timestamp
        since_ts = int(since.timestamp())

        queries = {
            'chrome': """
                SELECT url, title, last_visit_time
                FROM urls
                WHERE last_visit_time > ?
                ORDER BY last_visit_time DESC
                LIMIT 1000
            """,
            'firefox': """
                SELECT url, title, last_visit_date
                FROM moz_places
                WHERE last_visit_date > ?
                ORDER BY last_visit_date DESC
                LIMIT 1000
            """,
            'safari': """
                SELECT url, title, visit_time
                FROM history_visits
                JOIN history_items ON history_visits.history_item = history_items.id
                WHERE visit_time > ?
                ORDER BY visit_time DESC
                LIMIT 1000
            """,
        }

        query = queries.get(browser)
        if not query:
            return []

        # Chrome and Firefox use different timestamp formats
        if browser == 'chrome':
            # Chrome uses WebKit timestamp (microseconds since 1601-01-01)
            webkit_epoch = 11644473600
            since_webkit = (since_ts + webkit_epoch) * 1000000
            cursor.execute(query, (since_webkit,))
        elif browser == 'firefox':
            # Firefox uses microseconds since epoch
            since_firefox = since_ts * 1000000
            cursor.execute(query, (since_firefox,))
        else:
            # Safari uses seconds since 2001-01-01
            cursor.execute(query, (since_ts,))

        results = cursor.fetchall()

        # Normalize timestamps to datetime
        normalized = []
        for url, title, ts in results:
            if browser == 'chrome':
                # Convert WebKit timestamp to datetime
                webkit_epoch = 11644473600
                dt = datetime.fromtimestamp(ts / 1000000 - webkit_epoch)
            elif browser == 'firefox':
                # Convert microseconds to datetime
                dt = datetime.fromtimestamp(ts / 1000000)
            else:  # safari
                dt = datetime.fromtimestamp(ts)

            normalized.append((url, title or '', dt))

        return normalized

    def _create_visit_observation(
        self, browser: str, url: str, title: str, visit_time: datetime
    ) -> Observation:
        """Create an observation for a browser visit.

        Args:
            browser: Browser name
            url: Visited URL
            title: Page title
            visit_time: When the visit occurred

        Returns:
            Observation
        """
        # Parse URL for metadata
        parsed = urlparse(url)
        domain = parsed.netloc

        return Observation(
            timestamp=visit_time,
            observer_name=self.name,
            event_type='browser_visit',
            data={
                'browser': browser,
                'url': url,
                'title': title,
                'domain': domain,
                'path': parsed.path,
                'scheme': parsed.scheme,
            },
            metadata={
                'url_hash': hashlib.md5(url.encode()).hexdigest(),
            },
            tags=['browser', 'web', browser, domain],
        )

    def _save_page(self, url: str, title: str, visit_time: datetime):
        """Save a local copy of a visited page.

        Args:
            url: Page URL
            title: Page title
            visit_time: When visited
        """
        try:
            # Generate filename from URL hash
            url_hash = hashlib.md5(url.encode()).hexdigest()
            filename = f"{visit_time.strftime('%Y%m%d_%H%M%S')}_{url_hash}.json"
            filepath = self.save_dir / filename

            # Save metadata and URL (actual page content would require browser API)
            page_data = {
                'url': url,
                'title': title,
                'visit_time': visit_time.isoformat(),
                'saved_at': datetime.now().isoformat(),
            }

            with open(filepath, 'w') as f:
                json.dump(page_data, f, indent=2)

            # TODO: For actual page content, would need to use:
            # - Browser extension API
            # - Playwright/Selenium to render page
            # - mitmproxy to capture traffic

        except Exception as e:
            print(f"Error saving page {url}: {e}")
