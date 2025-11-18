"""Music/Spotify observer for OG (Own Ghost).

This observer tracks music listening activity.
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from og.base import Observation, PollingObserver


class MusicObserver(PollingObserver):
    """Observer for music listening activity.

    Tracks:
    - Songs played
    - Artists listened to
    - Listening duration
    - Music while working patterns

    Supports Spotify and Last.fm.
    """

    def __init__(
        self,
        name: str = 'music',
        poll_interval: float = 300.0,  # 5 minutes
        enabled: bool = True,
        provider: str = 'spotify',  # 'spotify' or 'lastfm'
        **config
    ):
        """Initialize music observer.

        Args:
            name: Observer name
            poll_interval: How often to check music (seconds)
            enabled: Whether observer is enabled
            provider: Music service ('spotify' or 'lastfm')
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)
        self.provider = provider
        self._client = None
        self._last_track_id: Optional[str] = None

    def poll(self) -> list[Observation]:
        """Poll for currently playing music."""
        observations = []

        try:
            if self.provider == 'spotify':
                observations = self._poll_spotify()
            elif self.provider == 'lastfm':
                observations = self._poll_lastfm()

        except Exception as e:
            print(f"Error polling music: {e}")

        return observations

    def _poll_spotify(self) -> list[Observation]:
        """Poll Spotify for currently playing track."""
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyOAuth

            if self._client is None:
                # Setup Spotify client
                scope = 'user-read-currently-playing user-read-playback-state'

                self._client = spotipy.Spotify(
                    auth_manager=SpotifyOAuth(
                        client_id=os.getenv('SPOTIFY_CLIENT_ID'),
                        client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
                        redirect_uri='http://localhost:8888/callback',
                        scope=scope,
                    )
                )

            # Get currently playing
            current = self._client.current_playback()

            if current and current['is_playing']:
                track = current['item']
                track_id = track['id']

                # Only record if it's a different track
                if track_id != self._last_track_id:
                    self._last_track_id = track_id

                    artist = track['artists'][0]['name']
                    song = track['name']
                    album = track['album']['name']
                    duration_ms = track['duration_ms']

                    obs = self.create_observation(
                        event_type='music_playing',
                        data={
                            'song': song,
                            'artist': artist,
                            'album': album,
                            'duration_seconds': duration_ms / 1000,
                        },
                        metadata={
                            'track_id': track_id,
                            'uri': track['uri'],
                        },
                        tags=['music', 'spotify', artist.lower()],
                    )
                    return [obs]

            return []

        except ImportError:
            print("spotipy required. Install with: pip install spotipy")
            return []

    def _poll_lastfm(self) -> list[Observation]:
        """Poll Last.fm for recent tracks."""
        import requests

        api_key = os.getenv('LASTFM_API_KEY')
        username = os.getenv('LASTFM_USERNAME')

        if not api_key or not username:
            print("Last.fm API key and username required")
            return []

        try:
            url = 'http://ws.audioscrobbler.com/2.0/'
            params = {
                'method': 'user.getrecenttracks',
                'user': username,
                'api_key': api_key,
                'format': 'json',
                'limit': 1,
            }

            response = requests.get(url, params=params)
            data = response.json()

            tracks = data.get('recenttracks', {}).get('track', [])

            if tracks:
                track = tracks[0] if isinstance(tracks, list) else tracks

                # Check if currently playing
                now_playing = track.get('@attr', {}).get('nowplaying') == 'true'

                if now_playing:
                    track_name = track['name']
                    artist = track['artist']['#text']

                    # Only record if different from last
                    track_id = f"{artist}_{track_name}"
                    if track_id != self._last_track_id:
                        self._last_track_id = track_id

                        obs = self.create_observation(
                            event_type='music_playing',
                            data={'song': track_name, 'artist': artist},
                            tags=['music', 'lastfm', artist.lower()],
                        )
                        return [obs]

            return []

        except Exception as e:
            print(f"Error polling Last.fm: {e}")
            return []
