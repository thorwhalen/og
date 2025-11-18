"""Configuration management for OG.

Handles configuration storage, loading, and management.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import platform


@dataclass
class OGConfig:
    """OG configuration."""

    # Storage settings
    storage_dir: str = ""

    # API keys and credentials
    openai_api_key: str = ""
    github_token: str = ""
    slack_token: str = ""
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    lastfm_api_key: str = ""
    notion_token: str = ""

    # Model settings
    ai_model: str = "gpt-4"

    # Core observers (enabled by default)
    enable_github_observer: bool = True
    enable_keyboard_observer: bool = True
    enable_browser_observer: bool = True
    enable_app_observer: bool = True
    enable_filesystem_observer: bool = True
    enable_git_observer: bool = True
    enable_terminal_observer: bool = True

    # Extended observers (disabled by default)
    enable_email_observer: bool = False
    enable_calendar_observer: bool = False
    enable_slack_observer: bool = False
    enable_music_observer: bool = False
    enable_ide_observer: bool = False

    # Advanced features
    enable_semantic_search: bool = True
    enable_patterns: bool = True
    enable_contexts: bool = True
    enable_insights: bool = True
    enable_web_dashboard: bool = False
    enable_privacy: bool = True

    # Productivity features
    enable_standup: bool = True
    enable_switching_analysis: bool = True
    enable_meeting_intelligence: bool = False
    enable_proactive_insights: bool = True
    enable_learning_tracking: bool = True
    enable_voice_interface: bool = False
    enable_focus_mode: bool = False
    enable_mood_tracking: bool = False
    enable_task_integration: bool = False
    enable_time_tracking: bool = False
    enable_predictive_scheduling: bool = False

    # Privacy settings
    encryption_enabled: bool = False
    encryption_password: str = ""
    retention_days: int = 365

    # Web dashboard settings
    dashboard_port: int = 5050
    dashboard_host: str = "localhost"

    # Notification settings
    enable_notifications: bool = True
    notification_priority: str = "normal"  # low, normal, high


class ConfigManager:
    """Manages OG configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager.

        Args:
            config_path: Path to config file (default: ~/.og/config.json)
        """
        if config_path is None:
            config_dir = Path.home() / '.og'
            config_dir.mkdir(exist_ok=True)
            config_path = config_dir / 'config.json'

        self.config_path = Path(config_path)
        self.config = self.load()

    def load(self) -> OGConfig:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)

                    # Set storage_dir default
                    if not data.get('storage_dir'):
                        data['storage_dir'] = str(Path.home() / '.og' / 'observations')

                    return OGConfig(**data)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
                return self._default_config()
        else:
            return self._default_config()

    def _default_config(self) -> OGConfig:
        """Get default configuration."""
        config = OGConfig()
        config.storage_dir = str(Path.home() / '.og' / 'observations')
        return config

    def save(self):
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

    def get(self, key: str, default=None) -> Any:
        """Get a config value."""
        return getattr(self.config, key, default)

    def set(self, key: str, value: Any):
        """Set a config value."""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
            self.save()
        else:
            raise KeyError(f"Unknown config key: {key}")

    def update(self, **kwargs):
        """Update multiple config values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save()

    def reset(self):
        """Reset to default configuration."""
        self.config = self._default_config()
        self.save()

    def export_env(self) -> Dict[str, str]:
        """Export config as environment variables."""
        env = {}

        if self.config.openai_api_key:
            env['OPENAI_API_KEY'] = self.config.openai_api_key
        if self.config.github_token:
            env['GITHUB_TOKEN'] = self.config.github_token
        if self.config.slack_token:
            env['SLACK_TOKEN'] = self.config.slack_token
        if self.config.spotify_client_id:
            env['SPOTIFY_CLIENT_ID'] = self.config.spotify_client_id
        if self.config.spotify_client_secret:
            env['SPOTIFY_CLIENT_SECRET'] = self.config.spotify_client_secret
        if self.config.lastfm_api_key:
            env['LASTFM_API_KEY'] = self.config.lastfm_api_key
        if self.config.notion_token:
            env['NOTION_TOKEN'] = self.config.notion_token

        if self.config.ai_model:
            env['OG_MODEL'] = self.config.ai_model

        return env

    def get_enabled_observers(self) -> list[str]:
        """Get list of enabled observers."""
        enabled = []

        observer_flags = {
            'enable_github_observer': 'github',
            'enable_keyboard_observer': 'keyboard',
            'enable_browser_observer': 'browser',
            'enable_app_observer': 'app_usage',
            'enable_filesystem_observer': 'filesystem',
            'enable_git_observer': 'git',
            'enable_terminal_observer': 'terminal',
            'enable_email_observer': 'email',
            'enable_calendar_observer': 'calendar',
            'enable_slack_observer': 'slack',
            'enable_music_observer': 'music',
            'enable_ide_observer': 'ide',
        }

        for flag, name in observer_flags.items():
            if getattr(self.config, flag, False):
                enabled.append(name)

        return enabled

    def get_enabled_features(self) -> list[str]:
        """Get list of enabled features."""
        enabled = []

        feature_flags = {
            'enable_semantic_search': 'Semantic Search',
            'enable_patterns': 'Pattern Detection',
            'enable_contexts': 'Context Management',
            'enable_insights': 'Advanced Insights',
            'enable_web_dashboard': 'Web Dashboard',
            'enable_privacy': 'Privacy Controls',
            'enable_standup': 'Automated Standup',
            'enable_switching_analysis': 'Context Switching Analysis',
            'enable_meeting_intelligence': 'Meeting Intelligence',
            'enable_proactive_insights': 'Proactive Insights',
            'enable_learning_tracking': 'Learning Tracking',
            'enable_voice_interface': 'Voice Interface',
            'enable_focus_mode': 'Focus Mode',
            'enable_mood_tracking': 'Mood Tracking',
            'enable_task_integration': 'Task Integration',
            'enable_time_tracking': 'Time Tracking',
            'enable_predictive_scheduling': 'Predictive Scheduling',
        }

        for flag, name in feature_flags.items():
            if getattr(self.config, flag, False):
                enabled.append(name)

        return enabled
