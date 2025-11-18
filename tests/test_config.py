"""Tests for configuration management."""

import pytest
import tempfile
from pathlib import Path
from og.config import ConfigManager, OGConfig


def test_og_config_creation():
    """Test OGConfig creation."""
    config = OGConfig()

    assert config.ai_model == "gpt-4"
    assert config.enable_github_observer is True
    assert config.enable_semantic_search is True


def test_config_manager_creation():
    """Test ConfigManager creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        manager = ConfigManager(config_path=str(config_path))

        assert manager is not None
        assert manager.config is not None


def test_config_save_and_load():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        manager = ConfigManager(config_path=str(config_path))

        # Modify config
        manager.config.openai_api_key = 'test-key'
        manager.config.enable_web_dashboard = True
        manager.save()

        # Load in new manager
        manager2 = ConfigManager(config_path=str(config_path))

        assert manager2.config.openai_api_key == 'test-key'
        assert manager2.config.enable_web_dashboard is True


def test_config_get_set():
    """Test get and set methods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        manager = ConfigManager(config_path=str(config_path))

        # Get
        assert manager.get('ai_model') == 'gpt-4'

        # Set
        manager.set('ai_model', 'gpt-3.5-turbo')
        assert manager.get('ai_model') == 'gpt-3.5-turbo'


def test_config_update():
    """Test update method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        manager = ConfigManager(config_path=str(config_path))

        manager.update(
            openai_api_key='new-key',
            enable_focus_mode=True,
        )

        assert manager.config.openai_api_key == 'new-key'
        assert manager.config.enable_focus_mode is True


def test_config_reset():
    """Test reset method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        manager = ConfigManager(config_path=str(config_path))

        # Modify
        manager.config.openai_api_key = 'test-key'
        manager.save()

        # Reset
        manager.reset()

        assert manager.config.openai_api_key == ''


def test_config_export_env():
    """Test export_env method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        manager = ConfigManager(config_path=str(config_path))

        manager.config.openai_api_key = 'test-key'
        manager.config.github_token = 'github-token'
        manager.config.ai_model = 'gpt-4'

        env = manager.export_env()

        assert env['OPENAI_API_KEY'] == 'test-key'
        assert env['GITHUB_TOKEN'] == 'github-token'
        assert env['OG_MODEL'] == 'gpt-4'


def test_get_enabled_observers():
    """Test get_enabled_observers method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        manager = ConfigManager(config_path=str(config_path))

        # Default has some enabled
        enabled = manager.get_enabled_observers()
        assert 'github' in enabled
        assert 'git' in enabled

        # Disable all, enable email
        manager.config.enable_github_observer = False
        manager.config.enable_git_observer = False
        manager.config.enable_email_observer = True

        enabled = manager.get_enabled_observers()
        assert 'github' not in enabled
        assert 'email' in enabled


def test_get_enabled_features():
    """Test get_enabled_features method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        manager = ConfigManager(config_path=str(config_path))

        # Default has some enabled
        features = manager.get_enabled_features()
        assert 'Semantic Search' in features
        assert 'Automated Standup' in features

        # Enable a feature
        manager.config.enable_voice_interface = True
        features = manager.get_enabled_features()
        assert 'Voice Interface' in features
