"""IDE observer for OG (Own Ghost).

This observer tracks IDE/editor activity.
Supports VS Code, PyCharm, and others via filesystem watching.
"""

import os
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from og.base import Observation, PollingObserver


class IDEObserver(PollingObserver):
    """Observer for IDE/editor activity.

    Tracks:
    - Files being edited
    - Programming languages used
    - Time spent in editor
    - Projects being worked on

    Supports VS Code, PyCharm, and other editors via file monitoring.
    """

    def __init__(
        self,
        name: str = 'ide',
        poll_interval: float = 60.0,  # 1 minute
        enabled: bool = True,
        ide_type: str = 'auto',  # 'vscode', 'pycharm', or 'auto'
        **config
    ):
        """Initialize IDE observer.

        Args:
            name: Observer name
            poll_interval: How often to check IDE activity (seconds)
            enabled: Whether observer is enabled
            ide_type: Type of IDE to monitor
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)
        self.ide_type = ide_type
        self._last_files: dict[str, datetime] = {}

    def poll(self) -> list[Observation]:
        """Poll IDE for activity."""
        observations = []

        # Detect IDE type if auto
        if self.ide_type == 'auto':
            self.ide_type = self._detect_ide()

        try:
            if self.ide_type == 'vscode':
                observations = self._poll_vscode()
            elif self.ide_type == 'pycharm':
                observations = self._poll_pycharm()

        except Exception as e:
            print(f"Error polling IDE: {e}")

        return observations

    def _detect_ide(self) -> str:
        """Auto-detect which IDE is being used."""
        # Check VS Code
        vscode_dir = Path.home() / '.vscode'
        if vscode_dir.exists():
            return 'vscode'

        # Check PyCharm
        pycharm_dirs = list(Path.home().glob('.PyCharm*'))
        if pycharm_dirs:
            return 'pycharm'

        return 'unknown'

    def _poll_vscode(self) -> list[Observation]:
        """Poll VS Code for recent activity."""
        observations = []

        # VS Code stores recent files in storage.json
        # Location varies by OS
        import platform

        system = platform.system()

        if system == 'Darwin':  # macOS
            storage_path = (
                Path.home()
                / 'Library/Application Support/Code/User/globalStorage/storage.json'
            )
        elif system == 'Linux':
            storage_path = (
                Path.home() / '.config/Code/User/globalStorage/storage.json'
            )
        elif system == 'Windows':
            storage_path = (
                Path(os.getenv('APPDATA'))
                / 'Code/User/globalStorage/storage.json'
            )
        else:
            return []

        if not storage_path.exists():
            return []

        try:
            with open(storage_path, 'r') as f:
                data = json.load(f)

            # Get recently opened files/workspaces
            recently_opened = data.get('openedPathsList', {}).get('entries', [])

            for entry in recently_opened[:5]:  # Last 5
                if 'folderUri' in entry:
                    # It's a workspace/folder
                    workspace = entry['folderUri'].replace('file://', '')

                    obs = self.create_observation(
                        event_type='ide_workspace_opened',
                        data={
                            'workspace': workspace,
                            'ide': 'vscode',
                        },
                        tags=['ide', 'vscode', 'coding'],
                    )
                    observations.append(obs)

                elif 'fileUri' in entry:
                    # It's a file
                    file_path = entry['fileUri'].replace('file://', '')

                    # Get file extension to determine language
                    ext = Path(file_path).suffix
                    language = self._ext_to_language(ext)

                    obs = self.create_observation(
                        event_type='ide_file_opened',
                        data={
                            'file': file_path,
                            'language': language,
                            'ide': 'vscode',
                        },
                        tags=['ide', 'vscode', 'coding', language],
                    )
                    observations.append(obs)

        except Exception as e:
            print(f"Error reading VS Code data: {e}")

        return observations

    def _poll_pycharm(self) -> list[Observation]:
        """Poll PyCharm for recent activity."""
        observations = []

        # PyCharm stores recent projects in recentProjects.xml
        # Find PyCharm config directory
        config_dirs = list(Path.home().glob('.PyCharm*/config'))

        if not config_dirs:
            return []

        recent_projects_file = config_dirs[0] / 'options' / 'recentProjects.xml'

        if not recent_projects_file.exists():
            return []

        try:
            # Parse XML (simple approach)
            with open(recent_projects_file, 'r') as f:
                content = f.read()

            # Extract project paths (simple regex, could use xml.etree)
            import re

            projects = re.findall(r'value="([^"]+)"', content)

            for project_path in projects[:5]:  # Last 5 projects
                if Path(project_path).exists():
                    obs = self.create_observation(
                        event_type='ide_project_opened',
                        data={
                            'project': project_path,
                            'ide': 'pycharm',
                        },
                        tags=['ide', 'pycharm', 'coding', 'python'],
                    )
                    observations.append(obs)

        except Exception as e:
            print(f"Error reading PyCharm data: {e}")

        return observations

    def _ext_to_language(self, ext: str) -> str:
        """Map file extension to programming language."""
        mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.html': 'html',
            '.css': 'css',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.sh': 'shell',
        }

        return mapping.get(ext.lower(), 'unknown')
