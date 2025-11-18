"""Terminal history observer for OG (Own Ghost).

This observer tracks terminal/shell command history.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from og.base import Observation, PollingObserver


class TerminalHistoryObserver(PollingObserver):
    """Observer for terminal command history.

    Tracks:
    - Commands executed in terminal
    - Command patterns
    - Working directories
    """

    def __init__(
        self,
        name: str = 'terminal',
        poll_interval: float = 60.0,
        enabled: bool = True,
        history_files: Optional[list[str]] = None,
        **config
    ):
        """Initialize terminal history observer.

        Args:
            name: Observer name
            poll_interval: How often to check history (seconds)
            enabled: Whether observer is enabled
            history_files: Paths to shell history files
            **config: Additional configuration
        """
        super().__init__(name, poll_interval, enabled, **config)

        # Auto-detect common shell history files
        if history_files is None:
            home = Path.home()
            self.history_files = [
                home / '.bash_history',
                home / '.zsh_history',
                home / '.fish_history',
                home / '.local/share/fish/fish_history',
            ]
            # Only keep files that exist
            self.history_files = [f for f in self.history_files if f.exists()]
        else:
            self.history_files = [Path(f) for f in history_files]

        self._last_positions: dict[Path, int] = {}

    def poll(self) -> list[Observation]:
        """Poll shell history files for new commands."""
        observations = []

        for history_file in self.history_files:
            try:
                hist_obs = self._poll_history_file(history_file)
                observations.extend(hist_obs)
            except Exception as e:
                print(f"Error polling {history_file}: {e}")

        return observations

    def _poll_history_file(self, history_file: Path) -> list[Observation]:
        """Poll a specific history file for new commands.

        Args:
            history_file: Path to history file

        Returns:
            List of observations for new commands
        """
        observations = []

        try:
            with open(history_file, 'r', errors='ignore') as f:
                # Get last position
                last_pos = self._last_positions.get(history_file, 0)

                # Seek to last position
                f.seek(last_pos)

                # Read new lines
                new_lines = f.readlines()

                # Update position
                self._last_positions[history_file] = f.tell()

                # Parse commands
                for line in new_lines:
                    line = line.strip()
                    if line:
                        command = self._parse_history_line(line, history_file)
                        if command:
                            obs = self.create_observation(
                                event_type='terminal_command',
                                data={
                                    'command': command,
                                    'shell': self._detect_shell(history_file),
                                },
                                tags=['terminal', 'command', 'shell'],
                            )
                            observations.append(obs)

        except Exception as e:
            print(f"Error reading {history_file}: {e}")

        return observations

    def _parse_history_line(self, line: str, history_file: Path) -> Optional[str]:
        """Parse a history line to extract the command.

        Args:
            line: Line from history file
            history_file: Which history file this came from

        Returns:
            Command string or None
        """
        # Different shells have different formats
        if 'zsh_history' in str(history_file):
            # Zsh format: : timestamp:0;command
            if line.startswith(':'):
                parts = line.split(';', 1)
                if len(parts) > 1:
                    return parts[1]

        elif 'fish_history' in str(history_file):
            # Fish format is YAML-like
            if line.startswith('- cmd:'):
                return line.replace('- cmd:', '').strip()

        # Bash and others: plain commands
        return line

    def _detect_shell(self, history_file: Path) -> str:
        """Detect which shell a history file belongs to.

        Args:
            history_file: Path to history file

        Returns:
            Shell name
        """
        name = history_file.name
        if 'bash' in name:
            return 'bash'
        elif 'zsh' in name:
            return 'zsh'
        elif 'fish' in name:
            return 'fish'
        return 'unknown'
