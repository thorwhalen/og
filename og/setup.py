"""Interactive setup and configuration wizard for OG.

Provides a menu-driven interface for installing, configuring, and managing OG.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from typing import Optional, List, Dict
from og.config import ConfigManager, OGConfig


class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


class SetupWizard:
    """Interactive setup and configuration wizard."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config

    def run(self):
        """Run the interactive setup wizard."""
        self.clear_screen()
        self.print_header()

        while True:
            self.show_main_menu()
            choice = self.get_input("\nEnter your choice")

            if choice == '1':
                self.quick_setup()
            elif choice == '2':
                self.manage_observers()
            elif choice == '3':
                self.manage_features()
            elif choice == '4':
                self.manage_credentials()
            elif choice == '5':
                self.manage_privacy()
            elif choice == '6':
                self.install_dependencies()
            elif choice == '7':
                self.view_current_config()
            elif choice == '8':
                self.export_config()
            elif choice == '9':
                self.reset_config()
            elif choice == '0':
                print(f"\n{Colors.GREEN}Goodbye!{Colors.END}\n")
                break
            else:
                print(f"{Colors.RED}Invalid choice. Please try again.{Colors.END}")
                self.wait_for_enter()

    def print_header(self):
        """Print the wizard header."""
        print(f"""
{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║          🔍 Own Ghost (OG) Setup & Configuration         ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝{Colors.END}
        """)

    def show_main_menu(self):
        """Show the main menu."""
        print(f"""
{Colors.BOLD}Main Menu:{Colors.END}

  {Colors.CYAN}Setup & Installation:{Colors.END}
    1) Quick Setup (Recommended for first-time users)
    2) Manage Observers (Enable/disable activity tracking)
    3) Manage Features (Enable/disable advanced features)
    4) Manage Credentials (API keys, tokens)
    5) Manage Privacy Settings
    6) Install Dependencies

  {Colors.CYAN}Configuration:{Colors.END}
    7) View Current Configuration
    8) Export Configuration
    9) Reset to Defaults

  0) Exit
        """)

    def quick_setup(self):
        """Quick setup for first-time users."""
        self.clear_screen()
        print(f"{Colors.BOLD}{Colors.GREEN}Quick Setup{Colors.END}\n")

        print("This will guide you through the essential setup steps.\n")

        # Step 1: Storage location
        print(f"{Colors.BOLD}Step 1: Storage Location{Colors.END}")
        current = self.config.storage_dir
        print(f"Current: {current}")

        if self.confirm("Use default storage location?", default=True):
            self.config.storage_dir = str(Path.home() / '.og' / 'observations')
        else:
            new_dir = self.get_input("Enter storage directory path")
            self.config.storage_dir = new_dir

        # Step 2: OpenAI API Key
        print(f"\n{Colors.BOLD}Step 2: OpenAI API Key (for AI features){Colors.END}")
        if self.config.openai_api_key:
            print(f"Current: {self.mask_key(self.config.openai_api_key)}")

        if self.confirm("Set OpenAI API key?"):
            key = self.get_input("Enter your OpenAI API key", password=True)
            self.config.openai_api_key = key

        # Step 3: Core observers
        print(f"\n{Colors.BOLD}Step 3: Core Observers{Colors.END}")
        print("Core observers track your daily activity.")

        if self.confirm("Enable core observers (GitHub, Git, Browser, Apps, Files)?", default=True):
            self.config.enable_github_observer = True
            self.config.enable_git_observer = True
            self.config.enable_browser_observer = True
            self.config.enable_app_observer = True
            self.config.enable_filesystem_observer = True
            print(f"{Colors.GREEN}✓ Core observers enabled{Colors.END}")

        # Step 4: Privacy keyboard observer
        print(f"\n{Colors.BOLD}Step 4: Keyboard Observer{Colors.END}")
        print("The keyboard observer tracks typing patterns (NO text logging - privacy-first).")

        self.config.enable_keyboard_observer = self.confirm(
            "Enable keyboard observer?",
            default=False
        )

        # Step 5: Advanced features
        print(f"\n{Colors.BOLD}Step 5: Advanced Features{Colors.END}")

        if self.confirm("Enable semantic search (requires ChromaDB)?", default=True):
            self.config.enable_semantic_search = True

        if self.confirm("Enable automated standup generation?", default=True):
            self.config.enable_standup = True

        if self.confirm("Enable proactive insights and notifications?", default=True):
            self.config.enable_proactive_insights = True

        # Save configuration
        self.config_manager.save()

        print(f"\n{Colors.GREEN}✓ Quick setup complete!{Colors.END}")

        # Offer to install dependencies
        if self.confirm("\nInstall required Python packages?", default=True):
            self.install_dependencies()

        self.wait_for_enter()

    def manage_observers(self):
        """Manage observer settings."""
        while True:
            self.clear_screen()
            print(f"{Colors.BOLD}{Colors.BLUE}Manage Observers{Colors.END}\n")

            observers = {
                '1': ('GitHub', 'enable_github_observer', 'Tracks GitHub activity (commits, PRs, issues)'),
                '2': ('Keyboard', 'enable_keyboard_observer', 'Tracks typing patterns (no text logging)'),
                '3': ('Browser', 'enable_browser_observer', 'Tracks visited websites'),
                '4': ('App Usage', 'enable_app_observer', 'Tracks application usage'),
                '5': ('Filesystem', 'enable_filesystem_observer', 'Tracks file changes'),
                '6': ('Git', 'enable_git_observer', 'Tracks local git commits'),
                '7': ('Terminal', 'enable_terminal_observer', 'Tracks shell commands'),
                '8': ('Email', 'enable_email_observer', 'Tracks email activity (requires setup)'),
                '9': ('Calendar', 'enable_calendar_observer', 'Tracks calendar events (requires setup)'),
                '10': ('Slack', 'enable_slack_observer', 'Tracks Slack messages (requires setup)'),
                '11': ('Music', 'enable_music_observer', 'Tracks music listening (Spotify/Last.fm)'),
                '12': ('IDE', 'enable_ide_observer', 'Tracks IDE activity (VS Code, PyCharm)'),
            }

            print(f"{Colors.CYAN}Core Observers:{Colors.END}")
            for num in ['1', '2', '3', '4', '5', '6', '7']:
                name, flag, desc = observers[num]
                status = self.get_status(getattr(self.config, flag))
                print(f"  {num}) {status} {name} - {desc}")

            print(f"\n{Colors.CYAN}Extended Observers:{Colors.END}")
            for num in ['8', '9', '10', '11', '12']:
                name, flag, desc = observers[num]
                status = self.get_status(getattr(self.config, flag))
                print(f"  {num}) {status} {name} - {desc}")

            print(f"\n  {Colors.YELLOW}a) Enable all core observers{Colors.END}")
            print(f"  {Colors.YELLOW}d) Disable all observers{Colors.END}")
            print(f"  0) Back to main menu")

            choice = self.get_input("\nSelect observer to toggle")

            if choice == '0':
                break
            elif choice == 'a':
                for num in ['1', '2', '3', '4', '5', '6', '7']:
                    _, flag, _ = observers[num]
                    setattr(self.config, flag, True)
                self.config_manager.save()
                print(f"{Colors.GREEN}✓ All core observers enabled{Colors.END}")
                self.wait_for_enter()
            elif choice == 'd':
                if self.confirm("Disable ALL observers?", default=False):
                    for _, flag, _ in observers.values():
                        setattr(self.config, flag, False)
                    self.config_manager.save()
                    print(f"{Colors.YELLOW}All observers disabled{Colors.END}")
                    self.wait_for_enter()
            elif choice in observers:
                name, flag, _ = observers[choice]
                current = getattr(self.config, flag)
                setattr(self.config, flag, not current)
                self.config_manager.save()
                new_status = "enabled" if not current else "disabled"
                print(f"{Colors.GREEN}✓ {name} {new_status}{Colors.END}")
                self.wait_for_enter()

    def manage_features(self):
        """Manage advanced features."""
        while True:
            self.clear_screen()
            print(f"{Colors.BOLD}{Colors.BLUE}Manage Features{Colors.END}\n")

            features = {
                '1': ('Semantic Search', 'enable_semantic_search', 'Vector database search'),
                '2': ('Pattern Detection', 'enable_patterns', 'Detect behavioral patterns'),
                '3': ('Context Management', 'enable_contexts', 'Project/context tracking'),
                '4': ('Advanced Insights', 'enable_insights', 'Productivity analytics'),
                '5': ('Web Dashboard', 'enable_web_dashboard', 'Real-time web UI'),
                '6': ('Privacy Controls', 'enable_privacy', 'Encryption and redaction'),
                '7': ('Automated Standup', 'enable_standup', 'Daily standup generation'),
                '8': ('Context Switching Analysis', 'enable_switching_analysis', 'Measure interruption cost'),
                '9': ('Meeting Intelligence', 'enable_meeting_intelligence', 'Analyze meeting efficiency'),
                '10': ('Proactive Insights', 'enable_proactive_insights', 'Smart notifications'),
                '11': ('Learning Tracking', 'enable_learning_tracking', 'Track skill development'),
                '12': ('Voice Interface', 'enable_voice_interface', 'Voice commands'),
                '13': ('Focus Mode', 'enable_focus_mode', 'Block distractions'),
                '14': ('Mood Tracking', 'enable_mood_tracking', 'Mood/energy correlation'),
                '15': ('Task Integration', 'enable_task_integration', 'Sync with task managers'),
                '16': ('Time Tracking', 'enable_time_tracking', 'Freelancer invoicing'),
                '17': ('Predictive Scheduling', 'enable_predictive_scheduling', 'AI-powered scheduling'),
            }

            for num, (name, flag, desc) in features.items():
                status = self.get_status(getattr(self.config, flag))
                print(f"  {num}) {status} {name} - {desc}")

            print(f"\n  {Colors.YELLOW}a) Enable all features{Colors.END}")
            print(f"  {Colors.YELLOW}r) Enable recommended features{Colors.END}")
            print(f"  0) Back to main menu")

            choice = self.get_input("\nSelect feature to toggle")

            if choice == '0':
                break
            elif choice == 'a':
                for _, flag, _ in features.values():
                    setattr(self.config, flag, True)
                self.config_manager.save()
                print(f"{Colors.GREEN}✓ All features enabled{Colors.END}")
                self.wait_for_enter()
            elif choice == 'r':
                # Recommended features
                recommended = [
                    'enable_semantic_search',
                    'enable_patterns',
                    'enable_contexts',
                    'enable_insights',
                    'enable_standup',
                    'enable_proactive_insights',
                    'enable_learning_tracking',
                ]
                for flag in recommended:
                    setattr(self.config, flag, True)
                self.config_manager.save()
                print(f"{Colors.GREEN}✓ Recommended features enabled{Colors.END}")
                self.wait_for_enter()
            elif choice in features:
                name, flag, _ = features[choice]
                current = getattr(self.config, flag)
                setattr(self.config, flag, not current)
                self.config_manager.save()
                new_status = "enabled" if not current else "disabled"
                print(f"{Colors.GREEN}✓ {name} {new_status}{Colors.END}")
                self.wait_for_enter()

    def manage_credentials(self):
        """Manage API keys and credentials."""
        while True:
            self.clear_screen()
            print(f"{Colors.BOLD}{Colors.BLUE}Manage Credentials{Colors.END}\n")

            credentials = {
                '1': ('OpenAI API Key', 'openai_api_key', 'For AI-powered features'),
                '2': ('GitHub Token', 'github_token', 'For GitHub observer'),
                '3': ('Slack Token', 'slack_token', 'For Slack observer'),
                '4': ('Spotify Client ID', 'spotify_client_id', 'For music tracking'),
                '5': ('Spotify Client Secret', 'spotify_client_secret', 'For music tracking'),
                '6': ('Last.fm API Key', 'lastfm_api_key', 'For music tracking'),
                '7': ('Notion Token', 'notion_token', 'For Notion export'),
            }

            for num, (name, flag, desc) in credentials.items():
                current = getattr(self.config, flag)
                status = f"{Colors.GREEN}[SET]{Colors.END}" if current else f"{Colors.YELLOW}[NOT SET]{Colors.END}"
                masked = self.mask_key(current) if current else "Not set"
                print(f"  {num}) {status} {name} - {desc}")
                print(f"      Current: {masked}")

            print(f"\n  0) Back to main menu")

            choice = self.get_input("\nSelect credential to set")

            if choice == '0':
                break
            elif choice in credentials:
                name, flag, _ = credentials[choice]
                current = getattr(self.config, flag)

                if current:
                    print(f"Current: {self.mask_key(current)}")
                    if not self.confirm(f"Update {name}?"):
                        continue

                value = self.get_input(f"Enter {name}", password=True)
                setattr(self.config, flag, value)
                self.config_manager.save()
                print(f"{Colors.GREEN}✓ {name} updated{Colors.END}")
                self.wait_for_enter()

    def manage_privacy(self):
        """Manage privacy settings."""
        while True:
            self.clear_screen()
            print(f"{Colors.BOLD}{Colors.BLUE}Privacy Settings{Colors.END}\n")

            encryption_status = self.get_status(self.config.encryption_enabled)
            print(f"  1) {encryption_status} Encryption")
            print(f"      Current: {'Enabled' if self.config.encryption_enabled else 'Disabled'}")

            print(f"\n  2) Retention Period")
            print(f"      Current: {self.config.retention_days} days")

            print(f"\n  0) Back to main menu")

            choice = self.get_input("\nSelect option")

            if choice == '0':
                break
            elif choice == '1':
                self.config.encryption_enabled = not self.config.encryption_enabled
                if self.config.encryption_enabled and not self.config.encryption_password:
                    password = self.get_input("Enter encryption password", password=True)
                    self.config.encryption_password = password
                self.config_manager.save()
                status = "enabled" if self.config.encryption_enabled else "disabled"
                print(f"{Colors.GREEN}✓ Encryption {status}{Colors.END}")
                self.wait_for_enter()
            elif choice == '2':
                days = self.get_input("Enter retention period (days)", default=str(self.config.retention_days))
                try:
                    self.config.retention_days = int(days)
                    self.config_manager.save()
                    print(f"{Colors.GREEN}✓ Retention period updated{Colors.END}")
                except ValueError:
                    print(f"{Colors.RED}Invalid number{Colors.END}")
                self.wait_for_enter()

    def install_dependencies(self):
        """Install required dependencies."""
        self.clear_screen()
        print(f"{Colors.BOLD}{Colors.GREEN}Install Dependencies{Colors.END}\n")

        dependency_groups = {
            '1': ('Core', ['dol', 'i2', 'meshed', 'oa', 'openai']),
            '2': ('Observers', ['PyGithub', 'pynput', 'watchdog', 'psutil']),
            '3': ('Semantic Search', ['chromadb', 'sentence-transformers']),
            '4': ('Web Dashboard', ['flask', 'plotly']),
            '5': ('Privacy', ['cryptography']),
            '6': ('Integrations', [
                'google-api-python-client',
                'google-auth-httplib2',
                'google-auth-oauthlib',
                'slack-sdk',
                'spotipy',
                'pylast',
                'notion-client'
            ]),
        }

        print("Select dependency groups to install:\n")
        for num, (name, _) in dependency_groups.items():
            print(f"  {num}) {name}")

        print(f"\n  a) Install all")
        print(f"  0) Back to main menu")

        choice = self.get_input("\nEnter choice")

        if choice == '0':
            return
        elif choice == 'a':
            # Install all
            all_deps = []
            for _, deps in dependency_groups.values():
                all_deps.extend(deps)
            self._install_packages(all_deps)
        elif choice in dependency_groups:
            name, deps = dependency_groups[choice]
            self._install_packages(deps)

        self.wait_for_enter()

    def _install_packages(self, packages: List[str]):
        """Install Python packages."""
        print(f"\n{Colors.CYAN}Installing packages...{Colors.END}\n")

        for package in packages:
            print(f"Installing {package}...")
            try:
                subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    check=True,
                    capture_output=True
                )
                print(f"{Colors.GREEN}✓ {package} installed{Colors.END}")
            except subprocess.CalledProcessError as e:
                print(f"{Colors.RED}✗ Failed to install {package}{Colors.END}")
                print(f"  Error: {e.stderr.decode()}")

    def view_current_config(self):
        """View current configuration."""
        self.clear_screen()
        print(f"{Colors.BOLD}{Colors.CYAN}Current Configuration{Colors.END}\n")

        # Storage
        print(f"{Colors.BOLD}Storage:{Colors.END}")
        print(f"  Location: {self.config.storage_dir}")

        # Observers
        print(f"\n{Colors.BOLD}Enabled Observers:{Colors.END}")
        enabled_observers = self.config_manager.get_enabled_observers()
        if enabled_observers:
            for obs in enabled_observers:
                print(f"  ✓ {obs}")
        else:
            print(f"  {Colors.YELLOW}None{Colors.END}")

        # Features
        print(f"\n{Colors.BOLD}Enabled Features:{Colors.END}")
        enabled_features = self.config_manager.get_enabled_features()
        if enabled_features:
            for feat in enabled_features:
                print(f"  ✓ {feat}")
        else:
            print(f"  {Colors.YELLOW}None{Colors.END}")

        # Credentials
        print(f"\n{Colors.BOLD}Credentials:{Colors.END}")
        creds = {
            'OpenAI API Key': self.config.openai_api_key,
            'GitHub Token': self.config.github_token,
            'Slack Token': self.config.slack_token,
            'Notion Token': self.config.notion_token,
        }
        for name, value in creds.items():
            status = f"{Colors.GREEN}SET{Colors.END}" if value else f"{Colors.YELLOW}NOT SET{Colors.END}"
            print(f"  {name}: {status}")

        # Privacy
        print(f"\n{Colors.BOLD}Privacy:{Colors.END}")
        print(f"  Encryption: {'Enabled' if self.config.encryption_enabled else 'Disabled'}")
        print(f"  Retention: {self.config.retention_days} days")

        print(f"\n{Colors.CYAN}Config file: {self.config_manager.config_path}{Colors.END}")

        self.wait_for_enter()

    def export_config(self):
        """Export configuration."""
        self.clear_screen()
        print(f"{Colors.BOLD}{Colors.BLUE}Export Configuration{Colors.END}\n")

        print("1) Export as environment variables")
        print("2) Show config file location")
        print("0) Back")

        choice = self.get_input("\nEnter choice")

        if choice == '1':
            env = self.config_manager.export_env()
            print(f"\n{Colors.CYAN}Add these to your ~/.bashrc or ~/.zshrc:{Colors.END}\n")
            for key, value in env.items():
                print(f"export {key}='{value}'")
        elif choice == '2':
            print(f"\n{Colors.CYAN}Config file location:{Colors.END}")
            print(f"{self.config_manager.config_path}")

        self.wait_for_enter()

    def reset_config(self):
        """Reset configuration to defaults."""
        self.clear_screen()
        print(f"{Colors.BOLD}{Colors.RED}Reset Configuration{Colors.END}\n")

        print(f"{Colors.YELLOW}WARNING: This will reset all configuration to defaults.{Colors.END}")
        print("Your API keys and settings will be lost.\n")

        if self.confirm("Are you sure?", default=False):
            self.config_manager.reset()
            print(f"{Colors.GREEN}✓ Configuration reset to defaults{Colors.END}")
        else:
            print("Reset cancelled")

        self.wait_for_enter()

    # Helper methods

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if platform.system() == 'Windows' else 'clear')

    def get_input(self, prompt: str, default: str = "", password: bool = False) -> str:
        """Get user input."""
        if default:
            prompt = f"{prompt} [{default}]: "
        else:
            prompt = f"{prompt}: "

        if password:
            import getpass
            return getpass.getpass(prompt) or default
        else:
            return input(prompt) or default

    def confirm(self, prompt: str, default: bool = True) -> bool:
        """Get yes/no confirmation."""
        default_str = "Y/n" if default else "y/N"
        response = input(f"{prompt} [{default_str}]: ").lower().strip()

        if not response:
            return default

        return response in ['y', 'yes']

    def wait_for_enter(self):
        """Wait for user to press enter."""
        input(f"\n{Colors.CYAN}Press Enter to continue...{Colors.END}")

    def get_status(self, enabled: bool) -> str:
        """Get status indicator."""
        return f"{Colors.GREEN}[✓]{Colors.END}" if enabled else f"{Colors.YELLOW}[ ]{Colors.END}"

    def mask_key(self, key: str) -> str:
        """Mask an API key for display."""
        if not key or len(key) < 8:
            return "****"

        return f"{key[:4]}...{key[-4:]}"


def main():
    """Entry point for setup wizard."""
    wizard = SetupWizard()
    wizard.run()


if __name__ == '__main__':
    main()
