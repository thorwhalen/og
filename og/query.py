"""Query interface for OG (Own Ghost).

This module provides a high-level interface for querying observations
and interacting with the OG system.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from og.ai import OGAgent
from og.registry import ObserverRegistry
from og.storage import ObservationMall


class OG:
    """Main interface to the Own Ghost system.

    This provides a simple API for:
    - Starting/stopping observation
    - Querying activity
    - Getting AI-powered insights
    - Managing observers
    - Semantic search
    - Pattern detection and alerts
    - Context/project tracking
    - Advanced insights
    - Export and privacy controls
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        registry: Optional[ObserverRegistry] = None,
        enable_semantic_search: bool = True,
        enable_patterns: bool = True,
        enable_contexts: bool = True,
    ):
        """Initialize OG system.

        Args:
            storage_dir: Directory for storing observations
            registry: Observer registry (creates default if None)
            enable_semantic_search: Enable vector database for semantic search
            enable_patterns: Enable pattern detection and alerts
            enable_contexts: Enable context/project tracking
        """
        from og.registry import create_default_registry

        self.mall = ObservationMall(storage_dir)
        self.registry = registry or create_default_registry()
        self.agent = OGAgent(self.mall)

        # Advanced features
        self.semantic_memory = None
        self.pattern_detector = None
        self.context_manager = None
        self.insight_engine = None
        self.exporter = None
        self.privacy_controls = None

        # Initialize advanced features
        if enable_semantic_search:
            self._init_semantic_search(storage_dir)

        if enable_patterns:
            self._init_pattern_detection()

        if enable_contexts:
            self._init_context_manager()

        # Always initialize these
        self._init_insights()
        self._init_export()
        self._init_privacy()

        self._running = False

    def _init_semantic_search(self, storage_dir: Optional[str]):
        """Initialize semantic search."""
        try:
            from og.semantic import SemanticMemory

            persist_dir = None
            if storage_dir:
                persist_dir = str(Path(storage_dir).parent / 'semantic_db')

            self.semantic_memory = SemanticMemory(persist_directory=persist_dir)
        except Exception as e:
            print(f"Could not initialize semantic search: {e}")

    def _init_pattern_detection(self):
        """Initialize pattern detection."""
        try:
            from og.patterns import PatternDetector, AlertHandler

            self.pattern_detector = PatternDetector()
            self.alert_handler = AlertHandler()

            # Add default handlers
            self.alert_handler.add_handler(AlertHandler.console_handler)
        except Exception as e:
            print(f"Could not initialize pattern detection: {e}")

    def _init_context_manager(self):
        """Initialize context manager."""
        try:
            from og.context import ContextManager

            self.context_manager = ContextManager()
        except Exception as e:
            print(f"Could not initialize context manager: {e}")

    def _init_insights(self):
        """Initialize insights engine."""
        try:
            from og.insights import InsightEngine

            self.insight_engine = InsightEngine(self.mall, self.agent)
        except Exception as e:
            print(f"Could not initialize insights: {e}")

    def _init_export(self):
        """Initialize exporter."""
        try:
            from og.export import Exporter

            self.exporter = Exporter(self.mall)
        except Exception as e:
            print(f"Could not initialize exporter: {e}")

    def _init_privacy(self):
        """Initialize privacy controls."""
        try:
            from og.privacy import PrivacyControls

            self.privacy_controls = PrivacyControls()
        except Exception as e:
            print(f"Could not initialize privacy controls: {e}")

    def start(self):
        """Start all enabled observers."""
        if self._running:
            print("OG is already running")
            return

        print("Starting Own Ghost observers...")
        self.registry.start_all()
        self._running = True
        print(f"Started {len(self.registry.get_enabled())} observers")

    def stop(self):
        """Stop all observers."""
        if not self._running:
            print("OG is not running")
            return

        print("Stopping Own Ghost observers...")
        self.registry.stop_all()
        self._running = False
        print("Stopped")

    def status(self) -> dict:
        """Get system status.

        Returns:
            Dictionary with system status information
        """
        enabled_observers = self.registry.get_enabled()
        all_observers = list(self.registry.keys())

        return {
            'running': self._running,
            'total_observers': len(all_observers),
            'enabled_observers': len(enabled_observers),
            'observers': {
                'enabled': [obs.name for obs in enabled_observers],
                'all': all_observers,
            },
            'stores': self.mall.list_stores(),
        }

    # ---- Querying Activity ----

    def summary(
        self,
        days: int = 1,
        detail: str = 'medium',
    ) -> str:
        """Get a summary of recent activity.

        Args:
            days: Number of days to summarize
            detail: Detail level ('brief', 'medium', 'detailed')

        Returns:
            AI-generated summary

        Example:
            >>> og = OG()
            >>> print(og.summary(days=1))
            "Today you focused primarily on coding..."
        """
        return self.agent.summarize_period(days=days, detail_level=detail)

    def ask(self, question: str, days: int = 7) -> str:
        """Ask a question about your activity.

        Args:
            question: Your question
            days: How many days of history to consider

        Returns:
            AI-generated answer

        Example:
            >>> og = OG()
            >>> print(og.ask("What did I work on yesterday?"))
        """
        return self.agent.answer_question(question, days=days)

    def report(
        self,
        days: int = 7,
        report_type: str = 'productivity',
    ) -> str:
        """Generate a detailed report.

        Args:
            days: Number of days to analyze
            report_type: Type of report ('productivity', 'technical', 'comprehensive')

        Returns:
            AI-generated report

        Example:
            >>> og = OG()
            >>> print(og.report(days=7, report_type='productivity'))
        """
        return self.agent.generate_report(days=days, report_type=report_type)

    def recent_activity(
        self,
        hours: Optional[int] = None,
        days: Optional[int] = None,
        event_type: Optional[str] = None,
    ) -> list:
        """Get recent observations.

        Args:
            hours: Get activity from last N hours
            days: Get activity from last N days
            event_type: Filter by event type

        Returns:
            List of observations

        Example:
            >>> og = OG()
            >>> commits = og.recent_activity(days=1, event_type='git_commit')
        """
        # Determine time range
        end_time = datetime.now()

        if hours:
            start_time = end_time - timedelta(hours=hours)
        elif days:
            start_time = end_time - timedelta(days=days)
        else:
            start_time = end_time - timedelta(days=1)  # Default to 1 day

        # Query observations
        if event_type:
            observations = self.mall.query_by_event_type(event_type)
            # Filter by time
            observations = [
                obs for obs in observations if start_time <= obs.timestamp <= end_time
            ]
        else:
            observations = self.mall.query_by_timerange(start_time, end_time)

        return observations

    # ---- Observer Management ----

    def list_observers(self) -> dict:
        """List all available observers.

        Returns:
            Dictionary with observer information
        """
        observers = {}
        for name in self.registry:
            observer = self.registry[name]
            observers[name] = {
                'enabled': getattr(observer, 'enabled', True),
                'running': getattr(observer, 'is_running', lambda: False)(),
                'metadata': self.registry.get_metadata(name),
            }
        return observers

    def enable_observer(self, name: str):
        """Enable an observer.

        Args:
            name: Observer name
        """
        self.registry.enable(name)
        print(f"Enabled observer: {name}")

    def disable_observer(self, name: str):
        """Disable an observer.

        Args:
            name: Observer name
        """
        self.registry.disable(name)
        print(f"Disabled observer: {name}")

    # ---- Convenience Methods ----

    def today(self, detail: str = 'medium') -> str:
        """Summarize today's activity.

        Args:
            detail: Detail level

        Returns:
            Summary of today
        """
        return self.summary(days=1, detail=detail)

    def yesterday(self, detail: str = 'medium') -> str:
        """Summarize yesterday's activity.

        Args:
            detail: Detail level

        Returns:
            Summary of yesterday
        """
        end_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = end_time - timedelta(days=1)

        observations = self.mall.query_by_timerange(start_time, end_time)
        context = self.agent._observations_to_context(observations, detail)

        prompt = f"Please summarize my activity from yesterday ({start_time.strftime('%Y-%m-%d')}):\n\n{context}"

        chat = self.agent._get_chat()
        messages = [
            {
                'role': 'system',
                'content': 'You are an AI assistant helping summarize daily activity.',
            },
            {'role': 'user', 'content': prompt},
        ]

        return chat(messages)

    def week(self, detail: str = 'medium') -> str:
        """Summarize this week's activity.

        Args:
            detail: Detail level

        Returns:
            Summary of this week
        """
        return self.summary(days=7, detail=detail)

    def stats(self) -> dict:
        """Get statistics about observations.

        Returns:
            Dictionary with statistics
        """
        stats = {
            'stores': {},
            'total_observations': 0,
        }

        for store_name in self.mall.list_stores():
            store = self.mall.get_store(store_name)
            count = len(store)
            stats['stores'][store_name] = count
            stats['total_observations'] += count

        return stats

    # ---- Semantic Search ----

    def search(self, query: str, n_results: int = 10) -> list:
        """Semantically search your activity.

        Args:
            query: Search query in natural language
            n_results: Number of results to return

        Returns:
            List of matching observations

        Example:
            >>> og = OG()
            >>> results = og.search("machine learning projects")
        """
        if not self.semantic_memory:
            print("Semantic search not enabled")
            return []

        results = self.semantic_memory.search(query, n_results=n_results)
        return results

    # ---- Pattern Detection & Alerts ----

    def get_alerts(self, acknowledged: Optional[bool] = None) -> list:
        """Get recent alerts.

        Args:
            acknowledged: Filter by acknowledgement status

        Returns:
            List of alerts
        """
        if not self.pattern_detector:
            print("Pattern detection not enabled")
            return []

        return self.pattern_detector.get_alerts(acknowledged=acknowledged)

    def enable_pattern(self, pattern_name: str):
        """Enable a specific pattern.

        Args:
            pattern_name: Pattern to enable
        """
        if self.pattern_detector:
            self.pattern_detector.enable_pattern(pattern_name)

    def disable_pattern(self, pattern_name: str):
        """Disable a specific pattern.

        Args:
            pattern_name: Pattern to disable
        """
        if self.pattern_detector:
            self.pattern_detector.disable_pattern(pattern_name)

    # ---- Context Management ----

    def create_context(self, name: str, **kwargs):
        """Create a new work context/project.

        Args:
            name: Context name
            **kwargs: Additional context parameters

        Returns:
            Created context
        """
        if not self.context_manager:
            print("Context management not enabled")
            return None

        return self.context_manager.create_context(name, **kwargs)

    def switch_context(self, name: str):
        """Switch to a different context.

        Args:
            name: Context to switch to
        """
        if self.context_manager:
            self.context_manager.switch_context(name)

    def list_contexts(self) -> list:
        """List all contexts.

        Returns:
            List of contexts
        """
        if not self.context_manager:
            return []

        return self.context_manager.get_all_contexts()

    # ---- Insights ----

    def productivity_patterns(self, days: int = 30) -> dict:
        """Analyze productivity patterns.

        Args:
            days: Days to analyze

        Returns:
            Productivity patterns
        """
        if self.insight_engine:
            return self.insight_engine.detect_productivity_patterns(days)
        return {}

    def deep_work_sessions(self, days: int = 7) -> list:
        """Find deep work sessions.

        Args:
            days: Days to analyze

        Returns:
            List of deep work sessions
        """
        if self.insight_engine:
            return self.insight_engine.identify_deep_work_sessions(days)
        return []

    def get_suggestions(self, days: int = 14) -> list:
        """Get optimization suggestions.

        Args:
            days: Days to analyze

        Returns:
            List of suggestions
        """
        if self.insight_engine:
            return self.insight_engine.suggest_optimizations(days)
        return []

    # ---- Export ----

    def export_json(self, output_file: str, days: Optional[int] = None) -> int:
        """Export observations to JSON.

        Args:
            output_file: Output file path
            days: Optional days to export

        Returns:
            Number of observations exported
        """
        if self.exporter:
            return self.exporter.to_json(output_file, days=days)
        return 0

    def export_markdown(self, output_file: str, days: Optional[int] = None) -> int:
        """Export observations to Markdown.

        Args:
            output_file: Output file path
            days: Optional days to export

        Returns:
            Number of observations exported
        """
        if self.exporter:
            return self.exporter.to_markdown(output_file, days=days)
        return 0

    def export_obsidian(self, vault_path: str, days: Optional[int] = None) -> int:
        """Export to Obsidian daily notes.

        Args:
            vault_path: Path to Obsidian vault
            days: Optional days to export

        Returns:
            Number of notes created
        """
        if self.exporter:
            return self.exporter.to_obsidian(vault_path, days=days)
        return 0

    # ---- Privacy ----

    def enable_encryption(self, password: Optional[str] = None):
        """Enable data encryption.

        Args:
            password: Optional encryption password
        """
        if self.privacy_controls:
            self.privacy_controls.enable_encryption(password)

    def exclude_pattern(self, pattern: str, pattern_type: str = 'general'):
        """Add pattern to exclude from observation.

        Args:
            pattern: Pattern to exclude
            pattern_type: Type ('general', 'url', 'app')
        """
        if self.privacy_controls:
            self.privacy_controls.add_exclude_pattern(pattern, pattern_type)

    def set_retention(self, days: int):
        """Set data retention policy.

        Args:
            days: Days to retain data
        """
        if self.privacy_controls:
            self.privacy_controls.set_retention_policy(days)
