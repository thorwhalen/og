"""Voice and natural language interface for OG.

This module provides voice command recognition and natural language
query processing for hands-free interaction with OG.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
import re


@dataclass
class VoiceCommand:
    """A recognized voice command."""

    raw_text: str
    intent: str  # 'query', 'action', 'report', 'summary'
    action: str  # 'get_summary', 'ask_question', 'generate_report'
    parameters: Dict[str, Any]
    confidence: float = 1.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class VoiceInterface:
    """Voice and natural language interface for OG."""

    # Command patterns
    PATTERNS = {
        'summary': [
            r'(?:what did i|show me what i) (?:do|work on) (?:today|yesterday|this week|last week)',
            r'(?:give me a )?summary(?: of)? (?:today|yesterday|this week)',
            r'(?:show|tell) me my (?:work|activity)(?: from)? (?:today|yesterday)',
        ],
        'question': [
            r'(?:what|how|when|where|why) (.+)',
            r'(?:show|tell|find) me (.+)',
            r'did i (.+)',
        ],
        'report': [
            r'(?:generate|create|make) a? (?:report|analysis)',
            r'(?:how was my )?productivity (?:this week|last week)',
        ],
        'standup': [
            r'(?:generate|create|give me) (?:a )?standup',
            r'(?:what\'s my )?standup',
        ],
        'focus': [
            r'(?:enable|start|activate) focus (?:mode|time)',
            r'(?:disable|stop|deactivate) focus (?:mode|time)',
        ],
    }

    def __init__(self, og_instance=None):
        """Initialize the voice interface.

        Args:
            og_instance: Optional OG instance
        """
        self.og = og_instance
        self._command_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default command handlers."""
        self._command_handlers['summary'] = self._handle_summary
        self._command_handlers['question'] = self._handle_question
        self._command_handlers['report'] = self._handle_report
        self._command_handlers['standup'] = self._handle_standup
        self._command_handlers['focus'] = self._handle_focus

    def process_voice_input(self, text: str) -> str:
        """Process voice input and return response.

        Args:
            text: Transcribed voice input

        Returns:
            Response text
        """
        # Parse command
        command = self.parse_command(text)

        # Execute command
        if command and command.intent in self._command_handlers:
            handler = self._command_handlers[command.intent]
            return handler(command)
        else:
            return "Sorry, I didn't understand that command."

    def parse_command(self, text: str) -> Optional[VoiceCommand]:
        """Parse natural language text into a command.

        Args:
            text: Input text

        Returns:
            VoiceCommand if recognized, None otherwise
        """
        text_lower = text.lower().strip()

        # Try to match patterns
        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    parameters = self._extract_parameters(text_lower, match)
                    return VoiceCommand(
                        raw_text=text,
                        intent=intent,
                        action=intent,
                        parameters=parameters,
                        confidence=0.8,
                    )

        # Fallback: treat as question
        if '?' in text or text_lower.startswith(('what', 'how', 'when', 'where', 'why', 'who')):
            return VoiceCommand(
                raw_text=text,
                intent='question',
                action='ask',
                parameters={'question': text},
                confidence=0.6,
            )

        return None

    def _extract_parameters(self, text: str, match: re.Match) -> Dict[str, Any]:
        """Extract parameters from matched text."""
        params = {}

        # Extract time period
        if 'today' in text:
            params['days'] = 1
            params['period'] = 'today'
        elif 'yesterday' in text:
            params['days'] = 1
            params['period'] = 'yesterday'
        elif 'this week' in text:
            params['days'] = 7
            params['period'] = 'week'
        elif 'last week' in text:
            params['days'] = 7
            params['period'] = 'last_week'

        # Extract question from match groups
        if match.groups():
            params['query'] = match.group(1).strip()

        return params

    def _handle_summary(self, command: VoiceCommand) -> str:
        """Handle summary command."""
        if not self.og:
            return "OG instance not available."

        days = command.parameters.get('days', 1)
        period = command.parameters.get('period', 'today')

        try:
            if period == 'today':
                summary = self.og.today()
            elif period == 'yesterday':
                summary = self.og.yesterday()
            elif period in ['week', 'last_week']:
                summary = self.og.week()
            else:
                summary = self.og.summary(days=days)

            return summary
        except Exception as e:
            return f"Error generating summary: {e}"

    def _handle_question(self, command: VoiceCommand) -> str:
        """Handle question command."""
        if not self.og:
            return "OG instance not available."

        question = command.parameters.get('question', command.raw_text)

        try:
            answer = self.og.ask(question)
            return answer
        except Exception as e:
            return f"Error answering question: {e}"

    def _handle_report(self, command: VoiceCommand) -> str:
        """Handle report generation command."""
        if not self.og:
            return "OG instance not available."

        days = command.parameters.get('days', 7)

        try:
            report = self.og.report(days=days, report_type='productivity')
            return report
        except Exception as e:
            return f"Error generating report: {e}"

    def _handle_standup(self, command: VoiceCommand) -> str:
        """Handle standup generation command."""
        if not self.og:
            return "OG instance not available."

        try:
            if hasattr(self.og, 'standup_generator'):
                standup = self.og.standup_generator.generate()
                return standup.to_text(format='simple')
            else:
                return "Standup generator not available."
        except Exception as e:
            return f"Error generating standup: {e}"

    def _handle_focus(self, command: VoiceCommand) -> str:
        """Handle focus mode command."""
        if not self.og:
            return "OG instance not available."

        if 'enable' in command.raw_text.lower() or 'start' in command.raw_text.lower():
            if hasattr(self.og, 'focus_mode'):
                self.og.focus_mode.enable()
                return "Focus mode enabled. Distractions will be minimized."
            else:
                return "Focus mode not available."
        else:
            if hasattr(self.og, 'focus_mode'):
                self.og.focus_mode.disable()
                return "Focus mode disabled."
            else:
                return "Focus mode not available."

    def register_handler(self, intent: str, handler: Callable):
        """Register a custom command handler.

        Args:
            intent: Command intent
            handler: Function to handle the command
        """
        self._command_handlers[intent] = handler


def text_to_speech(text: str) -> None:
    """Convert text to speech (placeholder for TTS integration).

    Args:
        text: Text to speak
    """
    # In production, integrate with TTS engine like:
    # - pyttsx3 (offline)
    # - Google TTS
    # - Amazon Polly
    print(f"[VOICE] {text}")


def speech_to_text() -> str:
    """Convert speech to text (placeholder for STT integration).

    Returns:
        Transcribed text
    """
    # In production, integrate with STT engine like:
    # - SpeechRecognition library
    # - Google Speech API
    # - Whisper (OpenAI)
    return input("[VOICE INPUT] > ")
