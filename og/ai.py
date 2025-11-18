"""AI agent layer for OG (Own Ghost).

This module provides AI-powered analysis and querying of observations.
"""

import os
from datetime import datetime, timedelta
from typing import Any, Optional

from og.storage import ObservationMall


class OGAgent:
    """AI agent for analyzing and querying observations.

    This agent uses LLMs to:
    - Summarize activity over time periods
    - Answer questions about activity
    - Generate reports
    - Identify patterns and insights
    """

    def __init__(
        self,
        mall: ObservationMall,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the OG agent.

        Args:
            mall: ObservationMall containing observations
            model: Model to use (defaults to environment or gpt-4)
            api_key: API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.mall = mall
        self.model = model or os.getenv('OG_MODEL', 'gpt-4')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

        # Lazy import and initialization
        self._chat = None

    def _get_chat(self):
        """Get chat interface (lazy loading).

        Uses oa package if available, falls back to direct OpenAI API.
        """
        if self._chat is not None:
            return self._chat

        try:
            # Try to use oa package
            from oa import openai_chat

            self._chat = openai_chat(model=self.model, api_key=self.api_key)
        except ImportError:
            # Fall back to OpenAI SDK
            try:
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key)

                def chat(messages):
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    )
                    return response.choices[0].message.content

                self._chat = chat
            except ImportError:
                raise ImportError(
                    "Either oa or openai package is required for AI functionality. "
                    "Install with: pip install oa OR pip install openai"
                )

        return self._chat

    def summarize_period(
        self,
        days: int = 1,
        end_time: Optional[datetime] = None,
        detail_level: str = 'medium',
    ) -> str:
        """Summarize activity over a time period.

        Args:
            days: Number of days to summarize
            end_time: End of time period (defaults to now)
            detail_level: Level of detail ('brief', 'medium', 'detailed')

        Returns:
            Summary text
        """
        if end_time is None:
            end_time = datetime.now()

        start_time = end_time - timedelta(days=days)

        # Query observations
        observations = self.mall.query_by_timerange(start_time, end_time)

        if not observations:
            return f"No activity recorded in the last {days} days."

        # Prepare context for LLM
        context = self._observations_to_context(observations, detail_level)

        # Create prompt
        prompt = self._create_summary_prompt(
            context, days, start_time, end_time, detail_level
        )

        # Get summary from LLM
        chat = self._get_chat()
        messages = [
            {
                'role': 'system',
                'content': 'You are an AI assistant helping summarize a user\'s activity. '
                'Be concise, insightful, and focus on what the user accomplished and how they spent their time.',
            },
            {'role': 'user', 'content': prompt},
        ]

        summary = chat(messages)
        return summary

    def answer_question(self, question: str, days: Optional[int] = 7) -> str:
        """Answer a question about recent activity.

        Args:
            question: User's question
            days: How many days of history to consider

        Returns:
            Answer to the question
        """
        # Get recent observations
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        observations = self.mall.query_by_timerange(start_time, end_time)

        if not observations:
            return f"No activity recorded in the last {days} days to answer this question."

        # Prepare context
        context = self._observations_to_context(observations, detail_level='detailed')

        # Create prompt
        prompt = f"""Based on my activity from the last {days} days, please answer this question:

{question}

Activity data:
{context}

Please provide a specific, detailed answer based on the data above."""

        # Get answer from LLM
        chat = self._get_chat()
        messages = [
            {
                'role': 'system',
                'content': 'You are an AI assistant helping answer questions about a user\'s activity. '
                'Be specific and cite the data when answering.',
            },
            {'role': 'user', 'content': prompt},
        ]

        answer = chat(messages)
        return answer

    def generate_report(
        self,
        days: int = 7,
        include_web_research: bool = False,
        report_type: str = 'productivity',
    ) -> str:
        """Generate a detailed report about activity.

        Args:
            days: Number of days to analyze
            include_web_research: Whether to include web research (not implemented yet)
            report_type: Type of report ('productivity', 'technical', 'comprehensive')

        Returns:
            Generated report
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        observations = self.mall.query_by_timerange(start_time, end_time)

        if not observations:
            return f"No activity recorded in the last {days} days."

        # Prepare detailed context
        context = self._observations_to_context(observations, detail_level='detailed')

        # Create report prompt based on type
        prompts = {
            'productivity': """Generate a productivity report analyzing:
- Time allocation across different activities
- Focus patterns (when was I most productive?)
- Key accomplishments and milestones
- Areas for improvement
- Recommendations for better productivity""",
            'technical': """Generate a technical report analyzing:
- Code and project work
- Technologies and tools used
- Repositories and commits
- Technical learning and research
- Technical achievements""",
            'comprehensive': """Generate a comprehensive report covering:
- Overall activity summary
- Productivity patterns
- Technical work
- Learning and research
- Time management insights
- Key accomplishments
- Recommendations""",
        }

        report_prompt = prompts.get(report_type, prompts['comprehensive'])

        prompt = f"""{report_prompt}

Activity period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ({days} days)

Activity data:
{context}

Please generate a well-structured, insightful report."""

        # Get report from LLM
        chat = self._get_chat()
        messages = [
            {
                'role': 'system',
                'content': 'You are an AI assistant helping generate insightful reports about user activity. '
                'Be analytical, identify patterns, and provide actionable insights.',
            },
            {'role': 'user', 'content': prompt},
        ]

        report = chat(messages)
        return report

    def _observations_to_context(
        self, observations: list, detail_level: str = 'medium'
    ) -> str:
        """Convert observations to context for LLM.

        Args:
            observations: List of observations
            detail_level: Level of detail to include

        Returns:
            Formatted context string
        """
        # Group observations by type
        by_type: dict[str, list] = {}
        for obs in observations:
            if obs.event_type not in by_type:
                by_type[obs.event_type] = []
            by_type[obs.event_type].append(obs)

        # Format each type
        context_parts = []

        for event_type, obs_list in by_type.items():
            context_parts.append(f"\n### {event_type} ({len(obs_list)} events)")

            if detail_level == 'brief':
                # Just show count and sample
                if obs_list:
                    sample = obs_list[0]
                    context_parts.append(
                        f"Example: {self._format_observation(sample, brief=True)}"
                    )

            elif detail_level == 'medium':
                # Show up to 10 recent events
                for obs in obs_list[:10]:
                    context_parts.append(f"- {self._format_observation(obs)}")
                if len(obs_list) > 10:
                    context_parts.append(f"... and {len(obs_list) - 10} more")

            else:  # detailed
                # Show all events
                for obs in obs_list:
                    context_parts.append(f"- {self._format_observation(obs)}")

        return '\n'.join(context_parts)

    def _format_observation(self, obs, brief: bool = False) -> str:
        """Format an observation for display.

        Args:
            obs: Observation object
            brief: Whether to use brief format

        Returns:
            Formatted string
        """
        if brief:
            return f"{obs.timestamp.strftime('%Y-%m-%d %H:%M')} - {obs.event_type}"

        # Format based on event type
        if obs.event_type == 'github_push':
            commits = obs.data.get('commits', [])
            messages = [c.get('message', '') for c in commits[:3]]
            return f"[{obs.timestamp.strftime('%H:%M')}] GitHub push to {obs.data.get('repo')}: {', '.join(messages)}"

        elif obs.event_type == 'browser_visit':
            return f"[{obs.timestamp.strftime('%H:%M')}] Visited: {obs.data.get('title')} ({obs.data.get('domain')})"

        elif obs.event_type == 'app_usage':
            duration = obs.data.get('duration_seconds', 0)
            mins = int(duration / 60)
            return f"[{obs.timestamp.strftime('%H:%M')}] Used {obs.data.get('application')} for {mins} minutes"

        elif obs.event_type == 'git_commit':
            return f"[{obs.timestamp.strftime('%H:%M')}] Git commit in {obs.data.get('repo')}: {obs.data.get('message')}"

        elif obs.event_type == 'keyboard_activity':
            kpm = obs.data.get('keystrokes_per_minute', 0)
            return f"[{obs.timestamp.strftime('%H:%M')}] Typing activity: {kpm:.0f} keys/min"

        elif obs.event_type == 'terminal_command':
            return f"[{obs.timestamp.strftime('%H:%M')}] Command: {obs.data.get('command')}"

        else:
            # Generic format
            return f"[{obs.timestamp.strftime('%H:%M')}] {obs.event_type}: {obs.data}"

    def _create_summary_prompt(
        self,
        context: str,
        days: int,
        start_time: datetime,
        end_time: datetime,
        detail_level: str,
    ) -> str:
        """Create a prompt for summarizing activity.

        Args:
            context: Formatted observation context
            days: Number of days
            start_time: Start of period
            end_time: End of period
            detail_level: Level of detail

        Returns:
            Prompt string
        """
        detail_instructions = {
            'brief': 'Provide a brief summary (2-3 sentences) highlighting the main activities.',
            'medium': 'Provide a medium-length summary (1-2 paragraphs) covering key activities and patterns.',
            'detailed': 'Provide a detailed summary (3-4 paragraphs) with insights into activities, patterns, and accomplishments.',
        }

        instruction = detail_instructions.get(detail_level, detail_instructions['medium'])

        return f"""Please summarize my activity from {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')} ({days} days).

{instruction}

Activity data:
{context}

Focus on:
1. Main activities and how time was spent
2. Key accomplishments or milestones
3. Any notable patterns or insights
4. Overall productivity assessment"""
