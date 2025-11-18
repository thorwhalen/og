# Own Ghost (OG)

**A daemon that observes your activities and helps you make AI contexts from them**

Own Ghost (OG) is a modular, extensible system for self-observation and intelligent reflection. It allows users to register observers—modular plugins that monitor activity across a wide range of sources: GitHub commits, keyboard input, application usage, browser history, and more. Each observer conforms to a shared interface, enabling consistent background tracking and asynchronous, timestamped data collection. These observations are serialized to local files, indexed continuously, and embedded into a vector database—making the full scope of one's recent activity instantly searchable.

On top of this observation layer, OG adds an intelligent agent capable of interpreting and summarizing behavior. Users can query their history conversationally or receive automated synopses such as "What did I work on yesterday?" or "What projects have I touched this week?" The AI agent doesn't just retrieve facts; it contextualizes them, helping users understand progress, identify patterns, and stay oriented.

## Features

### Core Capabilities
- 🔍 **Multi-Source Observation**: Track activity from GitHub, keyboard, browser, apps, filesystem, git commits, and terminal
- 🤖 **AI-Powered Analysis**: Get intelligent summaries and insights using LLMs
- 💬 **Conversational Queries**: Ask questions about your activity in natural language
- 📊 **Report Generation**: Create detailed productivity, technical, and comprehensive reports
- 🔧 **Modular & Extensible**: Easy to add custom observers for any activity source
- 💾 **Flexible Storage**: Uses `dol` for powerful, extensible storage backends
- 🎯 **Privacy-Focused**: Runs locally, your data stays on your machine

### Advanced Features
- 🔎 **Semantic Search**: Vector database-powered semantic search across all your observations
- 🚨 **Pattern Detection**: Automatic detection of behavioral patterns with customizable alerts
- 📁 **Context Management**: Automatic project/context detection and tracking
- 📈 **Advanced Insights**: Productivity analytics, deep work sessions, optimization suggestions
- 🌐 **Web Dashboard**: Real-time visualization and exploration of your activity
- 📤 **Export Integrations**: Export to Notion, Obsidian, Roam Research, Markdown, JSON, CSV
- 🔒 **Privacy Controls**: Encryption, sensitive data redaction, GDPR compliance, retention policies
- 📧 **Extended Observers**: Email, Calendar, Slack, Music, IDE tracking

### Productivity & Workflow Features
- 📝 **Automated Standup**: Generate daily standup reports automatically from your work
- 🔄 **Context Switching Analysis**: Measure cost of interruptions and identify flow states
- 🎤 **Meeting Intelligence**: Analyze meetings, extract action items, detect low-value meetings
- 💡 **Proactive Insights**: Anomaly detection, smart reminders, context-aware suggestions
- 🎓 **Learning Tracking**: Track skill development, detect learning modes, identify knowledge gaps
- 🎙️ **Voice Interface**: Voice commands and natural language queries
- 🔗 **Cross-Project Intelligence**: Find reusable patterns and similar solutions from past work
- ✅ **Task Integration**: Bi-directional sync with Todoist, Linear, Jira, GitHub Issues
- 🌐 **Browser Extension**: Enhanced capture with reading time, highlights, and scroll depth
- 🎯 **Focus Mode**: Active intervention to block distractions and protect calendar
- 📊 **Code Quality Trends**: Track complexity, coverage, and maintainability over time
- 😊 **Mood Correlation**: Track energy/mood and correlate with productivity
- 👥 **Team Features**: Privacy-preserving team insights and collaboration
- 💰 **Freelancer Tracking**: Auto-generate timesheets and invoices
- 📅 **Predictive Scheduling**: AI-powered optimal time scheduling for tasks

## Installation

```bash
# Basic installation
pip install og

# With all observers and features
pip install og[all]

# Specific feature sets
pip install og[web]           # Web dashboard
pip install og[privacy]       # Encryption and privacy features
pip install og[integrations]  # Email, Slack, Calendar, Music, Export integrations

# Development
pip install og[dev]

# Minimal installation (just core features)
pip install og[minimal]
```

## Quick Start

### Interactive Setup (Recommended)

Run the interactive setup wizard to configure OG:

```bash
# Launch the setup wizard
og setup

# Or use aliases
og config
og configure
```

The interactive wizard will guide you through:
- **Storage location** - Where to store observations
- **API keys** - OpenAI, GitHub, Slack, etc.
- **Observer configuration** - Enable/disable activity tracking
- **Feature toggles** - Turn features on/off
- **Privacy settings** - Encryption and retention policies
- **Dependency installation** - Automatically install required packages

You can run `og setup` anytime to:
- View current configuration
- Enable/disable observers and features
- Update API keys and credentials
- Install missing dependencies
- Export configuration

### Using the CLI

```bash
# Start the OG daemon to observe activity
og start

# Get a summary of today
og summary

# Get a summary of the last 7 days
og summary --days 7

# Ask a question about your activity
og ask "What did I work on yesterday?"

# Generate a productivity report
og report --days 7 --type productivity

# Check system status
og status

# List observers
og observers

# Get statistics
og stats
```

### Using the Python API

```python
from og import OG

# Create an OG instance
og = OG()

# Start observing (run as daemon)
og.start()

# Get a summary of today
print(og.today())

# Get a summary of the last week
print(og.week())

# Ask questions about your activity
answer = og.ask("What did I work on yesterday?")
print(answer)

# Generate reports
report = og.report(days=7, report_type='productivity')
print(report)

# Query recent activity
recent = og.recent_activity(days=1, event_type='git_commit')
for obs in recent:
    print(f"{obs.timestamp}: {obs.data}")

# Get statistics
stats = og.stats()
print(f"Total observations: {stats['total_observations']}")
```

## Available Observers

OG comes with several built-in observers:

### Core Observers (Enabled by Default)

| Observer | What it tracks | Requirements |
|----------|---------------|--------------|
| **GithubObserver** | GitHub activity (commits, PRs, issues) | `PyGithub` |
| **KeyboardObserver** | Keyboard activity (typing patterns) | `pynput` |
| **BrowserObserver** | Browser history (visited pages) | None (reads browser DBs) |
| **AppUsageObserver** | Application usage and switching | `psutil` |
| **FileSystemObserver** | File system changes | `watchdog` |
| **GitCommitObserver** | Local git commits | None (uses git CLI) |
| **TerminalHistoryObserver** | Shell command history | None (reads history files) |

### Extended Observers (Require Configuration)

| Observer | What it tracks | Requirements |
|----------|---------------|--------------|
| **EmailObserver** | Email activity (sent/received) | `google-api-python-client` or IMAP |
| **CalendarObserver** | Calendar events and meetings | `google-api-python-client` |
| **SlackObserver** | Slack messages and channels | `slack-sdk` |
| **MusicObserver** | Music listening (Spotify, Last.fm) | `spotipy`, `pylast` |
| **IDEObserver** | IDE projects and files (VS Code, PyCharm) | None (reads IDE files) |

### Observer Details

#### GitHub Observer
- Tracks commits, pull requests, issues, comments, and stars
- Polls GitHub API every 5 minutes (configurable)
- Requires: GitHub username and API token (via `GITHUB_TOKEN` env var)

#### Keyboard Observer
- Tracks typing activity and keystrokes per minute
- **Privacy-focused**: Does NOT log actual text, only metadata
- Aggregates statistics every minute

#### Browser Observer
- Reads browser history databases (Chrome, Firefox, Safari)
- Tracks visited URLs, page titles, and domains
- Optionally saves local page snapshots

#### App Usage Observer
- Tracks which applications you're using
- Measures time spent in each application
- Platform-specific: macOS, Linux, Windows

#### File System Observer
- Monitors file creations, modifications, and deletions
- Configurable watch paths and ignore patterns

#### Git Commit Observer
- Auto-discovers git repositories in common locations
- Tracks commits, messages, and changed files

#### Terminal History Observer
- Reads shell history files (bash, zsh, fish)
- Tracks commands executed

## Creating Custom Observers

You can easily create custom observers to track any activity:

```python
from og import BaseObserver, Observation

class CustomObserver(BaseObserver):
    def observe(self):
        while self._running:
            # Your observation logic here
            obs = self.create_observation(
                event_type='custom_event',
                data={'key': 'value'},
                tags=['custom'],
            )
            yield obs
            time.sleep(60)

# Register and use
from og import OG, ObserverRegistry

registry = ObserverRegistry()
registry['custom'] = CustomObserver()

og = OG(registry=registry)
```

See `examples/custom_observer.py` for more detailed examples.

## AI-Powered Features

OG uses AI (via `oa` or OpenAI API) to provide intelligent insights:

### Summaries
Get natural language summaries of your activity:
```python
# Brief summary
og.summary(days=1, detail='brief')

# Medium detail (default)
og.summary(days=7, detail='medium')

# Detailed analysis
og.summary(days=7, detail='detailed')
```

### Questions
Ask questions about your activity:
```python
og.ask("What did I work on yesterday?")
og.ask("How much time did I spend coding this week?")
og.ask("What websites did I visit most?")
```

### Reports
Generate comprehensive reports:
```python
# Productivity report
og.report(days=7, report_type='productivity')

# Technical report (focuses on coding/technical work)
og.report(days=7, report_type='technical')

# Comprehensive report
og.report(days=7, report_type='comprehensive')
```

## Advanced Features

### Semantic Search

Search your activity history using natural language queries with vector embeddings:

```python
# Enable semantic search (uses ChromaDB)
og = OG(enable_semantic_search=True)

# Search across all observations
results = og.search("machine learning work")

# Search within a timeframe
results = og.search(
    "debugging issues",
    start_date=datetime.now() - timedelta(days=7)
)
```

The semantic search indexes all observations in a vector database, allowing you to find related work even when exact keywords don't match.

### Pattern Detection & Alerts

OG can automatically detect behavioral patterns and alert you:

```python
# Enable pattern detection
og = OG(enable_patterns=True)

# Get recent alerts
alerts = og.get_alerts(hours=24)
for alert in alerts:
    print(f"{alert.severity}: {alert.message}")

# Built-in patterns detect:
# - Deep focus broken (frequent app switching)
# - Excessive browsing
# - Productivity milestones
# - Late night work
# - Frequent context switching

# Add custom patterns
from og.patterns import Pattern

def my_pattern_condition(observations):
    # Your custom logic
    return len(observations) > 10

custom_pattern = Pattern(
    name='my_pattern',
    description='Custom pattern',
    condition=my_pattern_condition,
    alert_message='Custom alert triggered!'
)

og.add_pattern(custom_pattern)
```

### Context Management

Automatically detect and track different projects/contexts:

```python
# Enable context management
og = OG(enable_contexts=True)

# Create a context for a project
og.create_context(
    'ml-project',
    description='Machine learning project',
    keywords=['ml', 'pytorch', 'training'],
    repos=['ml-repo']
)

# Switch to a context
og.switch_context('ml-project')

# Get context summary
summary = og.get_context_summary('ml-project')

# Auto-detect contexts from observations
og.auto_detect_contexts()
```

Contexts help organize your observations by project, making it easier to track work across different areas.

### Advanced Insights

Get deep analytics about your productivity and work patterns:

```python
# Enable insights engine
og = OG(enable_insights=True)

# Detect productivity patterns
patterns = og.productivity_patterns()
print(f"Peak hours: {patterns['peak_hours']}")
print(f"Most productive day: {patterns['best_day']}")

# Identify deep work sessions
sessions = og.deep_work_sessions(min_duration_minutes=30)
for session in sessions:
    print(f"Deep work: {session['start']} - {session['end']}")

# Get optimization suggestions
suggestions = og.suggest_optimizations()
for suggestion in suggestions:
    print(f"💡 {suggestion}")

# Track progress toward goals
og.track_goal('launch-feature', keywords=['feature', 'launch', 'deploy'])
progress = og.get_goal_progress('launch-feature')
```

### Web Dashboard

Launch a web interface to visualize your activity:

```python
# Start the web dashboard
og = OG(enable_web=True)
og.start_dashboard(port=5000)

# Visit http://localhost:5000 in your browser
```

The dashboard provides:
- Real-time activity timeline
- Heatmap of productivity
- Context/project visualization
- Recent alerts
- Semantic search interface
- Statistics and analytics

### Export & Integrations

Export your observations to various formats:

```python
from og.export import Exporter

exporter = Exporter(og)

# Export to JSON
exporter.to_json('observations.json', days=7)

# Export to Markdown
exporter.to_markdown('observations.md', days=7)

# Export to CSV
exporter.to_csv('observations.csv', days=7)

# Export to Notion
exporter.to_notion(
    database_id='your-database-id',
    notion_token='your-token',
    days=7
)

# Export to Obsidian daily notes
exporter.to_obsidian(
    vault_path='/path/to/obsidian/vault',
    days=7
)

# Export to Roam Research
exporter.to_roam('roam_export.json', days=7)
```

### Privacy & Security Controls

OG provides comprehensive privacy controls:

```python
from og.privacy import PrivacyControls

privacy = PrivacyControls()

# Enable encryption for stored observations
privacy.enable_encryption(password='your-secure-password')

# Redact sensitive data automatically
text = "Email me at user@example.com"
redacted = privacy.redact_sensitive_data(text)
# Output: "Email me at [REDACTED]"

# Exclude specific patterns (URLs, apps)
privacy.add_exclude_pattern(r'bank\.com', 'url')
privacy.add_exclude_pattern(r'1Password', 'app')

# Set retention policy (auto-delete old data)
privacy.set_retention_policy(days=90)

# Generate privacy report
report = privacy.generate_privacy_report(observations)
print(f"PII detected: {report['pii_detected']}")

# Export GDPR-compliant data
privacy.export_gdpr_data(user_id='user@example.com')
```

Privacy features include:
- **Encryption**: Fernet encryption for observation data
- **Sensitive Data Redaction**: Automatic removal of emails, SSNs, passwords, API keys
- **Exclude Patterns**: Skip tracking for specific URLs/apps
- **Retention Policies**: Auto-delete observations after specified period
- **Anonymization**: Hash personally identifiable information
- **GDPR Compliance**: Export user data in machine-readable format

## Productivity & Workflow Features

### Automated Standup Generation

Generate daily standup reports automatically:

```python
from og.standup import StandupGenerator

generator = StandupGenerator(og)

# Generate standup for today
standup = generator.generate()

# Format as text
print(standup.to_text())

# Format for Slack
print(standup.to_text(format='slack'))

# Generate weekly report
weekly = generator.generate_weekly()
```

### Context Switching Cost Analysis

Analyze the cost of context switching:

```python
from og.switching_cost import SwitchingCostAnalyzer

analyzer = SwitchingCostAnalyzer(og)

# Analyze last 7 days
report = analyzer.analyze(days=7)

print(f"Total switches: {report.total_switches}")
print(f"Estimated cost: {report.estimated_cost_minutes:.0f} minutes")
print(f"Flow sessions: {len(report.flow_sessions)}")

# Get markdown report
print(report.to_markdown())
```

### Meeting Intelligence

Analyze meetings for efficiency and value:

```python
from og.meetings import MeetingAnalyzer

analyzer = MeetingAnalyzer(og)

# Analyze a single meeting
meeting = analyzer.analyze_meeting(
    title='Sprint Planning',
    start_time=start,
    end_time=end,
    transcript=transcript,
)

print(f"Efficiency: {meeting.efficiency_score}/10")
print(f"Action items: {len(meeting.action_items)}")

# Get insights across all meetings
insights = analyzer.analyze_meetings(days=30)
print(insights.to_markdown())
```

### Proactive Insights

Get proactive suggestions and anomaly detection:

```python
from og.proactive import ProactiveInsightEngine

engine = ProactiveInsightEngine(og)

# Analyze and get insights
report = engine.analyze()

for anomaly in report.anomalies:
    print(f"⚠️ {anomaly.title}: {anomaly.message}")

for reminder in report.reminders:
    print(f"📝 {reminder.message}")

for suggestion in report.suggestions:
    print(f"💡 {suggestion.message}")
```

### Learning Trajectory Tracking

Track skill development over time:

```python
from og.learning import LearningTracker

tracker = LearningTracker(og)

# Track learning over 30 days
report = tracker.track(days=30)

print(f"New skills: {len(report.new_skills)}")
for skill in report.new_skills:
    print(f"  - {skill.name}: {skill.proficiency_score:.0%}")

print(f"\nKnowledge gaps:")
for gap in report.knowledge_gaps:
    print(f"  - {gap.skill}: {gap.recommendation}")
```

### Voice Interface

Use voice commands:

```python
from og.voice import VoiceInterface

voice = VoiceInterface(og)

# Process voice input
response = voice.process_voice_input("What did I work on today?")
print(response)

# Common commands:
# - "What did I work on today?"
# - "Give me a summary of this week"
# - "Generate a standup"
# - "Enable focus mode"
```

### Focus Mode

Block distractions during deep work:

```python
from og.focus_mode import FocusMode

focus = FocusMode(og)

# Enable focus mode for 90 minutes
focus.enable(duration_minutes=90)

# This will:
# - Block distracting websites
# - Silence notifications
# - Protect calendar from new meetings

# Disable when done
focus.disable()
```

### Task Integration

Sync with task managers:

```python
from og.tasks import TaskIntegration

tasks = TaskIntegration(og)

# Sync tasks from Todoist, Linear, Jira
tasks.sync_tasks(sources=['linear', 'github'])

# Auto-complete tasks based on detected work
tasks.auto_complete_tasks()

# Generate timesheet
timesheet = tasks.generate_timesheet(start_date, end_date)
print(f"Total hours: {timesheet['total_hours']}")
```

### Freelancer Time Tracking

Generate timesheets and invoices:

```python
from og.timesheet import FreelancerTimeTracker
from decimal import Decimal

tracker = FreelancerTimeTracker(og)

# Map repos to clients
tracker.map_project('my-repo', 'ProjectX', 'ClientA', Decimal('150.00'))

# Generate timesheet
entries = tracker.generate_timesheet(start_date, end_date)

# Generate invoice
invoice = tracker.generate_invoice(
    client='ClientA',
    start_date=start_date,
    end_date=end_date,
    invoice_number='INV-001',
)

print(tracker.export_invoice_markdown(invoice))
```

### Predictive Scheduling

Optimize your schedule with AI:

```python
from og.scheduling import PredictiveScheduler

scheduler = PredictiveScheduler(og)

# Learn patterns from history
scheduler.learn_patterns()

# Predict optimal time for deep work
optimal_times = scheduler.predict_optimal_time('deep_work', duration_hours=2)

# Get schedule recommendations
recommendations = scheduler.optimize_schedule(start_date, end_date)

for rec in recommendations:
    print(rec.message)
```

### Mood Correlation

Track mood and correlate with productivity:

```python
from og.mood import MoodTracker

mood = MoodTracker(og)

# Check in
mood.check_in(mood=4, energy=3, stress=2, notes='Feeling good')

# Get work suggestions based on energy
suggestion = mood.suggest_work_type()
print(suggestion)

# Correlate mood with productivity
correlations = mood.correlate_with_productivity()

# Detect burnout risk
risk = mood.detect_burnout_risk()
if risk:
    print(f"⚠️ {risk['risk_level']} burnout risk")
```

## Configuration

OG provides multiple ways to configure the system:

### Interactive Setup Wizard (Recommended)

The easiest way to configure OG is through the interactive wizard:

```bash
og setup
```

The wizard provides a menu-driven interface for:
- **Quick Setup** - Essential settings for first-time users
- **Manage Observers** - Enable/disable activity tracking
- **Manage Features** - Toggle advanced features
- **Manage Credentials** - Set API keys and tokens
- **Manage Privacy** - Configure encryption and retention
- **Install Dependencies** - Automatically install required packages
- **View Configuration** - See all current settings
- **Export Configuration** - Export as environment variables

### Configuration File

Configuration is stored in `~/.og/config.json`. You can edit this file directly or use the setup wizard.

Example configuration:
```json
{
  "storage_dir": "/home/user/.og/observations",
  "openai_api_key": "sk-...",
  "enable_github_observer": true,
  "enable_semantic_search": true,
  "enable_standup": true
}
```

### Python API

Configure OG programmatically:

```python
from og.config import ConfigManager

# Load configuration
config_manager = ConfigManager()

# View settings
print(config_manager.config.ai_model)

# Update settings
config_manager.set('enable_focus_mode', True)
config_manager.update(
    openai_api_key='sk-...',
    enable_web_dashboard=True
)

# Get enabled observers and features
observers = config_manager.get_enabled_observers()
features = config_manager.get_enabled_features()

# Export as environment variables
env_vars = config_manager.export_env()
```

### Environment Variables

You can also use environment variables (these take precedence):

Core settings:
- `OPENAI_API_KEY`: Your OpenAI API key for AI features
- `OG_MODEL`: Model to use (default: `gpt-4`)

Observer credentials:
- `GITHUB_TOKEN`: GitHub API token for GitHub observer
- `SLACK_TOKEN`: Slack API token for Slack observer
- `SPOTIFY_CLIENT_ID`: Spotify API client ID
- `SPOTIFY_CLIENT_SECRET`: Spotify API client secret
- `LASTFM_API_KEY`: Last.fm API key for music tracking
- `NOTION_TOKEN`: Notion integration token for exports

### Storage

By default, observations are stored in:
- `~/.og/observations/` (or system temp directory)

You can customize the storage location:
```python
og = OG(storage_dir='/path/to/storage')
```

## Architecture

OG follows a modular architecture inspired by patterns from `ef`, `dol`, `i2`, and `meshed`:

```
┌────────────────────────────────────────────────────────────┐
│  Observers (Modular Activity Trackers)                     │
│  Core: GitHub, Keyboard, Browser, Apps, Git, FS, Terminal  │
│  Extended: Email, Calendar, Slack, Music, IDE              │
└──────────────┬─────────────────────────────────────────────┘
               │ Observations
               ↓
┌────────────────────────────────────────────────────────────┐
│  Storage & Indexing Layer                                  │
│  - ObservationMall (dol-based store of stores)             │
│  - SemanticMemory (ChromaDB vector database)               │
│  - Privacy Controls (encryption, redaction)                │
└──────────────┬─────────────────────────────────────────────┘
               │ Query/Analysis
               ↓
┌────────────────────────────────────────────────────────────┐
│  Intelligence Layer                                         │
│  - AI Agent (oa-based): summaries, Q&A, reports            │
│  - Pattern Detector: behavioral patterns & alerts          │
│  - Context Manager: project/context tracking               │
│  - Insight Engine: productivity analytics                  │
└──────────────┬─────────────────────────────────────────────┘
               │ Interfaces
               ↓
┌────────────────────────────────────────────────────────────┐
│  User Interfaces                                            │
│  - CLI (og command)                                         │
│  - Python API (OG class)                                    │
│  - Web Dashboard (Flask)                                    │
│  - Export/Integrations (Notion, Obsidian, etc.)            │
└────────────────────────────────────────────────────────────┘
```

### Core Concepts

1. **Observers**: Modular components that track specific activity sources
2. **Observations**: Timestamped data points with event type, data, and metadata
3. **ObservationMall**: A "mall" (store of stores) that organizes observations by type
4. **Registry**: Manages observer registration and lifecycle
5. **SemanticMemory**: Vector database for semantic search across observations
6. **Pattern Detector**: Detects behavioral patterns and generates alerts
7. **Context Manager**: Tracks and organizes work by project/context
8. **Insight Engine**: Advanced analytics and optimization suggestions
9. **AI Agent**: Interprets observations and generates insights
10. **Privacy Controls**: Encryption, redaction, and data protection

## Privacy & Security

- **Local-First**: All data is stored locally on your machine
- **Privacy-Focused**: Keyboard observer doesn't log actual text, only metadata
- **Encryption**: Optional Fernet encryption for all stored observations
- **Sensitive Data Redaction**: Automatic removal of emails, SSNs, passwords, API keys
- **Exclude Patterns**: Skip tracking for specific URLs, apps, or patterns
- **Retention Policies**: Auto-delete observations after a specified period
- **Anonymization**: Hash personally identifiable information
- **GDPR Compliance**: Export user data in machine-readable format
- **Configurable**: Disable any observer you don't want
- **Transparent**: All code is open source and auditable

## Dependencies

### Core Dependencies

OG uses several powerful packages:

- **dol**: Storage layer (key-value stores)
- **i2**: Signature operations and wrapping
- **meshed**: DAG composition for workflows
- **oa**: AI/LLM interface (or OpenAI directly)
- **chromadb**: Vector database for semantic search
- **sentence-transformers**: Text embeddings

### Observer Dependencies

- **PyGithub**: GitHub API access
- **pynput**: Keyboard/mouse tracking
- **watchdog**: File system monitoring
- **psutil**: Process monitoring

### Advanced Features (Optional)

- **flask**: Web dashboard
- **plotly**: Visualization
- **cryptography**: Encryption and privacy
- **google-api-python-client**: Gmail and Calendar integration
- **slack-sdk**: Slack integration
- **spotipy**: Spotify integration
- **pylast**: Last.fm integration
- **notion-client**: Notion export

## Examples

See the `examples/` directory for:
- `basic_usage.py`: Common usage patterns
- `custom_observer.py`: Creating custom observers

## Contributing

Contributions welcome! OG is designed to be extensible and modular.

Recent additions (Phase 1):
- ✅ Semantic search with vector database
- ✅ Pattern detection and alerts
- ✅ Context/project management
- ✅ Advanced insights and analytics
- ✅ Web dashboard with visualizations
- ✅ Email, Calendar, Slack, Music, and IDE observers
- ✅ Export integrations (Notion, Obsidian, Roam)
- ✅ Privacy and security controls

Recent additions (Phase 2 - Productivity & Workflow):
- ✅ Automated standup generation
- ✅ Context switching cost analysis
- ✅ Meeting intelligence and analysis
- ✅ Proactive insights and anomaly detection
- ✅ Learning trajectory tracking
- ✅ Voice/natural language interface
- ✅ Cross-project intelligence
- ✅ Task manager integrations (Todoist, Linear, Jira, GitHub)
- ✅ Browser extension for enhanced capture
- ✅ Focus mode with active intervention
- ✅ Code quality trends tracking
- ✅ Energy/mood correlation
- ✅ Team collaboration features
- ✅ Freelancer time tracking and invoicing
- ✅ Predictive scheduling

Areas for future contribution:
- Mobile apps for on-the-go querying
- Enhanced machine learning models for predictions
- Additional IDE integrations (IntelliJ, Sublime, Emacs, Neovim)
- Cross-device synchronization (with encryption)
- Plugin marketplace for community observers
- Advanced data visualization and dashboards
- Integration with more productivity tools (Asana, Monday.com, ClickUp)

## License

MIT

## Related Projects

- **ef**: Lightweight embedding framework
- **dol**: Dictionary-of-locations (storage abstraction)
- **i2**: Interface to interface (signature manipulation)
- **meshed**: DAG-based workflow composition
- **oa**: OpenAI interface
