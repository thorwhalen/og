# Instructions for Refactoring OG to Lightweight Architecture

## Problem Statement

The current **Own Ghost (OG)** implementation uses an **active observer pattern** where background daemon processes continuously monitor various activities (keyboard, apps, filesystem, etc.). This approach:

1. **High resource consumption**: Multiple daemon processes running continuously
2. **Complex setup**: Requires managing multiple background services
3. **Redundant data collection**: macOS/Linux/Windows already collect much of this data
4. **Privacy concerns**: Active monitoring feels intrusive to some users

## Proposed Solution

Implement a **passive, on-demand architecture** that:

1. **Leverages existing OS data sources** (macOS Screen Time DB, Windows activity logs, Linux proc)
2. **Queries data only when needed** (no continuous background processes)
3. **Uses lightweight system APIs** (psutil) for real-time queries
4. **Optionally integrates ActivityWatch** for users who want automatic tracking without the overhead
5. **Keeps legacy observers as opt-in** for power users who need them

### Key Tools & Resources

**Primary Data Sources:**

1. **macOS Screen Time Database** (`knowledgeC.db`)
   - Location: `~/Library/Application Support/Knowledge/knowledgeC.db`
   - Already collecting app usage, web activity, media usage
   - Resources:
     - https://stackoverflow.com/questions/66935741/how-to-get-screen-time-data-on-macos
     - https://medium.com/@carmenliu0208/how-to-retrieve-screen-time-data-on-macos-via-the-command-line-66e269278ba5
     - https://github.com/mac4n6/APOLLO (pre-built queries)

2. **psutil** (Cross-platform system monitoring)
   - Python library for on-demand system queries
   - No daemon needed - query when needed
   - Resources:
     - https://github.com/giampaolo/psutil
     - https://psutil.readthedocs.io/
     - PyPI: https://pypi.org/project/psutil/

3. **ActivityWatch** (Optional lightweight tracker)
   - Open source, privacy-first, local storage
   - REST API for data access
   - Resources:
     - https://github.com/ActivityWatch/activitywatch
     - https://activitywatch.net/
     - https://docs.activitywatch.net/

4. **RescueTime** (Optional commercial alternative)
   - Lightweight, comprehensive API
   - Resources:
     - https://www.rescuetime.com/
     - https://www.rescuetime.com/rtx/developers

## Implementation Tasks

### 1. Create New Lightweight Observer Architecture

Create `og/observers/passive/` directory with:

**`og/observers/passive/base.py`** - Base class for passive observers
```python
"""
Base class for passive observers that query existing data sources.
No background processes - data retrieved on-demand.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional
from datetime import datetime, timedelta
from og.base import Observation

class PassiveObserver(ABC):
    """Base class for observers that query existing data rather than monitor actively."""
    
    @abstractmethod
    def query(
        self, 
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> Iterator[Observation]:
        """
        Query historical data from the source.
        
        Args:
            start_time: Start of time range (None = beginning of available data)
            end_time: End of time range (None = now)
            **kwargs: Additional query parameters
            
        Yields:
            Observation objects from the queried time range
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this data source is available on this system."""
        pass
```

**`og/observers/passive/macos_screentime.py`** - Query macOS Screen Time database
```python
"""
Query macOS Screen Time database (knowledgeC.db).
No background process needed - macOS already collects this data.

References:
- https://stackoverflow.com/questions/66935741/how-to-get-screen-time-data-on-macos
- https://github.com/mac4n6/APOLLO/blob/main/modules/knowledge_app_usage.txt
"""
import sqlite3
import platform
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterator, Optional
from og.observers.passive.base import PassiveObserver
from og.base import Observation

class MacOSScreenTimeObserver(PassiveObserver):
    """Query macOS Screen Time database for app usage data."""
    
    DB_PATH = Path.home() / "Library/Application Support/Knowledge/knowledgeC.db"
    # macOS uses 2001-01-01 as epoch reference
    COCOA_EPOCH_OFFSET = 978307200
    
    def is_available(self) -> bool:
        return platform.system() == "Darwin" and self.DB_PATH.exists()
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> Iterator[Observation]:
        if not self.is_available():
            return
        
        # Build query with time filters
        query = """
        SELECT 
            DATETIME(ZOBJECT.ZSTARTDATE + 978307200, 'UNIXEPOCH') AS start,
            DATETIME(ZOBJECT.ZENDDATE + 978307200, 'UNIXEPOCH') AS end,
            ZOBJECT.ZVALUESTRING AS bundle_id,
            (ZOBJECT.ZENDDATE - ZOBJECT.ZSTARTDATE) AS duration_seconds
        FROM ZOBJECT
        WHERE ZSTREAMNAME = '/app/usage'
        """
        
        params = []
        if start_time:
            query += " AND ZOBJECT.ZSTARTDATE >= ?"
            params.append((start_time.timestamp() - self.COCOA_EPOCH_OFFSET))
        if end_time:
            query += " AND ZOBJECT.ZENDDATE <= ?"
            params.append((end_time.timestamp() - self.COCOA_EPOCH_OFFSET))
        
        query += " ORDER BY ZOBJECT.ZSTARTDATE"
        
        with sqlite3.connect(f"file:{self.DB_PATH}?mode=ro", uri=True) as conn:
            cursor = conn.execute(query, params)
            for row in cursor:
                start_str, end_str, bundle_id, duration = row
                timestamp = datetime.fromisoformat(start_str)
                
                yield Observation(
                    timestamp=timestamp,
                    event_type='app_usage',
                    data={
                        'bundle_id': bundle_id,
                        'duration': duration,
                        'start': start_str,
                        'end': end_str,
                    },
                    tags=['macos', 'screentime', 'passive'],
                )
```

**`og/observers/passive/psutil_observer.py`** - Real-time system queries
```python
"""
Use psutil for on-demand system state queries.
No background process - query current state when needed.

Reference:
- https://github.com/giampaolo/psutil
- https://psutil.readthedocs.io/
"""
import psutil
from datetime import datetime
from typing import Iterator, Optional
from og.observers.passive.base import PassiveObserver
from og.base import Observation

class PsutilObserver(PassiveObserver):
    """Query current system state using psutil."""
    
    def is_available(self) -> bool:
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> Iterator[Observation]:
        """Get current system state (ignore time range for real-time queries)."""
        if not self.is_available():
            return
        
        now = datetime.now()
        
        # Current processes
        for proc in psutil.process_iter(['pid', 'name', 'username', 'create_time']):
            try:
                yield Observation(
                    timestamp=now,
                    event_type='process_snapshot',
                    data={
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'username': proc.info['username'],
                        'started': datetime.fromtimestamp(proc.info['create_time']).isoformat(),
                    },
                    tags=['psutil', 'snapshot', 'passive'],
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
```

**`og/observers/passive/activitywatch.py`** - Query ActivityWatch API
```python
"""
Query ActivityWatch REST API for historical data.
ActivityWatch runs as lightweight background service.

Reference:
- https://github.com/ActivityWatch/activitywatch
- https://docs.activitywatch.net/en/latest/api.html
"""
import requests
from datetime import datetime, timedelta
from typing import Iterator, Optional
from og.observers.passive.base import PassiveObserver
from og.base import Observation

class ActivityWatchObserver(PassiveObserver):
    """Query ActivityWatch API for historical activity data."""
    
    def __init__(self, base_url: str = "http://localhost:5600"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/0"
    
    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.api_url}/buckets", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> Iterator[Observation]:
        if not self.is_available():
            return
        
        # Get list of buckets
        buckets = requests.get(f"{self.api_url}/buckets").json()
        
        for bucket_id, bucket_info in buckets.items():
            # Query events from bucket
            params = {}
            if start_time:
                params['start'] = start_time.isoformat()
            if end_time:
                params['end'] = end_time.isoformat()
            
            events_url = f"{self.api_url}/buckets/{bucket_id}/events"
            events = requests.get(events_url, params=params).json()
            
            for event in events:
                yield Observation(
                    timestamp=datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')),
                    event_type=f"aw_{bucket_info['type']}",
                    data=event['data'],
                    tags=['activitywatch', 'passive'],
                )
```

### 2. Create Lightweight OG Core

**`og/lightweight.py`** - New lightweight OG implementation
```python
"""
Lightweight OG implementation using passive observers.
No background daemons - queries data on-demand.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Iterator
from og.base import Observation
from og.observers.passive.base import PassiveObserver
from og.observers.passive.macos_screentime import MacOSScreenTimeObserver
from og.observers.passive.psutil_observer import PsutilObserver
from og.observers.passive.activitywatch import ActivityWatchObserver

class LightweightOG:
    """
    Lightweight OG that queries existing data sources instead of running daemons.
    
    This is the default and recommended approach for most users.
    Uses:
    - macOS Screen Time database (if on macOS)
    - psutil for real-time system queries
    - ActivityWatch API (if installed)
    - Browser history, git repos, shell history (read-only)
    """
    
    def __init__(self):
        self.observers: List[PassiveObserver] = []
        self._register_observers()
    
    def _register_observers(self):
        """Register all available passive observers."""
        potential_observers = [
            MacOSScreenTimeObserver(),
            PsutilObserver(),
            ActivityWatchObserver(),
        ]
        
        for observer in potential_observers:
            if observer.is_available():
                self.observers.append(observer)
    
    def available_observers(self) -> List[str]:
        """List available data sources."""
        return [obs.__class__.__name__ for obs in self.observers]
    
    def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
    ) -> Iterator[Observation]:
        """
        Query all available data sources.
        
        Args:
            start_time: Start of time range (None = last 24 hours)
            end_time: End of time range (None = now)
            event_types: Filter by event types (None = all)
        
        Yields:
            Observations from all available sources
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        if end_time is None:
            end_time = datetime.now()
        
        for observer in self.observers:
            for obs in observer.query(start_time, end_time):
                if event_types is None or obs.event_type in event_types:
                    yield obs
    
    def today(self) -> str:
        """Get summary of today's activity."""
        from og.agent import summarize_observations
        
        start = datetime.now().replace(hour=0, minute=0, second=0)
        observations = list(self.query(start_time=start))
        return summarize_observations(observations, detail='medium')
    
    def ask(self, question: str) -> str:
        """Ask a question about your activity."""
        from og.agent import answer_question
        
        # Query last 7 days for context
        start = datetime.now() - timedelta(days=7)
        observations = list(self.query(start_time=start))
        return answer_question(question, observations)
```

### 3. Update Main OG Class

**Modify `og/__init__.py`**:

```python
"""
Own Ghost (OG) - Your activity tracking assistant

DEFAULT MODE: Lightweight (passive observers, no daemons)
  - Uses existing OS data (macOS Screen Time, etc.)
  - Queries data on-demand
  - Minimal resource usage

LEGACY MODE: Full daemon mode (opt-in for power users)
  - Background observers for keyboard, filesystem, etc.
  - See documentation for setup
"""

from og.lightweight import LightweightOG
from og.legacy import FullOG  # Rename current OG class

# Default to lightweight implementation
OG = LightweightOG

# Legacy full-featured version available as:
# from og.legacy import FullOG

__all__ = ['LightweightOG', 'FullOG', 'OG']
```

### 4. Update README.md

Add this section at the top, before existing content:

```markdown
# Own Ghost (OG)

**A lightweight activity assistant that helps you understand how you spend your time**

## 🚀 Quick Start (Recommended: Lightweight Mode)

OG now uses a **lightweight, passive architecture** by default - no background daemons needed!

### Installation

```bash
# Basic installation (lightweight mode)
pip install og

# Or run the installation wizard
curl -sSL https://raw.githubusercontent.com/USER/og/main/install.sh | bash
```

### How It Works (Lightweight Mode)

Instead of running background processes, OG queries existing data sources:

- **macOS**: Reads Screen Time database (already collecting your app usage)
- **All platforms**: Uses `psutil` for on-demand system queries
- **Optional**: Integrates with ActivityWatch for automatic tracking

**No daemons. No background processes. Just query when you need data.**

### Usage

```bash
# Get today's summary
og summary

# Ask questions
og ask "What did I work on this morning?"

# Check available data sources
og sources

# Get statistics
og stats
```

### Python API

```python
from og import OG

# Create lightweight OG instance (default)
og = OG()

# See what data sources are available
print(og.available_observers())

# Get today's activity
print(og.today())

# Query specific time range
from datetime import datetime, timedelta
start = datetime.now() - timedelta(hours=8)
observations = list(og.query(start_time=start))
```

## 📋 Data Sources

OG automatically detects and uses available data sources:

### Built-in (No Installation Required)

- **macOS Screen Time Database**: App usage, web activity (macOS only)
- **Browser History**: Chrome, Firefox, Safari (read-only)
- **Git Repositories**: Commit history (read-only)
- **Shell History**: Bash, zsh, fish commands (read-only)
- **System State**: Running processes, CPU, memory via psutil

### Optional Integrations

#### ActivityWatch (Recommended for cross-platform automatic tracking)

```bash
# Install ActivityWatch
# Download from: https://activitywatch.net/

# Or on macOS:
brew install --cask activitywatch

# Start ActivityWatch
# It runs as a lightweight service and provides API access
```

Benefits:
- Open source, privacy-first
- Lightweight background service
- REST API for data access
- Cross-platform (Windows, macOS, Linux, Android)
- More info: https://github.com/ActivityWatch/activitywatch

#### RescueTime (Commercial alternative)

```bash
# Download from: https://www.rescuetime.com/
# Free tier available with API access
```

## ⚙️ Advanced: Legacy Daemon Mode

For power users who need active monitoring of keyboard, filesystem, etc.:

```python
from og.legacy import FullOG

# Use legacy daemon-based observers
og = FullOG()
og.start()  # Starts background daemons
```

See [LEGACY.md](LEGACY.md) for full documentation on daemon mode.

## 🔒 Privacy

- **Lightweight mode**: Only reads existing local data, no network requests
- **All data stays local** (unless you opt-in to sync features)
- **No external tracking** - you own your data
- **Open source** - audit the code yourself

## 📦 Installation Options

### Automated Installation

```bash
# Run installation wizard (recommended)
curl -sSL https://raw.githubusercontent.com/USER/og/main/install.sh | bash
```

The wizard will:
1. Install OG
2. Detect your OS and available data sources
3. Optionally install ActivityWatch
4. Configure OG for your system
5. Test the installation

### Manual Installation

```bash
# Install OG
pip install og

# Check what data sources are available
og doctor

# Install optional dependencies
pip install og[activitywatch]  # ActivityWatch integration
pip install og[all]             # All optional features
```

### System Requirements

- **Python 3.8+**
- **macOS 10.15+** (for Screen Time integration) or **Linux** or **Windows**
- **Optional**: ActivityWatch for automatic tracking

## 🩺 Troubleshooting

```bash
# Check installation and available data sources
og doctor

# Verify permissions
og check-permissions

# Test data access
og test
```

### Common Issues

**macOS Screen Time not accessible:**
```bash
# Grant Terminal full disk access:
# System Preferences > Security & Privacy > Privacy > Full Disk Access
# Add Terminal or your terminal emulator
```

**No data sources found:**
```bash
# Install ActivityWatch for automatic tracking
brew install --cask activitywatch  # macOS
# or download from: https://activitywatch.net/
```

---

## 📚 Documentation

- [Lightweight Mode Guide](docs/lightweight.md) (recommended)
- [Legacy Daemon Mode](docs/legacy.md) (advanced)
- [API Reference](docs/api.md)
- [Custom Data Sources](docs/custom-sources.md)
```

### 5. Create Installation Script

**`install.sh`** - Automated installation wizard:

```bash
#!/bin/bash
# OG Installation Wizard
# Installs OG with lightweight mode and optional integrations

set -e

echo "================================="
echo "Own Ghost (OG) Installation"
echo "================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect OS
OS="unknown"
case "$(uname -s)" in
    Darwin*)    OS="macos";;
    Linux*)     OS="linux";;
    MINGW*|MSYS*|CYGWIN*) OS="windows";;
esac

echo "Detected OS: $OS"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not found.${NC}"
    echo "Please install Python 3.8+ from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

# Install OG
echo ""
echo "Installing OG..."
pip3 install --upgrade og

echo -e "${GREEN}✓${NC} OG installed"

# Check available data sources
echo ""
echo "Checking available data sources..."

if [ "$OS" = "macos" ]; then
    SCREENTIME_DB="$HOME/Library/Application Support/Knowledge/knowledgeC.db"
    if [ -f "$SCREENTIME_DB" ]; then
        echo -e "${GREEN}✓${NC} macOS Screen Time database found"
        HAS_SCREENTIME=true
    else
        echo -e "${YELLOW}⚠${NC} macOS Screen Time database not found"
        echo "  You may need to grant Full Disk Access to your terminal"
        HAS_SCREENTIME=false
    fi
fi

# Check if ActivityWatch is installed
if command -v activitywatch &> /dev/null || [ -d "/Applications/ActivityWatch.app" ]; then
    echo -e "${GREEN}✓${NC} ActivityWatch found"
    HAS_ACTIVITYWATCH=true
else
    echo -e "${YELLOW}⚠${NC} ActivityWatch not found"
    HAS_ACTIVITYWATCH=false
fi

# Offer to install ActivityWatch
if [ "$HAS_ACTIVITYWATCH" = false ]; then
    echo ""
    echo "ActivityWatch is recommended for automatic activity tracking."
    read -p "Would you like to install ActivityWatch? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ "$OS" = "macos" ]; then
            if command -v brew &> /dev/null; then
                echo "Installing ActivityWatch via Homebrew..."
                brew install --cask activitywatch
                echo -e "${GREEN}✓${NC} ActivityWatch installed"
            else
                echo "Homebrew not found. Please install from: https://activitywatch.net/"
                echo "Or install Homebrew first: https://brew.sh/"
            fi
        elif [ "$OS" = "linux" ]; then
            echo "Please install ActivityWatch manually:"
            echo "  Download from: https://activitywatch.net/"
            echo "  Or check your package manager (AUR, etc.)"
        else
            echo "Please install ActivityWatch from: https://activitywatch.net/"
        fi
    fi
fi

# Check psutil
echo ""
echo "Installing psutil for system monitoring..."
pip3 install psutil
echo -e "${GREEN}✓${NC} psutil installed"

# Test installation
echo ""
echo "Testing installation..."
if python3 -c "from og import OG; og = OG(); print('Available sources:', og.available_observers())" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} OG is working!"
else
    echo -e "${RED}✗${NC} OG test failed"
    echo "Run 'og doctor' for diagnostics"
fi

# Setup instructions
echo ""
echo "================================="
echo "Installation Complete!"
echo "================================="
echo ""
echo "Quick start:"
echo "  og summary          # Get today's summary"
echo "  og ask \"question\"   # Ask about your activity"
echo "  og sources          # List available data sources"
echo "  og doctor           # Check installation"
echo ""

if [ "$OS" = "macos" ] && [ "$HAS_SCREENTIME" = false ]; then
    echo -e "${YELLOW}Note:${NC} To access macOS Screen Time data:"
    echo "  1. Open System Preferences"
    echo "  2. Go to Security & Privacy > Privacy > Full Disk Access"
    echo "  3. Add your terminal application (Terminal, iTerm2, etc.)"
    echo "  4. Restart your terminal"
    echo ""
fi

if [ "$HAS_ACTIVITYWATCH" = true ]; then
    echo "Don't forget to start ActivityWatch for automatic tracking!"
    if [ "$OS" = "macos" ]; then
        echo "  Start ActivityWatch from Applications or Spotlight"
    fi
    echo ""
fi

echo "Documentation: https://github.com/USER/og"
echo ""
```

### 6. Create Diagnostic Tool

**`og/cli.py`** - Add these new commands:

```python
def cmd_doctor():
    """Check OG installation and available data sources."""
    print("OG Installation Diagnostics")
    print("=" * 50)
    print()
    
    # Check Python version
    import sys
    print(f"Python: {sys.version}")
    print()
    
    # Check OG mode
    from og import OG
    og = OG()
    print(f"OG Mode: {'Lightweight' if isinstance(og, LightweightOG) else 'Legacy'}")
    print()
    
    # Check available observers
    print("Available Data Sources:")
    observers = og.available_observers()
    if observers:
        for obs in observers:
            print(f"  ✓ {obs}")
    else:
        print("  ✗ No data sources found")
    print()
    
    # Check permissions
    import platform
    if platform.system() == "Darwin":
        from pathlib import Path
        screentime_db = Path.home() / "Library/Application Support/Knowledge/knowledgeC.db"
        if screentime_db.exists():
            print("✓ macOS Screen Time database accessible")
        else:
            print("✗ macOS Screen Time database not accessible")
            print("  Grant Full Disk Access to your terminal in System Preferences")
    print()
    
    # Check ActivityWatch
    try:
        import requests
        response = requests.get("http://localhost:5600/api/0/buckets", timeout=2)
        if response.status_code == 200:
            print("✓ ActivityWatch is running")
        else:
            print("✗ ActivityWatch not responding")
    except:
        print("✗ ActivityWatch not running")
        print("  Install from: https://activitywatch.net/")
    print()

def cmd_sources():
    """List available data sources."""
    from og import OG
    og = OG()
    
    print("Available Data Sources:")
    print()
    
    for observer_name in og.available_observers():
        print(f"• {observer_name}")
    print()
    
    if not og.available_observers():
        print("No data sources found.")
        print("Run 'og doctor' for diagnostics")

# Add to dispatch
_dispatch_funcs = [
    # ... existing functions ...
    cmd_doctor,
    cmd_sources,
]
```

### 7. Move Legacy Code

- Move current `og/og.py` to `og/legacy.py`
- Rename `OG` class to `FullOG`
- Move all active observers to `og/observers/active/`
- Create `LEGACY.md` documentation
- Update imports throughout codebase

### 8. Update Documentation

Create these new documentation files:

- `docs/lightweight.md` - Lightweight mode guide
- `docs/data-sources.md` - Available data sources
- `docs/activitywatch-integration.md` - ActivityWatch setup
- `LEGACY.md` - Legacy daemon mode documentation

### 9. Update Tests

Create tests for passive observers:

```python
# tests/test_passive_observers.py

def test_macos_screentime_observer():
    """Test macOS Screen Time observer."""
    from og.observers.passive.macos_screentime import MacOSScreenTimeObserver
    
    observer = MacOSScreenTimeObserver()
    # Test is_available()
    # Test query() with mocked database
    
def test_psutil_observer():
    """Test psutil observer."""
    from og.observers.passive.psutil_observer import PsutilObserver
    
    observer = PsutilObserver()
    assert observer.is_available()
    
    # Test query returns current processes
    observations = list(observer.query())
    assert len(observations) > 0
```

## Summary

This refactoring transforms OG from a daemon-heavy system to a lightweight, query-based architecture that:

1. **Uses existing OS data** (macOS Screen Time) instead of redundant monitoring
2. **Queries on-demand** instead of continuous background processes  
3. **Optionally integrates ActivityWatch** for users who want automatic tracking
4. **Keeps legacy mode available** for power users who need active observers
5. **Provides easy installation** with automated wizard
6. **Prioritizes user experience** with better defaults and clear documentation

The new default is lightweight, privacy-focused, and resource-efficient while maintaining all the AI-powered analysis capabilities that make OG unique.