"""Web dashboard for OG (Own Ghost).

This module provides an interactive web interface for visualizing
and exploring your activity data.
"""

import json
from datetime import datetime, timedelta
from typing import Optional

from flask import Flask, render_template, jsonify, request

from og.query import OG


class OGDashboard:
    """Interactive web dashboard for OG.

    Provides:
    - Timeline visualization
    - Activity heatmaps
    - Context/project views
    - Alert management
    - Real-time statistics
    """

    def __init__(self, og_instance: OG, host: str = '127.0.0.1', port: int = 5000):
        """Initialize dashboard.

        Args:
            og_instance: OG instance to visualize
            host: Host to bind to
            port: Port to listen on
        """
        self.og = og_instance
        self.host = host
        self.port = port
        self.app = Flask(__name__, template_folder='templates')

        # Register routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes."""

        @self.app.route('/')
        def index():
            """Main dashboard page."""
            return render_template('dashboard.html')

        @self.app.route('/api/status')
        def api_status():
            """Get system status."""
            return jsonify(self.og.status())

        @self.app.route('/api/stats')
        def api_stats():
            """Get statistics."""
            return jsonify(self.og.stats())

        @self.app.route('/api/timeline')
        def api_timeline():
            """Get timeline data."""
            days = request.args.get('days', 7, type=int)
            observations = self.og.recent_activity(days=days)

            # Convert to timeline format
            timeline = [
                {
                    'timestamp': obs.timestamp.isoformat(),
                    'event_type': obs.event_type,
                    'observer': obs.observer_name,
                    'data': obs.data,
                    'tags': obs.tags,
                }
                for obs in observations
            ]

            return jsonify(timeline)

        @self.app.route('/api/heatmap')
        def api_heatmap():
            """Get heatmap data (activity by hour/day)."""
            days = request.args.get('days', 30, type=int)
            observations = self.og.recent_activity(days=days)

            # Create heatmap data structure
            heatmap = {}
            for obs in observations:
                hour = obs.timestamp.hour
                day = obs.timestamp.strftime('%Y-%m-%d')

                if day not in heatmap:
                    heatmap[day] = {h: 0 for h in range(24)}

                heatmap[day][hour] += 1

            return jsonify(heatmap)

        @self.app.route('/api/contexts')
        def api_contexts():
            """Get context information."""
            if hasattr(self.og, 'context_manager'):
                contexts = [
                    {
                        'name': ctx.name,
                        'description': ctx.description,
                        'active': ctx.active,
                        'repos': ctx.repos,
                        'tags': ctx.tags,
                    }
                    for ctx in self.og.context_manager.get_all_contexts()
                ]
                return jsonify(contexts)
            return jsonify([])

        @self.app.route('/api/alerts')
        def api_alerts():
            """Get recent alerts."""
            if hasattr(self.og, 'pattern_detector'):
                acknowledged = request.args.get('acknowledged', type=str)
                ack_filter = (
                    acknowledged.lower() == 'true' if acknowledged else None
                )

                alerts = self.og.pattern_detector.get_alerts(acknowledged=ack_filter)

                alert_data = [
                    {
                        'pattern_name': alert.pattern_name,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity,
                        'acknowledged': alert.acknowledged,
                    }
                    for alert in alerts[-50:]  # Last 50
                ]

                return jsonify(alert_data)
            return jsonify([])

        @self.app.route('/api/summary')
        def api_summary():
            """Get AI summary."""
            days = request.args.get('days', 1, type=int)
            detail = request.args.get('detail', 'medium')

            try:
                summary = self.og.summary(days=days, detail=detail)
                return jsonify({'summary': summary})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/insights')
        def api_insights():
            """Get insights."""
            if hasattr(self.og, 'insight_engine'):
                days = request.args.get('days', 7, type=int)

                productivity = self.og.insight_engine.detect_productivity_patterns(
                    days
                )
                deep_work = self.og.insight_engine.identify_deep_work_sessions(days)
                suggestions = self.og.insight_engine.suggest_optimizations(days)

                return jsonify(
                    {
                        'productivity': productivity,
                        'deep_work': deep_work,
                        'suggestions': suggestions,
                    }
                )
            return jsonify({})

        @self.app.route('/api/search')
        def api_search():
            """Semantic search."""
            query = request.args.get('q', '')
            n_results = request.args.get('n', 10, type=int)

            if hasattr(self.og, 'semantic_memory') and query:
                results = self.og.semantic_memory.search(query, n_results=n_results)

                # Format results
                formatted = []
                if results['documents']:
                    for doc, metadata, distance in zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0],
                    ):
                        formatted.append(
                            {
                                'document': doc,
                                'metadata': metadata,
                                'score': 1 - distance,  # Convert distance to score
                            }
                        )

                return jsonify(formatted)

            return jsonify([])

    def create_templates(self):
        """Create default HTML templates."""
        import os

        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(template_dir, exist_ok=True)

        # Create dashboard.html
        dashboard_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Own Ghost Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-number {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .alert {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
            border-left: 4px solid;
        }
        .alert-info { border-color: #2196F3; background: #E3F2FD; }
        .alert-warning { border-color: #FF9800; background: #FFF3E0; }
        .alert-critical { border-color: #F44336; background: #FFEBEE; }
        .search-box {
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 Own Ghost Dashboard</h1>
        <p class="subtitle">Your personal activity observatory</p>

        <div id="stats" class="stats-grid">
            <div class="stat-card">
                <div class="stat-number" id="total-obs">-</div>
                <div class="stat-label">Total Observations</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="enabled-obs">-</div>
                <div class="stat-label">Active Observers</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="alerts-count">-</div>
                <div class="stat-label">Recent Alerts</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>Activity Heatmap</h2>
            <div id="heatmap"></div>
        </div>

        <div class="chart-container">
            <h2>Recent Alerts</h2>
            <div id="alerts-list"></div>
        </div>

        <div class="chart-container">
            <h2>Semantic Search</h2>
            <input type="text" id="search-box" class="search-box"
                   placeholder="Search your activity...">
            <div id="search-results"></div>
        </div>
    </div>

    <script>
        // Load stats
        fetch('/api/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('total-obs').textContent =
                    data.total_observations || 0;
            });

        fetch('/api/status')
            .then(r => r.json())
            .then(data => {
                document.getElementById('enabled-obs').textContent =
                    data.enabled_observers || 0;
            });

        // Load heatmap
        fetch('/api/heatmap?days=30')
            .then(r => r.json())
            .then(data => {
                const days = Object.keys(data).sort();
                const hours = Array.from({length: 24}, (_, i) => i);

                const z = days.map(day =>
                    hours.map(hour => data[day][hour] || 0)
                );

                const heatmapData = [{
                    z: z,
                    x: hours,
                    y: days,
                    type: 'heatmap',
                    colorscale: 'Viridis'
                }];

                Plotly.newPlot('heatmap', heatmapData, {
                    title: 'Activity by Hour and Day',
                    xaxis: { title: 'Hour of Day' },
                    yaxis: { title: 'Date' }
                });
            });

        // Load alerts
        fetch('/api/alerts')
            .then(r => r.json())
            .then(alerts => {
                document.getElementById('alerts-count').textContent = alerts.length;

                const container = document.getElementById('alerts-list');
                if (alerts.length === 0) {
                    container.innerHTML = '<p>No recent alerts</p>';
                } else {
                    container.innerHTML = alerts.slice(-10).reverse().map(alert => `
                        <div class="alert alert-${alert.severity}">
                            <strong>${alert.pattern_name}</strong>: ${alert.message}
                            <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                        </div>
                    `).join('');
                }
            });

        // Search functionality
        let searchTimeout;
        document.getElementById('search-box').addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                const query = e.target.value;
                if (query.length < 3) return;

                fetch(`/api/search?q=${encodeURIComponent(query)}`)
                    .then(r => r.json())
                    .then(results => {
                        const container = document.getElementById('search-results');
                        if (results.length === 0) {
                            container.innerHTML = '<p>No results found</p>';
                        } else {
                            container.innerHTML = results.map(r => `
                                <div style="padding: 10px; border-bottom: 1px solid #eee;">
                                    <div><strong>${r.metadata.event_type}</strong>
                                         (${r.metadata.observer_name})</div>
                                    <div style="color: #666; font-size: 14px;">${r.document}</div>
                                    <small>Score: ${(r.score * 100).toFixed(1)}%</small>
                                </div>
                            `).join('');
                        }
                    });
            }, 500);
        });

        // Auto-refresh every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
</body>
</html>'''

        with open(os.path.join(template_dir, 'dashboard.html'), 'w') as f:
            f.write(dashboard_html)

    def run(self, debug: bool = False):
        """Run the dashboard server.

        Args:
            debug: Whether to run in debug mode
        """
        # Create templates if they don't exist
        self.create_templates()

        print(f"\n🌐 Starting Own Ghost Dashboard...")
        print(f"📊 Dashboard URL: http://{self.host}:{self.port}")
        print(f"Press Ctrl+C to stop\n")

        self.app.run(host=self.host, port=self.port, debug=debug)


def create_dashboard(og_instance: OG, **kwargs):
    """Convenience function to create and run dashboard.

    Args:
        og_instance: OG instance to visualize
        **kwargs: Additional arguments for OGDashboard

    Example:
        >>> from og import OG
        >>> from og.web import create_dashboard
        >>> og = OG()
        >>> create_dashboard(og, port=5000)  # doctest: +SKIP
    """
    dashboard = OGDashboard(og_instance, **kwargs)
    return dashboard
