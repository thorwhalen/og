/**
 * OG Browser Extension - Popup Script
 */

const OG_DAEMON_URL = 'http://localhost:5050';

// Check OG daemon connection
async function checkConnection() {
  const statusEl = document.getElementById('status');

  try {
    const response = await fetch(`${OG_DAEMON_URL}/api/status`);
    if (response.ok) {
      statusEl.className = 'status connected';
      statusEl.textContent = '✓ Connected to OG';
      loadStats();
    } else {
      throw new Error('Not connected');
    }
  } catch (error) {
    statusEl.className = 'status disconnected';
    statusEl.textContent = '✗ OG daemon not running';
  }
}

// Load stats
async function loadStats() {
  try {
    const response = await fetch(`${OG_DAEMON_URL}/api/stats`);
    const stats = await response.json();

    document.getElementById('pagesCount').textContent = stats.pages_today || 0;
    document.getElementById('readingTime').textContent = `${Math.round((stats.reading_time_seconds || 0) / 60)} min`;
    document.getElementById('highlightsCount').textContent = stats.highlights_today || 0;
  } catch (error) {
    console.error('Failed to load stats:', error);
  }
}

// Open dashboard
document.getElementById('openDashboard').addEventListener('click', () => {
  chrome.tabs.create({ url: `${OG_DAEMON_URL}` });
});

// Export data
document.getElementById('exportData').addEventListener('click', async () => {
  try {
    const response = await fetch(`${OG_DAEMON_URL}/api/export/browser`);
    const blob = await response.blob();

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `og-browser-data-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
  } catch (error) {
    alert('Failed to export data');
  }
});

// Initialize
checkConnection();
