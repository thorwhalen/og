# OG Browser Extension

Enhanced browser activity capture for Own Ghost.

## Features

- **Reading Time Tracking**: Measures time spent on each page
- **Content Capture**: Saves page content for later reference
- **Highlight Tracking**: Captures text highlights
- **Scroll Depth**: Tracks how much of each page was read
- **Offline Queue**: Queues observations when OG daemon is offline

## Installation

### Chrome/Edge

1. Open `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select the `browser-extension` directory

### Firefox

1. Open `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Select `manifest.json` from the `browser-extension` directory

## Usage

1. Ensure OG daemon is running (`og start`)
2. The extension will automatically capture browser activity
3. Click the extension icon to view stats and access the dashboard

## Privacy

- All data is sent only to your local OG daemon (localhost:5050)
- No data is sent to external servers
- Content capture can be disabled in settings

## API Integration

The extension sends observations to the OG daemon at:

```
POST http://localhost:5050/api/observation
```

Observation format:

```json
{
  "event_type": "browser_visit_enhanced",
  "url": "https://example.com",
  "title": "Page Title",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

## Development

### File Structure

- `manifest.json` - Extension manifest
- `background.js` - Background service worker
- `content.js` - Content script (runs on all pages)
- `popup.html` - Extension popup UI
- `popup.js` - Popup logic

### Testing

1. Make changes to the extension files
2. Reload the extension in browser
3. Check OG daemon logs for incoming observations
