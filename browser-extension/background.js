/**
 * OG Browser Extension - Background Script
 *
 * Captures browser activity and sends to OG daemon.
 */

const OG_DAEMON_URL = 'http://localhost:5050';

// Track active tab and reading time
let activeTab = null;
let tabStartTime = {};
let readingData = {};

// Listen for tab activation
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  const tab = await chrome.tabs.get(activeInfo.tabId);

  // Record time spent on previous tab
  if (activeTab) {
    recordTabTime(activeTab);
  }

  // Start timing new tab
  activeTab = tab;
  tabStartTime[tab.id] = Date.now();
});

// Listen for tab updates (URL changes)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete') {
    // Page fully loaded
    recordPageVisit(tab);
  }
});

// Listen for window focus changes
chrome.windows.onFocusChanged.addListener((windowId) => {
  if (windowId === chrome.windows.WINDOW_ID_NONE) {
    // Browser lost focus
    if (activeTab) {
      recordTabTime(activeTab);
      activeTab = null;
    }
  }
});

/**
 * Record time spent on a tab
 */
function recordTabTime(tab) {
  if (!tab || !tabStartTime[tab.id]) return;

  const timeSpent = (Date.now() - tabStartTime[tab.id]) / 1000; // seconds

  const data = {
    event_type: 'browser_reading_time',
    url: tab.url,
    title: tab.title,
    time_seconds: timeSpent,
    timestamp: new Date().toISOString()
  };

  sendToOG(data);

  delete tabStartTime[tab.id];
}

/**
 * Record page visit
 */
function recordPageVisit(tab) {
  const data = {
    event_type: 'browser_visit_enhanced',
    url: tab.url,
    title: tab.title,
    timestamp: new Date().toISOString()
  };

  sendToOG(data);

  // Request page content for archiving
  chrome.tabs.sendMessage(tab.id, { action: 'getPageContent' }, (response) => {
    if (response && response.content) {
      sendPageContent(tab.url, response.content, response.highlights);
    }
  });
}

/**
 * Send page content to OG
 */
function sendPageContent(url, content, highlights) {
  const data = {
    event_type: 'browser_content',
    url: url,
    content: content,
    highlights: highlights || [],
    timestamp: new Date().toISOString()
  };

  sendToOG(data);
}

/**
 * Send data to OG daemon
 */
async function sendToOG(data) {
  try {
    await fetch(`${OG_DAEMON_URL}/api/observation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });
  } catch (error) {
    console.error('Failed to send to OG:', error);
    // Queue for later retry
    queueObservation(data);
  }
}

/**
 * Queue observation for retry
 */
function queueObservation(data) {
  chrome.storage.local.get(['pendingObservations'], (result) => {
    const pending = result.pendingObservations || [];
    pending.push(data);

    // Keep only last 100
    if (pending.length > 100) {
      pending.shift();
    }

    chrome.storage.local.set({ pendingObservations: pending });
  });
}

/**
 * Retry pending observations periodically
 */
setInterval(async () => {
  chrome.storage.local.get(['pendingObservations'], async (result) => {
    const pending = result.pendingObservations || [];

    if (pending.length > 0) {
      for (const data of pending) {
        try {
          await sendToOG(data);
          // Remove from queue if successful
          pending.shift();
        } catch (error) {
          // Stop retrying on first failure
          break;
        }
      }

      chrome.storage.local.set({ pendingObservations: pending });
    }
  });
}, 60000); // Every minute

// Listen for messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'highlight') {
    // User highlighted text
    const data = {
      event_type: 'browser_highlight',
      url: sender.tab.url,
      text: request.text,
      timestamp: new Date().toISOString()
    };

    sendToOG(data);
  }

  return true;
});
