/**
 * OG Browser Extension - Content Script
 *
 * Runs on all pages to capture highlights and reading patterns.
 */

// Track highlights
let highlights = [];

// Listen for text selection
document.addEventListener('mouseup', () => {
  const selection = window.getSelection();
  const selectedText = selection.toString().trim();

  if (selectedText.length > 10) {
    // User selected significant text
    highlights.push({
      text: selectedText,
      timestamp: new Date().toISOString()
    });

    // Send to background
    chrome.runtime.sendMessage({
      action: 'highlight',
      text: selectedText
    });
  }
});

// Listen for messages from background
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getPageContent') {
    // Extract page content
    const content = extractPageContent();

    sendResponse({
      content: content,
      highlights: highlights
    });
  }

  return true;
});

/**
 * Extract main content from page
 */
function extractPageContent() {
  // Try to find main content
  const mainContent =
    document.querySelector('main')?.innerText ||
    document.querySelector('article')?.innerText ||
    document.querySelector('#content')?.innerText ||
    document.querySelector('.content')?.innerText ||
    document.body.innerText;

  // Limit to first 10000 characters
  return mainContent.substring(0, 10000);
}

/**
 * Track scroll depth
 */
let maxScrollDepth = 0;

window.addEventListener('scroll', () => {
  const scrollPercentage = (window.scrollY + window.innerHeight) / document.body.scrollHeight * 100;

  if (scrollPercentage > maxScrollDepth) {
    maxScrollDepth = scrollPercentage;
  }
});

// Send scroll depth on page unload
window.addEventListener('beforeunload', () => {
  if (maxScrollDepth > 0) {
    chrome.runtime.sendMessage({
      action: 'scrollDepth',
      depth: maxScrollDepth,
      url: window.location.href
    });
  }
});
