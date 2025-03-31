// Background script for HyperExplainer extension

// Initialize extension when installed
chrome.runtime.onInstalled.addListener(() => {
  console.log('HyperExplainer extension installed');
  
  // Set default settings
  chrome.storage.sync.set({
    autoActivate: true,
    showSidebar: true,
    highlightParams: true,
    showImpactLevel: true
  });
});

// Listen for messages from popup or content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getSettings') {
    chrome.storage.sync.get(null, (settings) => {
      sendResponse({ settings });
    });
    return true; // Required for async response
  }
});

// Listen for tab updates to inject the content script on ChatGPT pages
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url && tab.url.includes('chat.openai.com')) {
    // Check if autoActivate is enabled
    chrome.storage.sync.get(['autoActivate'], (result) => {
      if (result.autoActivate) {
        // Activate analysis on the page
        chrome.tabs.sendMessage(tabId, { action: 'activateAnalysis' }, (response) => {
          // Handle no response (content script might not be loaded yet)
          if (chrome.runtime.lastError) {
            console.log('Content script not ready yet, retrying...');
            
            // Wait a bit and retry
            setTimeout(() => {
              chrome.tabs.sendMessage(tabId, { action: 'activateAnalysis' });
            }, 1000);
          }
        });
      }
    });
  }
});

// Add context menu for quick access to extension features
chrome.contextMenus.create({
  id: "analyze-code",
  title: "Analyze Code for Hyperparameters",
  contexts: ["selection"],
  documentUrlPatterns: ["*://*.openai.com/*"]
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyze-code") {
    chrome.tabs.sendMessage(tab.id, {
      action: 'activateAnalysis',
      selectedText: info.selectionText
    });
  }
});
