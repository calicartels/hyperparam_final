// Set up Chrome extension background script
chrome.runtime.onInstalled.addListener(() => {
  console.log('HyperExplainer extension has been installed');
  
  // Show tutorial on first install
  chrome.storage.local.get(['tutorialCompleted'], (result) => {
    if (!result.tutorialCompleted) {
      chrome.tabs.create({ url: chrome.runtime.getURL('index.html?tutorial=true') });
      chrome.storage.local.set({ tutorialCompleted: true });
    }
  });
});

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'GET_HYPERPARAMETER_INFO') {
    // Forward the request to our backend API
    const { paramName, paramValue, framework, codeContext } = message.data;
    
    fetch('https://hyperexplainer.replit.app/api/llm/explain-hyperparameter', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        paramName,
        paramValue,
        framework,
        codeContext
      }),
    })
    .then(response => response.json())
    .then(data => {
      sendResponse({ success: true, data });
    })
    .catch(error => {
      console.error('Error fetching hyperparameter info:', error);
      sendResponse({ success: false, error: error.message });
    });
    
    return true; // Keep the message channel open for async response
  }
  
  // Handle requests to show parameter details in the popup
  if (message.type === 'showParameterDetails') {
    // Open the popup with the parameter details
    const { paramName, paramValue, framework, codeContext } = message;
    
    // Open the extension popup with the specific parameter details
    const queryParams = new URLSearchParams({
      param: paramName,
      value: paramValue,
      framework: framework || 'unknown',
      source: 'content_script'
    }).toString();
    
    // Use the chrome runtime API to open the popup page
    const popupUrl = chrome.runtime.getURL(`index.html?${queryParams}`);
    
    // Open in a new tab as a fallback
    chrome.tabs.create({ url: popupUrl });
    
    sendResponse({ success: true });
    return true;
  }
});

// Listen for browser action (toolbar icon) clicks
chrome.action.onClicked.addListener((tab) => {
  // Check if we're on a supported site (like ChatGPT)
  const url = new URL(tab.url);
  const isSupportedSite = 
    url.hostname.includes('openai.com') || 
    url.hostname.includes('chat.openai.com') ||
    url.hostname.includes('github.com') ||
    url.hostname.includes('colab.research.google.com');
  
  if (isSupportedSite) {
    // Toggle the extension visibility on the page
    chrome.tabs.sendMessage(tab.id, { action: 'TOGGLE_EXTENSION' });
  } else {
    // Open options page if not on a supported site
    chrome.runtime.openOptionsPage();
  }
});

// Setup context menu for quick analysis
chrome.contextMenus.create({
  id: 'analyze-hyperparameters',
  title: 'Analyze hyperparameters in code',
  contexts: ['selection']
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'analyze-hyperparameters' && info.selectionText) {
    // Open new tab with analysis
    const encodedText = encodeURIComponent(info.selectionText);
    chrome.tabs.create({
      url: `https://hyperexplainer.replit.app/?code=${encodedText}`
    });
  }
});