// Background script for handling API requests
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'API_REQUEST') {
    fetch('http://localhost:3000' + request.endpoint, {
      method: request.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      body: request.body ? JSON.stringify(request.body) : undefined
    })
    .then(response => response.json())
    .then(data => sendResponse({ success: true, data }))
    .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
}); 