// Content script for HyperExplainer extension

// Global state to track if the analysis has been activated
let isAnalysisActive = false;
let hyperparameterSidebar = null;
let highlightedParameters = [];

// Create and inject the HyperExplainer sidebar into the page
function createSidebar() {
  if (hyperparameterSidebar) {
    return hyperparameterSidebar;
  }

  // Create the sidebar container
  hyperparameterSidebar = document.createElement('div');
  hyperparameterSidebar.id = 'hyperexplainer-sidebar';
  hyperparameterSidebar.style.cssText = `
    position: fixed;
    top: 0;
    right: 0;
    width: 350px;
    height: 100vh;
    background-color: white;
    box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1);
    z-index: 10000;
    display: flex;
    flex-direction: column;
    font-family: 'Inter', sans-serif;
    transition: transform 0.3s ease;
    transform: translateX(100%);
  `;

  // Create the header
  const header = document.createElement('div');
  header.style.cssText = `
    padding: 16px;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
  `;
  
  const title = document.createElement('h2');
  title.textContent = 'HyperExplainer';
  title.style.cssText = `
    font-size: 18px;
    font-weight: 600;
    color: #1e293b;
    display: flex;
    align-items: center;
  `;
  
  // Add an eye icon to the title
  title.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="#6366f1" style="width: 20px; height: 20px; margin-right: 8px;">
      <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
      <path fill-rule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clip-rule="evenodd" />
    </svg>
    HyperExplainer
  `;
  
  const closeButton = document.createElement('button');
  closeButton.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 20px; height: 20px;">
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
  `;
  closeButton.style.cssText = `
    background: none;
    border: none;
    cursor: pointer;
    color: #64748b;
  `;
  closeButton.addEventListener('click', () => {
    toggleSidebar(false);
  });
  
  header.appendChild(title);
  header.appendChild(closeButton);
  
  // Create the content area
  const content = document.createElement('div');
  content.id = 'parameter-details';
  content.style.cssText = `
    padding: 16px;
    flex: 1;
    overflow-y: auto;
  `;
  
  // Default state with no selection
  content.innerHTML = `
    <div class="text-sm text-gray-500 mb-3" style="font-size: 14px; color: #64748b; margin-bottom: 12px;">
      Click on a highlighted parameter to see details
    </div>
    <div id="no-selection" style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 32px 0; text-align: center;">
      <div style="width: 64px; height: 64px; border-radius: 50%; background-color: #f1f5f9; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="#94a3b8" style="width: 32px; height: 32px;">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <p style="color: #475569; font-weight: 500; font-size: 16px;">No parameter selected</p>
      <p style="color: #64748b; font-size: 14px; margin-top: 4px;">Select a highlighted parameter in the code to view its details</p>
    </div>
    <div id="parameter-card" style="display: none;"></div>
  `;
  
  // Create the footer
  const footer = document.createElement('div');
  footer.style.cssText = `
    border-top: 1px solid #e2e8f0;
    padding: 16px;
  `;
  
  const generateButton = document.createElement('button');
  generateButton.textContent = 'Generate Alternative Code';
  generateButton.style.cssText = `
    width: 100%;
    background-color: #6366f1;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
  `;
  generateButton.addEventListener('mouseover', () => {
    generateButton.style.backgroundColor = '#4f46e5';
  });
  generateButton.addEventListener('mouseout', () => {
    generateButton.style.backgroundColor = '#6366f1';
  });
  
  const exportButton = document.createElement('button');
  exportButton.textContent = 'Export Parameter Documentation';
  exportButton.style.cssText = `
    width: 100%;
    background-color: white;
    color: #475569;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    margin-top: 8px;
    transition: background-color 0.2s;
  `;
  exportButton.addEventListener('mouseover', () => {
    exportButton.style.backgroundColor = '#f8fafc';
  });
  exportButton.addEventListener('mouseout', () => {
    exportButton.style.backgroundColor = 'white';
  });
  
  footer.appendChild(generateButton);
  footer.appendChild(exportButton);
  
  // Assemble the sidebar
  hyperparameterSidebar.appendChild(header);
  hyperparameterSidebar.appendChild(content);
  hyperparameterSidebar.appendChild(footer);
  
  document.body.appendChild(hyperparameterSidebar);
  
  return hyperparameterSidebar;
}

// Toggle the sidebar visibility
function toggleSidebar(show) {
  if (!hyperparameterSidebar) {
    hyperparameterSidebar = createSidebar();
  }
  
  hyperparameterSidebar.style.transform = show ? 'translateX(0)' : 'translateX(100%)';
}

// Add hyperparameter highlighting to code blocks
function processCodeBlocks() {
  // Only process if analysis is active
  if (!isAnalysisActive) {
    return;
  }
  
  // Find all code blocks in ChatGPT responses
  const codeBlocks = document.querySelectorAll('pre');
  
  codeBlocks.forEach(block => {
    // Skip if already processed
    if (block.dataset.processed === 'true') {
      return;
    }
    
    // Add the processed flag
    block.dataset.processed = 'true';
    
    // Add analyze button if not already present
    if (!block.querySelector('.analyze-button')) {
      const analyzeButton = document.createElement('button');
      analyzeButton.className = 'analyze-button';
      analyzeButton.textContent = 'Analyze';
      analyzeButton.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        background-color: #6366f1;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        cursor: pointer;
        z-index: 10;
      `;
      
      // Make the code block container relative for absolute positioning
      block.style.position = 'relative';
      
      // Add click handler for analyze button
      analyzeButton.addEventListener('click', () => {
        analyzeCodeBlock(block);
        toggleSidebar(true);
      });
      
      block.appendChild(analyzeButton);
    }
  });
}

// Analyze a code block for hyperparameters
function analyzeCodeBlock(codeBlock) {
  const code = codeBlock.textContent;
  
  // Clear previously highlighted parameters
  highlightedParameters = [];
  
  // Basic patterns to identify common hyperparameters
  const patterns = [
    { regex: /lr\s*=\s*([\d.]+)/g, key: "learning_rate" },
    { regex: /learning_rate\s*=\s*([\d.]+)/g, key: "learning_rate" },
    { regex: /batch_size\s*=\s*(\d+)/g, key: "batch_size" },
    { regex: /dropout\s*\(\s*([\d.]+)\s*\)/g, key: "dropout_rate" },
    { regex: /dropout\s*=\s*([\d.]+)/g, key: "dropout_rate" },
    { regex: /epochs\s*=\s*(\d+)/g, key: "num_epochs" },
    { regex: /weight_decay\s*=\s*([\d.]+)/g, key: "weight_decay" }
  ];
  
  // Tokenize the code and replace with highlighted spans
  let tokenizedCode = code;
  let offset = 0;
  
  patterns.forEach(pattern => {
    let match;
    while ((match = pattern.regex.exec(code)) !== null) {
      const fullMatch = match[0];
      const paramValue = match[1];
      const startPos = match.index + code.substring(match.index).indexOf(paramValue);
      const endPos = startPos + paramValue.length;
      
      // Create an object to track this parameter
      const param = {
        key: pattern.key,
        value: paramValue,
        element: null
      };
      
      highlightedParameters.push(param);
    }
  });
  
  // Create highlighted spans for the parameters
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = code;
  
  // Sort parameters by their position in reverse order to avoid messing up indices
  highlightedParameters.sort((a, b) => {
    const aIndex = code.indexOf(a.value);
    const bIndex = code.indexOf(b.value);
    return bIndex - aIndex;
  });
  
  highlightedParameters.forEach(param => {
    // Find the text to replace
    const pos = code.indexOf(param.value);
    if (pos !== -1) {
      const beforeText = tokenizedCode.substring(0, pos);
      const afterText = tokenizedCode.substring(pos + param.value.length);
      
      // Create the highlighted span
      tokenizedCode = beforeText + 
        `<span class="param-highlight" data-param="${param.key}" data-value="${param.value}" style="background-color: #fef08a; border-radius: 4px; padding: 0 4px; cursor: pointer;">` + 
        param.value + 
        '</span>' + 
        afterText;
    }
  });
  
  // Update the code block with highlighted parameters
  codeBlock.innerHTML = tokenizedCode;
  
  // Add click handlers to the highlighted parameters
  const paramHighlights = codeBlock.querySelectorAll('.param-highlight');
  paramHighlights.forEach(highlight => {
    highlight.addEventListener('click', () => {
      const paramKey = highlight.dataset.param;
      const paramValue = highlight.dataset.value;
      showParameterDetails(paramKey, paramValue);
    });
  });
  
  // Show notification
  showNotification(`${highlightedParameters.length} hyperparameters identified!`);
}

// Show parameter details in the sidebar
function showParameterDetails(paramKey, paramValue) {
  if (!hyperparameterSidebar) {
    hyperparameterSidebar = createSidebar();
  }
  
  // Get the parameter card element
  const parameterCard = hyperparameterSidebar.querySelector('#parameter-card');
  const noSelection = hyperparameterSidebar.querySelector('#no-selection');
  
  // Hide the no selection message
  noSelection.style.display = 'none';
  
  // Show the parameter card
  parameterCard.style.display = 'block';
  
  // Hyperparameter database (simplified version for content script)
  const paramInfoMap = {
    learning_rate: {
      name: "Learning Rate",
      value: paramValue,
      description: "Controls how much to adjust the model weights in response to the estimated error each time the model weights are updated.",
      impact: "high",
      framework: "PyTorch",
      alternatives: [
        { value: "0.01", description: "Faster learning, but may overshoot optimal solution", type: "higher" },
        { value: "0.0001", description: "Slower learning, but more stable convergence", type: "lower" },
        { value: "Scheduled", description: "Start high, decrease over time (e.g., ReduceLROnPlateau)", type: "advanced" }
      ]
    },
    batch_size: {
      name: "Batch Size",
      value: paramValue,
      description: "Number of training examples used in one iteration. Affects memory usage and training speed.",
      impact: "medium",
      framework: "PyTorch",
      alternatives: [
        { value: "64", description: "More stable gradients, but higher memory usage", type: "higher" },
        { value: "16", description: "Less memory usage, but potentially less stable", type: "lower" },
        { value: "Power of 2", description: "Values like 32, 64, 128 utilize GPU memory better", type: "advanced" }
      ]
    },
    dropout_rate: {
      name: "Dropout Rate",
      value: paramValue,
      description: "Probability of setting a neuron's output to zero during training, which helps prevent overfitting by making the network more robust.",
      impact: "medium",
      framework: "PyTorch",
      alternatives: [
        { value: "0.5", description: "More aggressive regularization, better for large networks", type: "higher" },
        { value: "0.1", description: "Milder regularization, for smaller networks or less overfitting", type: "lower" },
        { value: "0 (None)", description: "Disable dropout, useful for small datasets or final training", type: "extreme" }
      ]
    },
    num_epochs: {
      name: "Number of Epochs",
      value: paramValue,
      description: "The number of complete passes through the training dataset. Affects how long the model trains and its final performance.",
      impact: "high",
      framework: "PyTorch",
      alternatives: [
        { value: "10", description: "Longer training time, potentially better model", type: "higher" },
        { value: "3", description: "Shorter training, useful for quick experiments", type: "lower" },
        { value: "Early Stopping", description: "Use validation performance to determine when to stop", type: "advanced" }
      ]
    },
    weight_decay: {
      name: "Weight Decay",
      value: paramValue,
      description: "L2 regularization parameter that prevents the weights from growing too large, helping to reduce overfitting.",
      impact: "medium",
      framework: "PyTorch",
      alternatives: [
        { value: "0.01", description: "Stronger regularization effect", type: "higher" },
        { value: "0.0001", description: "Weaker regularization effect", type: "lower" },
        { value: "0", description: "No weight decay/regularization", type: "extreme" }
      ]
    }
  };
  
  const paramInfo = paramInfoMap[paramKey] || {
    name: paramKey.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' '),
    value: paramValue,
    description: `This parameter controls the ${paramKey.replace(/_/g, ' ')} in the neural network.`,
    impact: "medium",
    framework: "Unknown",
    alternatives: [
      { value: "Higher", description: "Increase this value for stronger effect", type: "higher" },
      { value: "Lower", description: "Decrease this value for milder effect", type: "lower" }
    ]
  };
  
  // Create the card HTML
  const impactColor = paramInfo.impact === "high" ? "from-primary-600 to-primary-500" : 
                     paramInfo.impact === "medium" ? "from-secondary-600 to-secondary-500" : 
                     "from-green-600 to-green-500";
  
  parameterCard.innerHTML = `
    <div class="rounded-xl border border-gray-200 overflow-hidden mb-4" style="border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; margin-bottom: 16px;">
      <div style="background: linear-gradient(to right, ${paramInfo.impact === "high" ? "#4f46e5, #6366f1" : "#0d9488, #14b8a6"}); padding: 16px 12px; color: white;">
        <h3 style="font-size: 18px; font-weight: 500;">${paramInfo.name}</h3>
        <p style="font-size: 14px; opacity: 0.8;">Current Value: ${paramInfo.value}</p>
      </div>
      
      <div style="padding: 16px; background: white;">
        <div style="margin-bottom: 16px;">
          <h4 style="font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 4px;">Description</h4>
          <p style="font-size: 14px; color: #475569;">${paramInfo.description}</p>
        </div>
        
        <div style="margin-bottom: 16px;">
          <h4 style="font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 4px;">Impact</h4>
          <div style="display: flex; align-items: center; gap: 4px;">
            ${Array(5).fill().map((_, i) => {
              const isActive = (paramInfo.impact === "high" && i < 3) || 
                              (paramInfo.impact === "medium" && i < 2) ||
                              (paramInfo.impact === "low" && i < 1);
              const bgColor = isActive ? 
                (paramInfo.impact === "high" ? "#ef4444" : 
                paramInfo.impact === "medium" ? "#f59e0b" : "#22c55e") : 
                "#e2e8f0";
              return `<div style="height: 8px; width: 8px; border-radius: 50%; background-color: ${bgColor};"></div>`;
            }).join('')}
            <span style="font-size: 12px; color: #64748b; margin-left: 8px;">
              ${paramInfo.impact === "high" ? "High Impact" : 
                paramInfo.impact === "medium" ? "Medium Impact" : "Low Impact"}
            </span>
          </div>
        </div>

        <div>
          <h4 style="font-size: 14px; font-weight: 600; color: #334155; margin-bottom: 8px;">Alternatives</h4>
          <div style="display: flex; flex-direction: column; gap: 8px;">
            ${paramInfo.alternatives.map(alt => {
              const isFeatured = alt.type === 'advanced' || alt.type === 'extreme';
              const borderColor = isFeatured ? '#c7d2fe' : '#e2e8f0';
              const bgColor = isFeatured ? '#eef2ff' : 'white';
              const hoverBg = isFeatured ? '#e0e7ff' : '#f8fafc';
              
              return `
                <div style="border-radius: 6px; border: 1px solid ${borderColor}; padding: 8px; background-color: ${bgColor}; cursor: pointer; transition: all 0.2s;"
                     onmouseover="this.style.backgroundColor='${hoverBg}';" 
                     onmouseout="this.style.backgroundColor='${bgColor}';">
                  <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                      <span style="font-size: 14px; font-weight: 500; color: #334155;">${alt.value}</span>
                      <p style="font-size: 12px; color: #64748b; margin-top: 4px;">${alt.description}</p>
                    </div>
                    <div style="background-color: ${isFeatured ? '#dbeafe' : '#f1f5f9'}; 
                                color: ${isFeatured ? '#3b82f6' : '#475569'}; 
                                font-size: 12px; 
                                padding: 2px 8px; 
                                border-radius: 4px;">
                      ${alt.type.charAt(0).toUpperCase() + alt.type.slice(1)}
                    </div>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      </div>
      
      <div style="background-color: #f8fafc; padding: 12px 16px; border-top: 1px solid #e2e8f0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div style="font-size: 12px; color: #64748b;">
            <span style="font-weight: 500;">Framework:</span> ${paramInfo.framework}
          </div>
          <button style="font-size: 12px; color: #6366f1; font-weight: 500; display: flex; align-items: center; background: none; border: none; cursor: pointer;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 16px; height: 16px; margin-right: 4px;">
              <circle cx="12" cy="12" r="10" />
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            Learn More
          </button>
        </div>
      </div>
    </div>
  `;
}

// Show a notification
function showNotification(message) {
  const notification = document.createElement('div');
  notification.style.cssText = `
    position: fixed;
    top: 16px;
    right: 16px;
    background-color: #6366f1;
    color: white;
    padding: 8px 16px;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    z-index: 10001;
    font-family: 'Inter', sans-serif;
    display: flex;
    align-items: center;
    animation: fadeIn 0.3s ease;
  `;
  
  notification.innerHTML = `
    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor" style="width: 20px; height: 20px; margin-right: 8px;">
      <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
    </svg>
    <span>${message}</span>
  `;
  
  document.body.appendChild(notification);
  
  // Remove after 3 seconds
  setTimeout(() => {
    notification.style.animation = 'fadeOut 0.3s ease';
    notification.addEventListener('animationend', () => {
      notification.remove();
    });
  }, 3000);
  
  // Add keyframe animations to the document if not already present
  if (!document.querySelector('#hyperexplainer-keyframes')) {
    const style = document.createElement('style');
    style.id = 'hyperexplainer-keyframes';
    style.textContent = `
      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-20px); }
      }
    `;
    document.head.appendChild(style);
  }
}

// Set up a MutationObserver to detect new code blocks added to the page
const observer = new MutationObserver((mutations) => {
  if (isAnalysisActive) {
    processCodeBlocks();
  }
});

// Start observing the document with the configured parameters
observer.observe(document.body, { childList: true, subtree: true });

// Listen for messages from the popup or background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'activateAnalysis') {
    isAnalysisActive = true;
    processCodeBlocks();
    sendResponse({ success: true });
  } else if (message.action === 'deactivateAnalysis') {
    isAnalysisActive = false;
    toggleSidebar(false);
    sendResponse({ success: true });
  }
  return true;
});

// Add the required CSS for proper display
const style = document.createElement('style');
style.textContent = `
  .param-highlight {
    background-color: #fef08a;
    border-radius: 4px;
    padding: 0 4px;
    cursor: pointer;
    transition: background-color 0.2s;
  }
  
  .param-highlight:hover {
    background-color: #fde047;
  }
  
  .analyze-button {
    position: absolute;
    top: 8px;
    right: 8px;
    background-color: #6366f1;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    cursor: pointer;
    z-index: 10;
    font-family: 'Inter', sans-serif;
    transition: background-color 0.2s;
  }
  
  .analyze-button:hover {
    background-color: #4f46e5;
  }
`;
document.head.appendChild(style);

// Load fonts
const fontLink = document.createElement('link');
fontLink.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap';
fontLink.rel = 'stylesheet';
document.head.appendChild(fontLink);

// Initial process of code blocks (in case they're already on the page)
setTimeout(() => {
  // Check chrome.storage for auto-activate setting
  chrome.storage?.sync?.get(['autoActivate'], (result) => {
    if (result.autoActivate !== false) { // Default to true if not set
      isAnalysisActive = true;
      processCodeBlocks();
    }
  });
}, 1000);
