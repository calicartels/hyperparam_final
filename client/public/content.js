// Global settings
const EXTENSION_ID = chrome.runtime.id;
const API_ENDPOINT = 'https://hyperexplainer.replit.app';
let sidebarVisible = false;
let detectedParameters = [];
let currentFrameworks = [];

// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
  // Wait a bit for dynamic content to load
  setTimeout(() => {
    processCodeBlocks();
    
    // Add mutation observer to detect new code blocks
    const observer = new MutationObserver((mutations) => {
      let shouldProcess = false;
      
      mutations.forEach((mutation) => {
        if (mutation.addedNodes.length > 0) {
          shouldProcess = true;
        }
      });
      
      if (shouldProcess) {
        setTimeout(processCodeBlocks, 1000);
      }
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }, 1500);
});

// Process all code blocks on the page
function processCodeBlocks() {
  // Look for code blocks in various formats (pre, code elements, etc.)
  const codeElements = document.querySelectorAll('pre code, pre, .code-block, .hljs');
  
  if (codeElements.length === 0) return;
  
  detectedParameters = [];
  currentFrameworks = [];
  
  codeElements.forEach((codeBlock) => {
    if (codeBlock.dataset.hyperprocessed) return;
    
    analyzeCodeBlock(codeBlock);
    codeBlock.dataset.hyperprocessed = 'true';
  });
  
  if (detectedParameters.length > 0 && !sidebarVisible) {
    createSidebar();
    toggleSidebar(true);
  }
}

// Analyze a single code block for hyperparameters
function analyzeCodeBlock(codeBlock) {
  const code = codeBlock.textContent || codeBlock.innerText;
  if (!code) return;
  
  // Detect the framework first
  const frameworks = detectFrameworks(code);
  if (frameworks.length > 0) {
    frameworks.forEach(framework => {
      if (!currentFrameworks.includes(framework)) {
        currentFrameworks.push(framework);
      }
    });
  }
  
  // Define regex patterns for different hyperparameters
  const hyperparameterPatterns = [
    // Learning rate patterns
    { 
      pattern: /learning_rate\s*=\s*([\d.]+)/g, 
      paramName: 'learning_rate',
      framework: 'tensorflow'
    },
    { 
      pattern: /lr\s*=\s*([\d.]+)/g,
      paramName: 'lr',
      framework: 'pytorch'
    },
    
    // Batch size patterns
    { 
      pattern: /batch_size\s*=\s*(\d+)/g,
      paramName: 'batch_size',
      framework: 'common'
    },
    
    // Dropout patterns
    { 
      pattern: /dropout\s*=\s*([\d.]+)/g,
      paramName: 'dropout',
      framework: 'common'
    },
    { 
      pattern: /Dropout\s*\(\s*([\d.]+)\s*\)/g,
      paramName: 'dropout',
      framework: 'common'
    },
    
    // Epochs patterns
    { 
      pattern: /epochs\s*=\s*(\d+)/g,
      paramName: 'epochs',
      framework: 'common'
    },
    
    // Weight decay/regularization patterns
    { 
      pattern: /weight_decay\s*=\s*([\d.]+)/g,
      paramName: 'weight_decay',
      framework: 'pytorch'
    },
    
    // Optimizer patterns
    { 
      pattern: /optimizer\s*=\s*['"]?(\w+)['"]?/g,
      paramName: 'optimizer',
      framework: 'common'
    },
    { 
      pattern: /Optimizer\.\w+\s*\(/g,
      paramName: 'optimizer',
      framework: 'tensorflow'
    },
    
    // Activation function patterns
    { 
      pattern: /activation\s*=\s*['"](\w+)['"]/g,
      paramName: 'activation',
      framework: 'common'
    },
    
    // Momentum patterns
    { 
      pattern: /momentum\s*=\s*([\d.]+)/g,
      paramName: 'momentum',
      framework: 'common'
    },
    
    // Hidden layers/units patterns
    { 
      pattern: /units\s*=\s*(\d+)/g,
      paramName: 'units',
      framework: 'tensorflow'
    },
    { 
      pattern: /hidden_size\s*=\s*(\d+)/g,
      paramName: 'hidden_size',
      framework: 'pytorch'
    }
  ];
  
  // Process the code with each pattern
  hyperparameterPatterns.forEach(({ pattern, paramName, framework }) => {
    let match;
    while ((match = pattern.exec(code)) !== null) {
      const paramValue = match[1];
      const matchedText = match[0];
      
      // Find the actual location in the DOM
      const textNode = findTextNodeContaining(codeBlock, matchedText);
      if (!textNode) return;
      
      // Create a unique ID for this parameter
      const paramId = `hyperparam-${paramName}-${Math.floor(Math.random() * 10000)}`;
      
      // Split the text node to insert our highlighted element
      const range = document.createRange();
      range.setStart(textNode.node, textNode.start + match.index);
      range.setEnd(textNode.node, textNode.start + match.index + matchedText.length);
      
      // Create the highlight span
      const span = document.createElement('span');
      span.id = paramId;
      span.className = 'hyperexplainer-highlight';
      span.setAttribute('data-param-name', paramName);
      span.setAttribute('data-param-value', paramValue);
      span.setAttribute('data-framework', framework);
      span.innerHTML = matchedText;
      span.style.backgroundColor = 'rgba(255, 212, 0, 0.3)';
      span.style.padding = '2px';
      span.style.borderRadius = '3px';
      span.style.cursor = 'pointer';
      span.style.position = 'relative';
      
      // Replace the text with our span
      range.deleteContents();
      range.insertNode(span);
      
      // Add click handler to show details
      span.addEventListener('click', () => {
        showParameterDetails(paramName, paramValue, framework, code);
      });
      
      // Add to our list of detected parameters
      const existingParam = detectedParameters.find(p => 
        p.paramName === paramName && p.paramValue === paramValue
      );
      
      if (!existingParam) {
        detectedParameters.push({
          id: paramId,
          paramName,
          paramValue,
          framework,
          element: span,
          codeContext: code
        });
      }
    }
  });
  
  // If we found parameters, add a floating icon to access the sidebar
  if (detectedParameters.length > 0 && !document.querySelector('.hyperexplainer-fab')) {
    const fab = document.createElement('div');
    fab.className = 'hyperexplainer-fab';
    fab.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20.24 12.24a6 6 0 0 0-8.49-8.49L5 10.5V19h8.5z"></path>
        <line x1="16" y1="8" x2="2" y2="22"></line>
        <line x1="17.5" y1="15" x2="9" y2="15"></line>
      </svg>
    `;
    fab.style.position = 'fixed';
    fab.style.right = '20px';
    fab.style.bottom = '20px';
    fab.style.width = '48px';
    fab.style.height = '48px';
    fab.style.borderRadius = '50%';
    fab.style.backgroundColor = '#4F46E5';
    fab.style.color = 'white';
    fab.style.display = 'flex';
    fab.style.alignItems = 'center';
    fab.style.justifyContent = 'center';
    fab.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
    fab.style.cursor = 'pointer';
    fab.style.zIndex = '10000';
    
    fab.addEventListener('click', () => {
      toggleSidebar(!sidebarVisible);
    });
    
    document.body.appendChild(fab);
  }
}

// Helper function to find text node containing a specific string
function findTextNodeContaining(element, text) {
  const textNodes = [];
  
  function getTextNodes(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      textNodes.push({
        node,
        text: node.textContent,
        start: 0
      });
    } else {
      for (let i = 0; i < node.childNodes.length; i++) {
        getTextNodes(node.childNodes[i]);
      }
    }
  }
  
  getTextNodes(element);
  
  for (const textNode of textNodes) {
    const index = textNode.text.indexOf(text);
    if (index !== -1) {
      return {
        node: textNode.node,
        start: index
      };
    }
  }
  
  return null;
}

// Create the sidebar to display all detected hyperparameters
function createSidebar() {
  // Remove existing sidebar if any
  const existingSidebar = document.getElementById('hyperexplainer-sidebar');
  if (existingSidebar) {
    existingSidebar.remove();
  }
  
  // Create new sidebar
  const sidebar = document.createElement('div');
  sidebar.id = 'hyperexplainer-sidebar';
  sidebar.style.position = 'fixed';
  sidebar.style.top = '0';
  sidebar.style.right = '-350px';
  sidebar.style.width = '350px';
  sidebar.style.height = '100vh';
  sidebar.style.backgroundColor = 'white';
  sidebar.style.boxShadow = '-2px 0 8px rgba(0, 0, 0, 0.15)';
  sidebar.style.transition = 'right 0.3s ease-in-out';
  sidebar.style.zIndex = '10001';
  sidebar.style.overflow = 'auto';
  sidebar.style.padding = '20px';
  sidebar.style.fontFamily = 'system-ui, -apple-system, sans-serif';
  
  // Add header
  sidebar.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
      <h2 style="margin: 0; color: #4F46E5; font-size: 18px;">HyperExplainer</h2>
      <button id="hyperexplainer-close" style="background: none; border: none; cursor: pointer; font-size: 20px;">×</button>
    </div>
    <p style="margin-bottom: 15px; font-size: 14px; color: #666;">
      Detected ${detectedParameters.length} hyperparameters in the code.
      Click on each to learn more.
    </p>
    <div id="hyperexplainer-parameters-list"></div>
  `;
  
  document.body.appendChild(sidebar);
  
  // Add parameters to list
  const parametersList = document.getElementById('hyperexplainer-parameters-list');
  
  detectedParameters.forEach((param) => {
    const paramItem = document.createElement('div');
    paramItem.className = 'hyperexplainer-param-item';
    paramItem.style.padding = '12px';
    paramItem.style.marginBottom = '10px';
    paramItem.style.border = '1px solid #eee';
    paramItem.style.borderRadius = '6px';
    paramItem.style.cursor = 'pointer';
    paramItem.style.transition = 'background-color 0.2s';
    
    // Determine importance indicator
    let importance = 'medium';
    if (['learning_rate', 'lr', 'batch_size', 'optimizer'].includes(param.paramName)) {
      importance = 'high';
    } else if (['weight_decay', 'momentum'].includes(param.paramName)) {
      importance = 'medium';
    }
    
    const importanceColor = {
      high: '#ef4444',
      medium: '#f59e0b',
      low: '#10b981'
    };
    
    paramItem.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <div style="font-weight: 600; font-size: 15px; color: #333;">
            ${param.paramName}
            <span style="font-weight: normal; color: #666;">=</span>
            <span style="color: #4F46E5;">${param.paramValue}</span>
          </div>
          <div style="font-size: 13px; color: #666; margin-top: 3px;">
            Framework: ${param.framework}
          </div>
        </div>
        <div style="display: flex; align-items: center;">
          <span style="
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: ${importanceColor[importance]};
            margin-right: 6px;
          "></span>
          <span style="font-size: 12px; color: #666; text-transform: capitalize;">${importance} impact</span>
        </div>
      </div>
      <div style="font-size: 13px; margin-top: 8px; display: flex; justify-content: flex-end;">
        <button class="hyperexplainer-learn-more" style="
          background: none;
          border: none;
          color: #4F46E5;
          font-size: 13px;
          cursor: pointer;
          text-decoration: underline;
          padding: 0;
        ">Learn more</button>
      </div>
    `;
    
    paramItem.addEventListener('mouseover', () => {
      paramItem.style.backgroundColor = '#f9fafb';
    });
    
    paramItem.addEventListener('mouseout', () => {
      paramItem.style.backgroundColor = 'white';
    });
    
    paramItem.querySelector('.hyperexplainer-learn-more').addEventListener('click', (e) => {
      e.stopPropagation();
      // Open details page in new tab
      window.open(`${API_ENDPOINT}/?param=${param.paramName}&value=${param.paramValue}&framework=${param.framework}`, '_blank');
    });
    
    paramItem.addEventListener('click', () => {
      showParameterDetails(param.paramName, param.paramValue, param.framework, param.codeContext);
    });
    
    parametersList.appendChild(paramItem);
  });
  
  // Add close button event
  document.getElementById('hyperexplainer-close').addEventListener('click', () => {
    toggleSidebar(false);
  });
  
  return sidebar;
}

// Toggle sidebar visibility
function toggleSidebar(show) {
  const sidebar = document.getElementById('hyperexplainer-sidebar');
  if (!sidebar) return;
  
  if (show) {
    sidebar.style.right = '0';
    sidebarVisible = true;
  } else {
    sidebar.style.right = '-350px';
    sidebarVisible = false;
  }
}

// Show compact parameter details popup
function showParameterDetails(paramKey, paramValue, framework, codeContext) {
  // Remove any existing popup
  const existingPopup = document.getElementById('hyperexplainer-popup');
  if (existingPopup) {
    existingPopup.remove();
  }
  
  // Create new popup with minimal content
  const popup = document.createElement('div');
  popup.id = 'hyperexplainer-popup';
  popup.style.position = 'fixed';
  popup.style.top = '50%';
  popup.style.left = '50%';
  popup.style.transform = 'translate(-50%, -50%)';
  popup.style.width = '350px';
  popup.style.backgroundColor = 'white';
  popup.style.borderRadius = '8px';
  popup.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.15)';
  popup.style.zIndex = '10002';
  popup.style.padding = '20px';
  popup.style.fontFamily = 'system-ui, -apple-system, sans-serif';
  
  // Add popup content
  popup.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
      <h3 style="margin: 0; color: #4F46E5; font-size: 18px;">${paramKey}</h3>
      <button id="hyperexplainer-popup-close" style="background: none; border: none; cursor: pointer; font-size: 20px;">×</button>
    </div>
    <div style="margin-bottom: 15px;">
      <div style="font-size: 15px; margin-bottom: 10px;">
        <strong>Current value:</strong> <span style="color: #4F46E5; font-weight: 500;">${paramValue}</span>
      </div>
      <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
        <strong>Impact:</strong> This parameter has a <span style="color: #ef4444; font-weight: 500;">high impact</span> on model performance.
      </div>
      <div style="font-size: 14px; color: #666; margin-bottom: 15px;">
        <strong>Usage in:</strong> ${framework}
      </div>
    </div>
    <div style="display: flex; justify-content: center;">
      <button id="hyperexplainer-learn-more-btn" style="
        background-color: #4F46E5;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 14px;
        cursor: pointer;
        transition: background-color 0.2s;
      ">See detailed analysis</button>
    </div>
  `;
  
  document.body.appendChild(popup);
  
  // Add backdrop
  const backdrop = document.createElement('div');
  backdrop.id = 'hyperexplainer-backdrop';
  backdrop.style.position = 'fixed';
  backdrop.style.top = '0';
  backdrop.style.left = '0';
  backdrop.style.width = '100%';
  backdrop.style.height = '100%';
  backdrop.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
  backdrop.style.zIndex = '10001';
  
  document.body.appendChild(backdrop);
  
  // Add event listeners
  document.getElementById('hyperexplainer-popup-close').addEventListener('click', () => {
    popup.remove();
    backdrop.remove();
  });
  
  document.getElementById('hyperexplainer-learn-more-btn').addEventListener('click', () => {
    // Open details page in new tab with appropriate query parameters
    window.open(`${API_ENDPOINT}/?param=${paramKey}&value=${paramValue}&framework=${framework}`, '_blank');
    
    // Close popup
    popup.remove();
    backdrop.remove();
  });
  
  backdrop.addEventListener('click', () => {
    popup.remove();
    backdrop.remove();
  });
}

// Show notification
function showNotification(message) {
  const notification = document.createElement('div');
  notification.className = 'hyperexplainer-notification';
  notification.style.position = 'fixed';
  notification.style.bottom = '20px';
  notification.style.left = '20px';
  notification.style.padding = '12px 16px';
  notification.style.backgroundColor = '#4F46E5';
  notification.style.color = 'white';
  notification.style.borderRadius = '6px';
  notification.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
  notification.style.zIndex = '10003';
  notification.style.transition = 'opacity 0.3s, transform 0.3s';
  notification.style.opacity = '0';
  notification.style.transform = 'translateY(20px)';
  notification.style.fontFamily = 'system-ui, -apple-system, sans-serif';
  notification.style.fontSize = '14px';
  
  notification.innerHTML = message;
  
  document.body.appendChild(notification);
  
  // Show with animation
  setTimeout(() => {
    notification.style.opacity = '1';
    notification.style.transform = 'translateY(0)';
  }, 10);
  
  // Auto-hide after 4 seconds
  setTimeout(() => {
    notification.style.opacity = '0';
    notification.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
      notification.remove();
    }, 300);
  }, 4000);
}

// Detect ML framework from code
function detectFrameworks(code) {
  const frameworks = [];
  
  // TensorFlow patterns
  if (code.includes('tf.') || code.includes('tensorflow') || 
      code.includes('keras.') || code.includes('Sequential()')) {
    frameworks.push('tensorflow');
  }
  
  // PyTorch patterns
  if (code.includes('torch.') || code.includes('nn.Module') || 
      code.includes('optim.') || code.includes('from torch import')) {
    frameworks.push('pytorch');
  }
  
  // Scikit-learn patterns
  if (code.includes('sklearn.') || code.includes('from sklearn import')) {
    frameworks.push('scikit-learn');
  }
  
  // XGBoost patterns
  if (code.includes('xgboost') || code.includes('XGBClassifier') || 
      code.includes('XGBRegressor')) {
    frameworks.push('xgboost');
  }
  
  // Hugging Face patterns
  if (code.includes('transformers') || code.includes('from transformers import')) {
    frameworks.push('huggingface');
  }
  
  return frameworks.length > 0 ? frameworks : ['unknown'];
}