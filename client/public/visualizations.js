// Load the Chart.js library dynamically
function loadChartJS() {
  return new Promise((resolve, reject) => {
    if (window.Chart) {
      resolve(window.Chart);
      return;
    }
    
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js';
    script.onload = () => resolve(window.Chart);
    script.onerror = (error) => reject(error);
    document.head.appendChild(script);
  });
}

// Create hyperparameter visualization in a container
async function createHyperparameterVisualization(paramKey, paramValue, container) {
  try {
    await loadChartJS();
    
    // Handle different parameter types
    if (paramKey.includes('learning_rate') || paramKey.includes('lr')) {
      generateLearningRateCurve(container, paramValue);
    } else if (paramKey.includes('batch_size')) {
      generateBatchSizeVisualization(container, paramValue);
    } else if (paramKey.includes('dropout')) {
      generateDropoutVisualization(container, paramValue);
    } else if (paramKey.includes('optimizer')) {
      generateOptimizerComparison(container, paramValue);
    } else if (paramKey.includes('epoch')) {
      generateEpochsVisualization(container, paramValue);
    } else if (paramKey.includes('momentum')) {
      generateMomentumVisualization(container, paramValue);
    } else if (paramKey.includes('weight_decay') || paramKey.includes('regularization')) {
      generateWeightDecayVisualization(container, paramValue);
    } else {
      // Generic visualization
      generateGenericVisualization(container, paramKey, paramValue);
    }
  } catch (error) {
    console.error('Error creating visualization:', error);
    container.innerHTML = `
      <div style="text-align: center; padding: 20px; color: #666;">
        <p>Unable to load visualization. Please click "Learn more" for detailed information.</p>
      </div>
    `;
  }
}

// Learning rate visualization
function generateLearningRateCurve(container, learningRate) {
  const lr = parseFloat(learningRate);
  
  // Canvas setup
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  
  // Convert learning rate to position in log scale
  const lrValues = [0.0001, 0.001, 0.01, 0.1, 1.0];
  const lrLabels = lrValues.map(v => v.toString());
  
  // Generate accuracy curves (bell curve with peak at optimal learning rate)
  const trainAccData = [];
  const valAccData = [];
  
  const optimalLR = 0.01; // For example purposes
  const currentLRIndex = lrValues.indexOf(lr) !== -1 
    ? lrValues.indexOf(lr) 
    : findClosestIndex(lrValues, lr);
  
  lrValues.forEach((value, i) => {
    // Distance from optimal LR in log space
    const distance = Math.abs(Math.log10(value) - Math.log10(optimalLR));
    const trainingAcc = 0.9 - 0.3 * (distance ** 2);
    const valAcc = 0.85 - 0.3 * (distance ** 2);
    
    trainAccData.push(trainingAcc);
    valAccData.push(valAcc);
  });
  
  // Create chart
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: lrLabels,
      datasets: [
        {
          label: 'Training Accuracy',
          data: trainAccData,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Validation Accuracy',
          data: valAccData,
          borderColor: 'rgba(153, 102, 255, 1)',
          backgroundColor: 'rgba(153, 102, 255, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Learning Rate Impact'
        },
        tooltip: {
          callbacks: {
            title: function(context) {
              return `Learning Rate: ${context[0].label}`;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Learning Rate (log scale)'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Accuracy'
          },
          min: 0,
          max: 1
        }
      }
    }
  });
  
  // Add annotation for current value
  addCurrentValueMarker(container, lr, currentLRIndex, lrValues);
}

// Batch size visualization
function generateBatchSizeVisualization(container, batchSize) {
  const bs = parseInt(batchSize);
  
  // Canvas setup
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  
  // Batch size values
  const bsValues = [8, 16, 32, 64, 128, 256];
  const bsLabels = bsValues.map(v => v.toString());
  
  // Find closest batch size in our scale
  const currentBSIndex = bsValues.indexOf(bs) !== -1 
    ? bsValues.indexOf(bs) 
    : findClosestIndex(bsValues, bs);
  
  // Generate training time and memory usage data
  const trainingTimeData = bsValues.map(value => 1.0 - 0.7 * (Math.log(value) / Math.log(256)));
  const memoryUsageData = bsValues.map(value => 0.2 + 0.7 * (Math.log(value) / Math.log(256)));
  const convergenceData = bsValues.map(value => {
    // Small batches can have noise, very large batches may not generalize as well
    const distanceFromOptimal = Math.abs(Math.log(value) - Math.log(64)) / Math.log(256);
    return 0.9 - 0.4 * distanceFromOptimal;
  });
  
  // Create chart
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: bsLabels,
      datasets: [
        {
          label: 'Training Speed',
          data: trainingTimeData.map(v => 1-v), // Invert for intuitive display
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.2,
          pointRadius: 4,
          pointHoverRadius: 6,
          yAxisID: 'y'
        },
        {
          label: 'Memory Usage',
          data: memoryUsageData,
          borderColor: 'rgba(54, 162, 235, 1)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          tension: 0.2,
          pointRadius: 4,
          pointHoverRadius: 6,
          yAxisID: 'y'
        },
        {
          label: 'Convergence Quality',
          data: convergenceData,
          borderColor: 'rgba(255, 206, 86, 1)',
          backgroundColor: 'rgba(255, 206, 86, 0.2)',
          tension: 0.2,
          pointRadius: 4,
          pointHoverRadius: 6,
          yAxisID: 'y'
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Batch Size Impact'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Batch Size'
          }
        },
        y: {
          type: 'linear',
          display: true,
          position: 'left',
          title: {
            display: true,
            text: 'Relative Value'
          },
          min: 0,
          max: 1
        }
      }
    }
  });
  
  // Add annotation for current value
  addCurrentValueMarker(container, bs, currentBSIndex, bsValues);
}

// Dropout visualization
function generateDropoutVisualization(container, dropoutRate) {
  const rate = parseFloat(dropoutRate);
  
  // Canvas setup
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  
  // Dropout values
  const dropoutValues = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
  const dropoutLabels = dropoutValues.map(v => v.toString());
  
  // Find the closest dropout value in our scale
  const currentDropoutIndex = dropoutValues.indexOf(rate) !== -1 
    ? dropoutValues.indexOf(rate) 
    : findClosestIndex(dropoutValues, rate);
  
  // Generate training and validation accuracy based on dropout rate
  const trainingAccData = dropoutValues.map(value => {
    // Training accuracy decreases with higher dropout
    return 0.99 - value * 0.4;
  });
  
  const valAccData = dropoutValues.map(value => {
    // Validation accuracy typically follows an inverted U shape
    // with optimal values around 0.2-0.3 for many tasks
    const distanceFromOptimal = Math.abs(value - 0.3);
    return 0.9 - 0.3 * (distanceFromOptimal ** 1.5);
  });
  
  // Create chart
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: dropoutLabels,
      datasets: [
        {
          label: 'Training Accuracy',
          data: trainingAccData,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Validation Accuracy',
          data: valAccData,
          borderColor: 'rgba(153, 102, 255, 1)',
          backgroundColor: 'rgba(153, 102, 255, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Dropout Rate Impact'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Dropout Rate'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Accuracy'
          },
          min: 0.5,
          max: 1
        }
      }
    }
  });
  
  // Add annotation for current value
  addCurrentValueMarker(container, rate, currentDropoutIndex, dropoutValues);
}

// Optimizer comparison
function generateOptimizerComparison(container, optimizerName) {
  // Canvas setup
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  
  // Optimizer values
  const optimizers = ['SGD', 'Adam', 'RMSprop', 'AdaGrad', 'AdamW'];
  
  // Find the current optimizer index
  const currentOptimizerIndex = optimizers.findIndex(
    opt => opt.toLowerCase() === optimizerName.toLowerCase()
  );
  
  // Generate performance metrics
  const convergenceSpeedData = [0.5, 0.9, 0.7, 0.6, 0.85];
  const finalAccuracyData = [0.88, 0.92, 0.90, 0.87, 0.93];
  const memoryUsageData = [0.4, 0.7, 0.65, 0.55, 0.75];
  
  // Create chart
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: optimizers,
      datasets: [
        {
          label: 'Convergence Speed',
          data: convergenceSpeedData,
          backgroundColor: 'rgba(54, 162, 235, 0.7)',
          borderColor: 'rgba(54, 162, 235, 1)',
          borderWidth: 1
        },
        {
          label: 'Final Accuracy',
          data: finalAccuracyData,
          backgroundColor: 'rgba(75, 192, 192, 0.7)',
          borderColor: 'rgba(75, 192, 192, 1)',
          borderWidth: 1
        },
        {
          label: 'Memory Usage',
          data: memoryUsageData,
          backgroundColor: 'rgba(255, 99, 132, 0.7)',
          borderColor: 'rgba(255, 99, 132, 1)',
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Optimizer Comparison'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Optimizer'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Relative Performance'
          },
          min: 0,
          max: 1
        }
      }
    }
  });
  
  // Add annotation for current optimizer
  if (currentOptimizerIndex >= 0) {
    const highlight = document.createElement('div');
    highlight.style.textAlign = 'center';
    highlight.style.marginTop = '10px';
    highlight.style.padding = '8px';
    highlight.style.backgroundColor = 'rgba(75, 192, 192, 0.1)';
    highlight.style.border = '1px solid rgba(75, 192, 192, 0.5)';
    highlight.style.borderRadius = '4px';
    highlight.style.fontSize = '14px';
    
    highlight.innerHTML = `
      <span style="font-weight: 600;">Current optimizer:</span>
      <span style="color: #4F46E5; font-weight: 500; margin-left: 5px;">${optimizerName}</span>
    `;
    
    container.appendChild(highlight);
  }
}

// Generic parameter visualization
function generateGenericVisualization(container, paramKey, paramValue) {
  const numValue = parseFloat(paramValue) || 0.5;
  
  // Canvas setup
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  
  // Parameter range (centered around current value)
  const min = Math.max(0, numValue * 0.5);
  const max = numValue * 1.5;
  const step = (max - min) / 4;
  
  const paramValues = [];
  for (let i = 0; i <= 4; i++) {
    paramValues.push(min + i * step);
  }
  const paramLabels = paramValues.map(v => v.toFixed(4).toString());
  
  // Find current value index
  const currentValueIndex = paramValues.indexOf(numValue) !== -1 
    ? paramValues.indexOf(numValue) 
    : findClosestIndex(paramValues, numValue);
  
  // Generate performance and cost curves
  const performanceData = paramValues.map((val) => {
    const distance = Math.abs(val - numValue);
    return 0.9 - 0.3 * (distance / (max - min)) ** 2;
  });
  
  const costData = paramValues.map((val) => {
    const normalizedVal = (val - min) / (max - min);
    return 0.3 + 0.5 * normalizedVal;
  });
  
  // Create chart
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: paramLabels,
      datasets: [
        {
          label: 'Model Performance',
          data: performanceData,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Computational Cost',
          data: costData,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: `${paramKey} Impact`
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: `${paramKey} Value`
          }
        },
        y: {
          title: {
            display: true,
            text: 'Relative Value'
          },
          min: 0,
          max: 1
        }
      }
    }
  });
  
  // Add annotation for current value
  addCurrentValueMarker(container, numValue, currentValueIndex, paramValues);
}

// Helper to generate epochs visualization
function generateEpochsVisualization(container, epochs) {
  const numEpochs = parseInt(epochs);
  
  // Canvas setup
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  
  // Epoch values
  const maxEpochs = Math.max(100, numEpochs * 1.5);
  const epochValues = [1, Math.ceil(maxEpochs * 0.25), Math.ceil(maxEpochs * 0.5), 
                       Math.ceil(maxEpochs * 0.75), maxEpochs];
  const epochLabels = epochValues.map(v => v.toString());
  
  // Find closest epoch in our scale
  const currentEpochIndex = findClosestIndex(epochValues, numEpochs);
  
  // Generate training and validation curves
  const trainAccData = epochValues.map(value => {
    // Training accuracy increases with epochs but plateaus
    return 0.5 + 0.49 * (1 - Math.exp(-value / (maxEpochs * 0.3)));
  });
  
  const valAccData = epochValues.map(value => {
    // Validation accuracy increases and then slightly decreases (overfitting)
    const normalizedEpoch = value / maxEpochs;
    if (normalizedEpoch < 0.6) {
      return 0.5 + 0.4 * (1 - Math.exp(-value / (maxEpochs * 0.3)));
    } else {
      // Slight decline due to overfitting
      return 0.9 - 0.1 * ((normalizedEpoch - 0.6) / 0.4);
    }
  });
  
  // Create chart
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochLabels,
      datasets: [
        {
          label: 'Training Accuracy',
          data: trainAccData,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Validation Accuracy',
          data: valAccData,
          borderColor: 'rgba(153, 102, 255, 1)',
          backgroundColor: 'rgba(153, 102, 255, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Training Epochs Impact'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Epochs'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Accuracy'
          },
          min: 0.4,
          max: 1
        }
      }
    }
  });
  
  // Add annotation for current value
  addCurrentValueMarker(container, numEpochs, currentEpochIndex, epochValues);
}

// Helper to generate momentum visualization
function generateMomentumVisualization(container, momentum) {
  const momentumValue = parseFloat(momentum);
  
  // Canvas setup
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  
  // Momentum values
  const momentumValues = [0, 0.5, 0.9, 0.95, 0.99];
  const momentumLabels = momentumValues.map(v => v.toString());
  
  // Find closest momentum in our scale
  const currentMomentumIndex = findClosestIndex(momentumValues, momentumValue);
  
  // Generate convergence data
  const convergenceSpeedData = momentumValues.map(value => {
    // Convergence typically improves with momentum but can diverge at very high values
    if (value < 0.9) {
      return 0.4 + 0.5 * value;
    } else {
      return 0.9 - (value - 0.9) * 2; // Sharp decline after 0.9
    }
  });
  
  const stabilityData = momentumValues.map(value => {
    // Stability decreases with very high momentum
    return 0.9 - value * 0.4;
  });
  
  // Create chart
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: momentumLabels,
      datasets: [
        {
          label: 'Convergence Speed',
          data: convergenceSpeedData,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Training Stability',
          data: stabilityData,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Momentum Impact'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Momentum Value'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Relative Performance'
          },
          min: 0,
          max: 1
        }
      }
    }
  });
  
  // Add annotation for current value
  addCurrentValueMarker(container, momentumValue, currentMomentumIndex, momentumValues);
}

// Helper to generate weight decay visualization
function generateWeightDecayVisualization(container, weightDecay) {
  const decayValue = parseFloat(weightDecay);
  
  // Canvas setup
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  
  // Weight decay values (log scale)
  const decayValues = [0, 0.0001, 0.001, 0.01, 0.1];
  const decayLabels = decayValues.map(v => v.toString());
  
  // Find closest decay value in our scale
  const currentDecayIndex = findClosestIndex(decayValues, decayValue);
  
  // Generate training and validation curves
  const trainAccData = decayValues.map(value => {
    // Training accuracy decreases with higher regularization
    return 0.95 - 0.15 * (value ** 0.5);
  });
  
  const valAccData = decayValues.map(value => {
    // Validation accuracy typically follows an inverted U shape
    if (value === 0) return 0.8; // No regularization
    const logValue = Math.log10(value);
    const distance = Math.abs(logValue + 3); // Optimal around 0.001
    return 0.9 - 0.2 * (distance / 3);
  });
  
  const modelComplexityData = decayValues.map(value => {
    // Model complexity decreases with higher regularization
    return 0.9 - 0.7 * (value ** 0.5);
  });
  
  // Create chart
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: decayLabels,
      datasets: [
        {
          label: 'Training Accuracy',
          data: trainAccData,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Validation Accuracy',
          data: valAccData,
          borderColor: 'rgba(153, 102, 255, 1)',
          backgroundColor: 'rgba(153, 102, 255, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        },
        {
          label: 'Model Complexity',
          data: modelComplexityData,
          borderColor: 'rgba(255, 159, 64, 1)',
          backgroundColor: 'rgba(255, 159, 64, 0.2)',
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'Weight Decay/Regularization Impact'
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Weight Decay Value'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Relative Value'
          },
          min: 0,
          max: 1
        }
      }
    }
  });
  
  // Add annotation for current value
  addCurrentValueMarker(container, decayValue, currentDecayIndex, decayValues);
}

// Helper function to find closest index in an array
function findClosestIndex(array, value) {
  let closestIndex = 0;
  let minDiff = Math.abs(array[0] - value);
  
  for (let i = 1; i < array.length; i++) {
    const diff = Math.abs(array[i] - value);
    if (diff < minDiff) {
      minDiff = diff;
      closestIndex = i;
    }
  }
  
  return closestIndex;
}

// Helper function to add current value marker
function addCurrentValueMarker(container, value, index, valueArray) {
  const highlight = document.createElement('div');
  highlight.style.textAlign = 'center';
  highlight.style.marginTop = '10px';
  highlight.style.padding = '8px';
  highlight.style.backgroundColor = 'rgba(75, 192, 192, 0.1)';
  highlight.style.border = '1px solid rgba(75, 192, 192, 0.5)';
  highlight.style.borderRadius = '4px';
  highlight.style.fontSize = '14px';
  
  let valueDisplay = value;
  if (valueArray[index] !== value) {
    // Show both the actual value and the closest value used in the chart
    valueDisplay = `${value} (closest to ${valueArray[index].toFixed(4)})`;
  }
  
  highlight.innerHTML = `
    <span style="font-weight: 600;">Current value:</span>
    <span style="color: #4F46E5; font-weight: 500; margin-left: 5px;">${valueDisplay}</span>
  `;
  
  container.appendChild(highlight);
}