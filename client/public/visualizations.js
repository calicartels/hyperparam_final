// Hyperparameter Visualization Library for HyperExplainer

// Visualization data for different hyperparameters
const visualizationData = {
  learning_rate: {
    title: "Learning Rate Visualization",
    description: "How learning rate affects model training over time",
    xAxisLabel: "Training Iterations",
    yAxisLabel: "Loss",
    visualization: function(container, value) {
      // Parse the current value
      const currentValue = parseFloat(value);
      
      // Generate data based on current value
      const lowLR = generateLearningRateCurve(Math.max(0.0001, currentValue / 10));
      const currentLR = generateLearningRateCurve(currentValue);
      const highLR = generateLearningRateCurve(currentValue * 10);
      
      // Create canvas for chart
      const canvas = document.createElement('canvas');
      canvas.width = 400;
      canvas.height = 200;
      container.appendChild(canvas);
      
      // Create chart
      const ctx = canvas.getContext('2d');
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: Array.from({length: 100}, (_, i) => i + 1),
          datasets: [{
            label: 'Lower Learning Rate',
            data: lowLR,
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 2,
            fill: false,
            tension: 0.4
          }, {
            label: 'Current Learning Rate',
            data: currentLR,
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 3,
            fill: false,
            tension: 0.4
          }, {
            label: 'Higher Learning Rate',
            data: highLR,
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            fill: false,
            tension: 0.4
          }]
        },
        options: {
          responsive: true,
          interaction: {
            mode: 'index',
            intersect: false
          },
          plugins: {
            title: {
              display: true,
              text: 'Learning Rate Effect on Training Loss'
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return context.dataset.label + ': ' + context.parsed.y.toFixed(2);
                }
              }
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Training Iterations'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Loss'
              }
            }
          }
        }
      });
      
      // Add explanation text below chart
      const explanation = document.createElement('div');
      explanation.innerHTML = `
        <div style="margin-top: 16px; font-size: 14px; color: #334155;">
          <p><strong>How Learning Rate Affects Training:</strong></p>
          <ul style="padding-left: 20px; margin-top: 8px;">
            <li><span style="color: rgba(75, 192, 192, 1);">Lower learning rate (${(currentValue/10).toExponential(2)})</span>: Converges more slowly but can be more stable.</li>
            <li><span style="color: rgba(54, 162, 235, 1);">Current learning rate (${currentValue.toExponential(2)})</span>: Your selected rate.</li>
            <li><span style="color: rgba(255, 99, 132, 1);">Higher learning rate (${(currentValue*10).toExponential(2)})</span>: Converges faster initially but may oscillate or diverge.</li>
          </ul>
          <p style="margin-top: 8px;"><strong>Key Insight:</strong> The ideal learning rate finds balance between convergence speed and stability.</p>
        </div>
      `;
      container.appendChild(explanation);
    }
  },
  
  batch_size: {
    title: "Batch Size Effects",
    description: "How batch size affects training variance and efficiency",
    xAxisLabel: "Training Iterations",
    yAxisLabel: "Gradient Variance",
    visualization: function(container, value) {
      // Parse the current batch size
      const currentBatchSize = parseInt(value);
      
      // Create canvas for chart
      const canvas = document.createElement('canvas');
      canvas.width = 400;
      canvas.height = 200;
      container.appendChild(canvas);
      
      // Generate data based on batch size
      const smallBatch = generateBatchSizeVariance(Math.max(1, currentBatchSize / 4));
      const currentBatch = generateBatchSizeVariance(currentBatchSize);
      const largeBatch = generateBatchSizeVariance(currentBatchSize * 4);
      
      // Create chart
      const ctx = canvas.getContext('2d');
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: Array.from({length: 50}, (_, i) => i + 1),
          datasets: [{
            label: 'Smaller Batch Size',
            data: smallBatch,
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 2,
            fill: false
          }, {
            label: 'Current Batch Size',
            data: currentBatch,
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 3,
            fill: false
          }, {
            label: 'Larger Batch Size',
            data: largeBatch,
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 2,
            fill: false
          }]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: 'Batch Size Effect on Gradient Variance'
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return context.dataset.label + ': ' + context.parsed.y.toFixed(2);
                }
              }
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Training Iterations'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Gradient Variance'
              }
            }
          }
        }
      });
      
      // Add explanation text
      const explanation = document.createElement('div');
      explanation.innerHTML = `
        <div style="margin-top: 16px; font-size: 14px; color: #334155;">
          <p><strong>How Batch Size Affects Training:</strong></p>
          <ul style="padding-left: 20px; margin-top: 8px;">
            <li><span style="color: rgba(255, 99, 132, 1);">Smaller batch size (${Math.max(1, currentBatchSize/4)})</span>: Higher variance in gradient updates, but can help escape local minima.</li>
            <li><span style="color: rgba(54, 162, 235, 1);">Current batch size (${currentBatchSize})</span>: Your selected batch size.</li>
            <li><span style="color: rgba(75, 192, 192, 1);">Larger batch size (${currentBatchSize*4})</span>: More stable updates but may converge to poorer solutions and requires more memory.</li>
          </ul>
          <p style="margin-top: 8px;"><strong>Memory Usage:</strong> Larger batch sizes consume more GPU/CPU memory, which can be a limiting factor.</p>
          <p><strong>Key Insight:</strong> Batch sizes that are powers of 2 (32, 64, 128) often provide the best hardware utilization.</p>
        </div>
      `;
      container.appendChild(explanation);
    }
  },
  
  dropout_rate: {
    title: "Dropout Rate Visualization",
    description: "How dropout affects model robustness and overfitting",
    xAxisLabel: "Epochs",
    yAxisLabel: "Error",
    visualization: function(container, value) {
      // Parse the current dropout rate
      const currentDropout = parseFloat(value);
      
      // Create canvas for chart
      const canvas = document.createElement('canvas');
      canvas.width = 400;
      canvas.height = 200;
      container.appendChild(canvas);
      
      // Generate synthetic training and validation errors for different dropout rates
      const epochs = Array.from({length: 50}, (_, i) => i + 1);
      
      // No dropout
      const noDropoutTrain = generateDropoutCurves(0, "train");
      const noDropoutVal = generateDropoutCurves(0, "val");
      
      // Current dropout
      const currentDropoutTrain = generateDropoutCurves(currentDropout, "train");
      const currentDropoutVal = generateDropoutCurves(currentDropout, "val");
      
      // High dropout
      const highDropoutTrain = generateDropoutCurves(Math.min(0.8, currentDropout * 2), "train");
      const highDropoutVal = generateDropoutCurves(Math.min(0.8, currentDropout * 2), "val");
      
      // Create chart
      const ctx = canvas.getContext('2d');
      
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'No Dropout - Training',
              data: noDropoutTrain,
              borderColor: 'rgba(255, 99, 132, 1)',
              backgroundColor: 'rgba(255, 99, 132, 0.1)',
              borderWidth: 2,
              borderDash: [5, 5],
              fill: false
            },
            {
              label: 'No Dropout - Validation',
              data: noDropoutVal,
              borderColor: 'rgba(255, 99, 132, 1)',
              backgroundColor: 'rgba(255, 99, 132, 0.1)',
              borderWidth: 2,
              fill: false
            },
            {
              label: `Current Dropout (${currentDropout}) - Training`,
              data: currentDropoutTrain,
              borderColor: 'rgba(54, 162, 235, 1)',
              backgroundColor: 'rgba(54, 162, 235, 0.1)',
              borderWidth: 2,
              borderDash: [5, 5],
              fill: false
            },
            {
              label: `Current Dropout (${currentDropout}) - Validation`,
              data: currentDropoutVal,
              borderColor: 'rgba(54, 162, 235, 1)',
              backgroundColor: 'rgba(54, 162, 235, 0.1)',
              borderWidth: 2,
              fill: false
            },
            {
              label: `Higher Dropout (${Math.min(0.8, currentDropout * 2).toFixed(1)}) - Training`,
              data: highDropoutTrain,
              borderColor: 'rgba(75, 192, 192, 1)',
              backgroundColor: 'rgba(75, 192, 192, 0.1)',
              borderWidth: 2, 
              borderDash: [5, 5],
              fill: false
            },
            {
              label: `Higher Dropout (${Math.min(0.8, currentDropout * 2).toFixed(1)}) - Validation`,
              data: highDropoutVal,
              borderColor: 'rgba(75, 192, 192, 1)',
              backgroundColor: 'rgba(75, 192, 192, 0.1)',
              borderWidth: 2,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: 'Dropout Rate Effect on Model Overfitting'
            },
            tooltip: {
              mode: 'index',
              intersect: false
            },
            legend: {
              labels: {
                usePointStyle: true
              }
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Epochs'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Error'
              }
            }
          }
        }
      });
      
      // Add explanation text
      const explanation = document.createElement('div');
      explanation.innerHTML = `
        <div style="margin-top: 16px; font-size: 14px; color: #334155;">
          <p><strong>How Dropout Affects Overfitting:</strong></p>
          <ul style="padding-left: 20px; margin-top: 8px;">
            <li><span style="color: rgba(255, 99, 132, 1);">No dropout (0.0)</span>: Training error decreases rapidly but validation error increases (overfitting).</li>
            <li><span style="color: rgba(54, 162, 235, 1);">Current dropout (${currentDropout})</span>: Balanced approach with some regularization.</li>
            <li><span style="color: rgba(75, 192, 192, 1);">Higher dropout (${Math.min(0.8, currentDropout * 2).toFixed(1)})</span>: Stronger regularization, but may underfit if too high.</li>
          </ul>
          <p style="margin-top: 8px;"><strong>Key Insight:</strong> Dropout is a regularization technique that prevents overfitting by randomly deactivating neurons during training. The solid lines represent validation error, while dashed lines show training error.</p>
        </div>
      `;
      container.appendChild(explanation);
    }
  },
  
  num_epochs: {
    title: "Number of Epochs Visualization",
    description: "How the number of training epochs affects model performance",
    xAxisLabel: "Epochs",
    yAxisLabel: "Error/Accuracy",
    visualization: function(container, value) {
      // Parse the current epoch value
      const currentEpochs = parseInt(value);
      
      // Create canvas for chart
      const canvas = document.createElement('canvas');
      canvas.width = 400;
      canvas.height = 200;
      container.appendChild(canvas);
      
      // Generate data
      const epochs = Array.from({length: Math.max(30, currentEpochs * 1.5)}, (_, i) => i + 1);
      const trainLoss = generateEpochCurves("train_loss", epochs.length);
      const valLoss = generateEpochCurves("val_loss", epochs.length);
      const trainAcc = generateEpochCurves("train_acc", epochs.length);
      const valAcc = generateEpochCurves("val_acc", epochs.length);
      
      // Mark current epochs setting
      const currentEpochLine = {
        type: 'line',
        mode: 'vertical',
        scaleID: 'x',
        value: currentEpochs,
        borderColor: 'rgba(255, 99, 132, 0.7)',
        borderWidth: 2,
        label: {
          content: 'Current Setting',
          enabled: true,
          position: 'top'
        }
      };
      
      // Mark early stopping point
      const earlyStopValue = Math.floor(epochs.length * 0.6);
      const earlyStopLine = {
        type: 'line',
        mode: 'vertical',
        scaleID: 'x',
        value: earlyStopValue,
        borderColor: 'rgba(54, 162, 235, 0.7)',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          content: 'Early Stopping',
          enabled: true,
          position: 'top'
        }
      };
      
      // Create chart
      const ctx = canvas.getContext('2d');
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: epochs,
          datasets: [
            {
              label: 'Training Loss',
              data: trainLoss,
              borderColor: 'rgba(255, 99, 132, 1)',
              backgroundColor: 'rgba(255, 99, 132, 0.1)',
              borderWidth: 2,
              yAxisID: 'y'
            },
            {
              label: 'Validation Loss',
              data: valLoss,
              borderColor: 'rgba(54, 162, 235, 1)',
              backgroundColor: 'rgba(54, 162, 235, 0.1)',
              borderWidth: 2,
              yAxisID: 'y'
            },
            {
              label: 'Training Accuracy',
              data: trainAcc,
              borderColor: 'rgba(75, 192, 192, 1)',
              backgroundColor: 'rgba(75, 192, 192, 0.1)',
              borderWidth: 2,
              yAxisID: 'y1'
            },
            {
              label: 'Validation Accuracy',
              data: valAcc,
              borderColor: 'rgba(153, 102, 255, 1)',
              backgroundColor: 'rgba(153, 102, 255, 0.1)',
              borderWidth: 2,
              yAxisID: 'y1'
            }
          ]
        },
        options: {
          responsive: true,
          interaction: {
            mode: 'index',
            intersect: false
          },
          plugins: {
            title: {
              display: true,
              text: 'Training Progress Over Epochs'
            },
            annotation: {
              annotations: {
                currentEpochLine,
                earlyStopLine
              }
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Epochs'
              }
            },
            y: {
              display: true,
              position: 'left',
              title: {
                display: true,
                text: 'Loss'
              }
            },
            y1: {
              display: true,
              position: 'right',
              title: {
                display: true,
                text: 'Accuracy'
              },
              min: 0,
              max: 1,
              grid: {
                drawOnChartArea: false
              }
            }
          }
        }
      });
      
      // Add explanation text
      const explanation = document.createElement('div');
      explanation.innerHTML = `
        <div style="margin-top: 16px; font-size: 14px; color: #334155;">
          <p><strong>How Number of Epochs Affects Training:</strong></p>
          <ul style="padding-left: 20px; margin-top: 8px;">
            <li><strong>Current setting:</strong> Training for ${currentEpochs} epochs (vertical red line).</li>
            <li><strong>Early stopping:</strong> The dashed blue line shows when validation metrics stop improving significantly.</li>
            <li><strong>Overfitting:</strong> Notice how validation loss starts increasing after the early stopping point while training loss continues to decrease.</li>
          </ul>
          <p style="margin-top: 8px;"><strong>Key Insight:</strong> Training for too many epochs leads to overfitting, while too few epochs results in underfitting. Early stopping helps find the optimal point.</p>
        </div>
      `;
      container.appendChild(explanation);
    }
  },
  
  weight_decay: {
    title: "Weight Decay Visualization",
    description: "How weight decay (L2 regularization) affects model parameters",
    xAxisLabel: "Training Progress",
    yAxisLabel: "Parameter Magnitude",
    visualization: function(container, value) {
      // Parse the current weight decay value
      const currentDecay = parseFloat(value);
      
      // Create canvas for chart
      const canvas = document.createElement('canvas');
      canvas.width = 400;
      canvas.height = 200;
      container.appendChild(canvas);
      
      // Generate data for different weight decay rates
      const steps = Array.from({length: 100}, (_, i) => i + 1);
      const noDecay = generateWeightDecayCurve(0);
      const lowDecay = generateWeightDecayCurve(currentDecay / 10);
      const mediumDecay = generateWeightDecayCurve(currentDecay);
      const highDecay = generateWeightDecayCurve(currentDecay * 10);
      
      // Create chart
      const ctx = canvas.getContext('2d');
      new Chart(ctx, {
        type: 'line',
        data: {
          labels: steps,
          datasets: [
            {
              label: 'No Weight Decay',
              data: noDecay,
              borderColor: 'rgba(255, 99, 132, 1)',
              borderWidth: 2,
              fill: false
            },
            {
              label: `Low Decay (${(currentDecay/10).toExponential(1)})`,
              data: lowDecay,
              borderColor: 'rgba(75, 192, 192, 1)',
              borderWidth: 2,
              fill: false
            },
            {
              label: `Current Decay (${currentDecay.toExponential(1)})`,
              data: mediumDecay,
              borderColor: 'rgba(54, 162, 235, 1)',
              borderWidth: 3,
              fill: false
            },
            {
              label: `High Decay (${(currentDecay*10).toExponential(1)})`,
              data: highDecay,
              borderColor: 'rgba(153, 102, 255, 1)',
              borderWidth: 2,
              fill: false
            }
          ]
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: 'Weight Decay Effect on Parameter Magnitude'
            },
            tooltip: {
              mode: 'index',
              intersect: false
            }
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: 'Training Progress'
              }
            },
            y: {
              display: true,
              title: {
                display: true,
                text: 'Parameter Magnitude'
              }
            }
          }
        }
      });
      
      // Add explanation text
      const explanation = document.createElement('div');
      explanation.innerHTML = `
        <div style="margin-top: 16px; font-size: 14px; color: #334155;">
          <p><strong>How Weight Decay Affects Model Parameters:</strong></p>
          <ul style="padding-left: 20px; margin-top: 8px;">
            <li><span style="color: rgba(255, 99, 132, 1);">No weight decay</span>: Parameters can grow large, leading to potential overfitting.</li>
            <li><span style="color: rgba(75, 192, 192, 1);">Low weight decay (${(currentDecay/10).toExponential(1)})</span>: Gentle regularization.</li>
            <li><span style="color: rgba(54, 162, 235, 1);">Current weight decay (${currentDecay.toExponential(1)})</span>: Your selected setting.</li>
            <li><span style="color: rgba(153, 102, 255, 1);">High weight decay (${(currentDecay*10).toExponential(1)})</span>: Strong regularization, may lead to underfitting.</li>
          </ul>
          <p style="margin-top: 8px;"><strong>Key Insight:</strong> Weight decay penalizes large parameter values, encouraging the model to find simpler solutions that are more likely to generalize well.</p>
        </div>
      `;
      container.appendChild(explanation);
    }
  }
};

// Helper functions to generate visualization data
function generateLearningRateCurve(lr) {
  // Simulate training loss curve based on learning rate
  const iterations = 100;
  const startLoss = 5;
  const optimalLoss = 0.5;
  
  return Array.from({length: iterations}, (_, i) => {
    if (lr < 0.001) {
      // Very slow convergence for very small learning rates
      return startLoss * Math.exp(-i * lr * 2) + optimalLoss + (Math.random() * 0.05);
    } else if (lr > 0.5) {
      // Oscillation and possible divergence for very large learning rates
      return startLoss * Math.exp(-i * 0.01) * (1 + 0.5 * Math.sin(i * 0.2)) + optimalLoss + (Math.random() * 0.2) + (i > 50 ? (i-50) * 0.05 : 0);
    } else {
      // Normal convergence for reasonable learning rates
      return startLoss * Math.exp(-i * lr * 0.05) + optimalLoss + (Math.random() * 0.1);
    }
  });
}

function generateBatchSizeVariance(batchSize) {
  // Simulate gradient variance based on batch size
  // Smaller batch sizes have higher variance
  const iterations = 50;
  const baseVariance = 1 / Math.sqrt(batchSize);
  
  return Array.from({length: iterations}, (_, i) => {
    // Decay variance over time, but keep the characteristic difference by batch size
    const timeDecay = 1 / (1 + i * 0.05);
    return baseVariance * timeDecay * (1 + Math.random() * 0.5);
  });
}

function generateDropoutCurves(dropoutRate, type) {
  // Simulate training and validation error curves for different dropout rates
  const iterations = 50;
  
  // Parameters to control curve shapes
  const baseTrain = 1.0;
  const baseVal = 1.0;
  const trainDecayFast = 0.15;
  const trainDecaySlow = 0.05;
  const overfitFactor = 0.03 * (1 - dropoutRate * 2); // Higher dropout = less overfitting
  
  return Array.from({length: iterations}, (_, i) => {
    const x = i / iterations;
    
    if (type === "train") {
      // Training error decreases faster with lower dropout
      const trainDecay = dropoutRate < 0.3 ? trainDecayFast : trainDecaySlow;
      return baseTrain * Math.exp(-i * trainDecay) + 0.1 + (Math.random() * 0.05);
    } else {
      // Validation error initially decreases but then increases due to overfitting
      // Higher dropout reduces overfitting
      const baseError = baseVal * Math.exp(-i * 0.06) + 0.2;
      const overfitting = overfitFactor * Math.max(0, i - iterations * 0.3) ** 2;
      return baseError + overfitting + (Math.random() * 0.05);
    }
  });
}

function generateEpochCurves(type, numEpochs) {
  // Generate curves for training and validation loss/accuracy
  const baseValue = type.includes('acc') ? 0.3 : 2.0; // Starting point
  const finalValue = type.includes('acc') ? 0.95 : 0.3; // Asymptotic value
  const overfitStart = Math.floor(numEpochs * 0.6); // When overfitting starts
  
  return Array.from({length: numEpochs}, (_, i) => {
    const progress = i / numEpochs;
    const noise = Math.random() * 0.03;
    
    if (type === 'train_loss') {
      // Training loss keeps decreasing
      return baseValue * Math.exp(-i * 0.1) + 0.1 + noise;
    } else if (type === 'val_loss') {
      // Validation loss decreases, then increases (overfitting)
      const baseError = baseValue * Math.exp(-i * 0.08) + 0.2;
      const overfitting = i > overfitStart ? 0.01 * (i - overfitStart) ** 1.5 : 0;
      return baseError + overfitting + noise;
    } else if (type === 'train_acc') {
      // Training accuracy increases asymptotically
      return finalValue - (finalValue - 0.5) * Math.exp(-i * 0.15) + noise;
    } else { // val_acc
      // Validation accuracy increases, then slightly decreases
      const baseAcc = finalValue - 0.15 - (finalValue - 0.5) * Math.exp(-i * 0.1);
      const overfitting = i > overfitStart ? 0.001 * (i - overfitStart) : 0;
      return baseAcc - overfitting + noise;
    }
  });
}

function generateWeightDecayCurve(decayFactor) {
  // Simulate parameter magnitude growth under different weight decay settings
  const iterations = 100;
  
  return Array.from({length: iterations}, (_, i) => {
    const x = i / iterations;
    
    // Natural growth of parameters without decay
    const naturalGrowth = 8 * (1 - Math.exp(-x * 3));
    
    // Decay effect
    const decay = decayFactor * i * 0.1;
    
    // Add some noise for realism
    const noise = Math.random() * 0.3;
    
    return Math.max(0.5, naturalGrowth * Math.exp(-decay) + noise);
  });
}

// Function to create a visualization for a specific hyperparameter
function createHyperparameterVisualization(paramKey, paramValue) {
  // Return a function that will render the visualization when called
  return function(container) {
    // Check if we have a visualization for this parameter
    if (!visualizationData[paramKey]) {
      container.innerHTML = `
        <div style="padding: 20px; text-align: center; color: #64748b;">
          <p>No visualization available for this hyperparameter.</p>
        </div>
      `;
      return;
    }
    
    // Create visualization title
    const titleElement = document.createElement('h3');
    titleElement.textContent = visualizationData[paramKey].title;
    titleElement.style.cssText = 'font-size: 16px; font-weight: 600; color: #334155; margin-bottom: 8px;';
    container.appendChild(titleElement);
    
    // Create description
    const descElement = document.createElement('p');
    descElement.textContent = visualizationData[paramKey].description;
    descElement.style.cssText = 'font-size: 14px; color: #64748b; margin-bottom: 20px;';
    container.appendChild(descElement);
    
    // Load Chart.js dynamically
    if (typeof Chart === 'undefined') {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/chart.js';
      script.onload = function() {
        // Once loaded, create the visualization
        visualizationData[paramKey].visualization(container, paramValue);
      };
      document.head.appendChild(script);
    } else {
      // Chart.js already loaded, create the visualization directly
      visualizationData[paramKey].visualization(container, paramValue);
    }
  };
}

// Export the visualization creator function
window.createHyperparameterVisualization = createHyperparameterVisualization;