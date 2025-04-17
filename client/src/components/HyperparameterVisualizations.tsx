import React, { useEffect, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import Chart from 'chart.js/auto';

interface HyperparameterVisualizationProps {
  paramName: string;
  paramValue: string;
  framework?: string;
}

// Define dataset types to fix TypeScript errors
interface DatasetEntry {
  label: string;
  data: number[];
  borderColor: string;
  backgroundColor: string;
  tension?: number;
  yAxisID?: string;
  borderWidth?: number;
  borderDash?: number[];
  pointStyle?: string;
  pointRadius?: number;
  type?: 'line' | 'bar';
  stack?: string;
  fill?: boolean;
  order?: number;
}

interface VisualizationData {
  type: 'line' | 'bar' | 'radar' | 'scatter' | 'bubble' | 'polarArea';
  title: string;
  xLabel: string;
  yLabel: string;
  data: {
    labels: string[];
    datasets: DatasetEntry[];
  };
  secondaryType?: 'line' | 'bar'; // For combined charts
  options?: any; // Any additional chart-specific options
}

export function HyperparameterVisualizations({ 
  paramName, 
  paramValue,
  framework 
}: HyperparameterVisualizationProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  useEffect(() => {
    console.log('HyperparameterVisualizations useEffect triggered', { paramName, paramValue });
    
    if (!chartRef.current) {
      console.error('Chart ref is null in HyperparameterVisualizations');
      return;
    }
    
    // Set up ResizeObserver to handle chart sizing
    const resizeObserver = new ResizeObserver(entries => {
      console.log('Chart ResizeObserver triggered', entries[0].contentRect);
      
      if (entries[0].contentRect.width > 0 && entries[0].contentRect.height > 0) {
        initializeChart();
      }
    });
    
    // Observe the canvas
    resizeObserver.observe(chartRef.current);
    
    // Initialize chart with delay to ensure it's visible
    const timeoutId = setTimeout(() => {
      initializeChart();
    }, 500);
    
    // Function to initialize or reinitialize the chart
    function initializeChart() {
      if (!chartRef.current) {
        console.error('Chart ref is null during initialization');
        return;
      }
      
      // Clean up previous chart instance
      if (chartInstanceRef.current) {
        console.log('Destroying previous chart instance');
        chartInstanceRef.current.destroy();
      }
      
      console.log('Getting visualization data for', paramName, paramValue);
      const visualization = getVisualizationData(paramName, paramValue);
      console.log('Visualization data:', visualization);
      
      const ctx = chartRef.current.getContext('2d');
      if (!ctx) {
        console.error('Failed to get 2D context from canvas');
        return;
      }
      
      // Make sure canvas has dimensions
      const rect = chartRef.current.getBoundingClientRect();
      console.log('Chart canvas dimensions:', rect.width, 'x', rect.height);
      
      if (rect.width === 0 || rect.height === 0) {
        console.error('Chart canvas has zero width or height!');
        return;
      }
      
      // Set explicit dimensions just to be sure
      const dpr = window.devicePixelRatio || 1;
      chartRef.current.width = Math.max(1, rect.width * dpr);
      chartRef.current.height = Math.max(1, rect.height * dpr);
      chartRef.current.style.width = `${rect.width}px`;
      chartRef.current.style.height = `${rect.height}px`;
      
      try {
        // Create the default chart options
        const defaultOptions = {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'top',
              labels: {
                usePointStyle: true,
                padding: 15,
                boxWidth: 10,
                boxHeight: 10,
                font: {
                  size: 11
                }
              }
            },
            title: {
              display: true,
              text: visualization.title,
              font: {
                size: 14,
                weight: 'bold'
              },
              padding: {
                bottom: 15
              }
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  // Default label formatter for regular charts
                  if (context.parsed.y !== undefined) {
                    const value = context.parsed.y;
                    // Format value based on metric type
                    if (context.dataset.label.includes('Accuracy')) {
                      return ` ${context.dataset.label}: ${(value * 100).toFixed(1)}%`;
                    } else if (context.dataset.label.includes('Time')) {
                      return ` ${context.dataset.label}: ${value.toFixed(2)}x`;
                    }
                    return ` ${context.dataset.label}: ${value.toFixed(3)}`;
                  }
                  // Return the raw value for other chart types
                  return ` ${context.dataset.label}: ${context.raw}`;
                }
              }
            }
          },
          scales: {
            x: {
              title: {
                display: visualization.xLabel !== '',
                text: visualization.xLabel,
                font: {
                  weight: 'bold'
                },
                padding: {
                  top: 10
                }
              },
              grid: {
                display: false
              }
            },
            y: {
              title: {
                display: visualization.yLabel !== '',
                text: visualization.yLabel,
                font: {
                  weight: 'bold'
                }
              },
              beginAtZero: false,
              grid: {
                color: 'rgba(0, 0, 0, 0.05)'
              },
              ticks: {
                callback: function(value) {
                  if (visualization.yLabel.includes('Accuracy') || 
                      visualization.data.datasets.some(d => d.label?.includes('Accuracy'))) {
                    return (value * 100).toFixed(0) + '%';
                  }
                  return value;
                }
              }
            }
          },
          interaction: {
            mode: 'index',
            intersect: false
          },
          animations: {
            tension: {
              duration: 1000,
              easing: 'easeOutQuad',
              from: 0.2,
              to: 0.4
            }
          }
        };
        
        // Merge default options with visualization-specific options
        const chartOptions = visualization.options 
          ? { ...defaultOptions, ...visualization.options } 
          : defaultOptions;
        
        // Special handling for chart types that don't use x-y axes
        if (['radar', 'polarArea', 'pie', 'doughnut'].includes(visualization.type)) {
          // Remove x and y scales for radial charts
          delete chartOptions.scales.x;
          delete chartOptions.scales.y;
        }
        
        // Create new chart with the merged options
        chartInstanceRef.current = new Chart(ctx, {
          type: visualization.type,
          data: visualization.data,
          options: chartOptions
        });
        
        console.log('Chart created successfully');
      } catch (error) {
        console.error('Error creating chart:', error);
      }
    }
    
    // Cleanup function
    return () => {
      console.log('Cleaning up HyperparameterVisualizations');
      clearTimeout(timeoutId);
      resizeObserver.disconnect();
      
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [paramName, paramValue, framework]);

  return (
    <Card className="w-full mt-4">
      <CardHeader>
        <CardTitle className="text-lg">
          Impact Visualization
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full h-64">
          <canvas ref={chartRef} />
        </div>
        <div className="mt-4 text-sm text-gray-500">
          <p>This visualization shows how different values for <strong>{paramName}</strong> affect multiple aspects of model performance.</p>
          <ul className="mt-2 list-disc list-inside text-xs space-y-1">
            <li><span className="text-emerald-700 font-medium">Solid lines</span> represent primary metrics like accuracy or loss</li>
            <li><span className="text-pink-700 font-medium">Dashed lines</span> represent trade-offs like complexity, memory usage or computational cost</li>
            <li>The optimal range depends on your specific dataset, hardware constraints, and problem requirements</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

// Function to generate visualization data based on hyperparameter name and value
function getVisualizationData(paramName: string, paramValue: string): VisualizationData {
  // Convert parameter value to number if possible
  const numValue = parseFloat(paramValue);
  
  // Generate a unique seed for randomization based on parameter name
  const seed = paramName.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const random = () => {
    const x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
  };
  
  // Helper function to generate a complementary color to a given color
  const getComplementaryColor = (rgbaStr: string): string => {
    // Extract RGB values from rgba string
    const match = rgbaStr.match(/rgba\((\d+),\s*(\d+),\s*(\d+)/);
    if (!match) return 'rgba(100, 100, 100, 1)';
    
    const r = 255 - parseInt(match[1]);
    const g = 255 - parseInt(match[2]);
    const b = 255 - parseInt(match[3]);
    
    return `rgba(${r}, ${g}, ${b}, 1)`;
  };
  
  // Set of predefined color schemes for different parameter types
  const colorSchemes = {
    accuracy: {
      primary: 'rgba(75, 192, 192, 1)',  // Teal
      secondary: 'rgba(153, 102, 255, 1)', // Purple
      tertiary: 'rgba(255, 99, 132, 1)',   // Pink
    },
    speed: {
      primary: 'rgba(255, 159, 64, 1)',   // Orange
      secondary: 'rgba(54, 162, 235, 1)',  // Blue
      tertiary: 'rgba(255, 205, 86, 1)',   // Yellow
    },
    memory: {
      primary: 'rgba(54, 162, 235, 1)',   // Blue
      secondary: 'rgba(255, 99, 132, 1)',  // Pink
      tertiary: 'rgba(75, 192, 192, 1)',   // Teal
    },
    stability: {
      primary: 'rgba(153, 102, 255, 1)',  // Purple
      secondary: 'rgba(255, 205, 86, 1)',  // Yellow
      tertiary: 'rgba(54, 162, 235, 1)',   // Blue
    }
  };
  
  // Choose color scheme based on parameter type
  let colorScheme;
  if (paramName.includes('learning_rate') || paramName.includes('lr')) {
    colorScheme = colorSchemes.speed;
  } else if (paramName.includes('batch_size') || paramName.includes('memory')) {
    colorScheme = colorSchemes.memory;
  } else if (paramName.includes('dropout') || paramName.includes('regularization')) {
    colorScheme = colorSchemes.stability;
  } else {
    colorScheme = colorSchemes.accuracy;
  }
  
  // Default visualization configuration
  let visualization: VisualizationData = {
    type: 'line',
    title: `${paramName.replace(/_/g, ' ')} Impact on Model`,
    xLabel: 'Value',
    yLabel: 'Performance',
    data: {
      labels: [] as string[],
      datasets: [] as DatasetEntry[]
    },
    options: {
      animation: {
        duration: 2000,
        easing: 'easeOutQuart'
      }
    }
  };
  
  // Generate different visualizations based on parameter name
  if (paramName.includes('learning_rate') || paramName.includes('lr')) {
    const labels = [
      '0.00001', '0.0001', '0.001', '0.01', '0.1', '1.0'
    ];
    
    visualization = {
      type: 'line',
      title: 'Learning Rate Impact on Model Performance',
      xLabel: 'Learning Rate (log scale)',
      yLabel: 'Model Performance',
      data: {
        labels,
        datasets: [
          {
            label: 'Training Accuracy',
            data: [0.55, 0.65, 0.78, 0.88, 0.82, 0.70],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Accuracy',
            data: [0.54, 0.64, 0.76, 0.84, 0.76, 0.62],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          },
          {
            label: 'Convergence Speed',
            data: [0.1, 0.3, 0.6, 0.8, 0.95, 1.0],
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.4,
            borderDash: [5, 5]
          }
        ]
      }
    };
  } 
  else if (paramName.includes('batch_size')) {
    const labels = ['4', '8', '16', '32', '64', '128', '256', '512'];
    
    visualization = {
      type: 'line',
      title: 'Batch Size Impact on Training',
      xLabel: 'Batch Size',
      yLabel: 'Metric Value',
      data: {
        labels,
        datasets: [
          {
            label: 'Training Accuracy',
            data: [0.89, 0.87, 0.86, 0.85, 0.83, 0.81, 0.80, 0.78],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.2
          },
          {
            label: 'Training Time (relative)',
            data: [2.5, 1.8, 1.2, 1.0, 0.7, 0.5, 0.4, 0.35],
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.2,
            borderDash: [5, 5]
          },
          {
            label: 'Memory Usage (relative)',
            data: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            borderColor: 'rgba(54, 162, 235, 1)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            tension: 0.2,
            borderDash: [5, 5]
          }
        ]
      }
    };
  }
  else if (paramName.includes('dropout')) {
    const labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'];
    
    visualization = {
      type: 'line',
      title: 'Dropout Rate Impact on Model Performance',
      xLabel: 'Dropout Rate',
      yLabel: 'Accuracy',
      data: {
        labels,
        datasets: [
          {
            label: 'Training Accuracy',
            data: [0.99, 0.97, 0.95, 0.92, 0.88, 0.84, 0.78, 0.72, 0.65, 0.55],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Accuracy',
            data: [0.78, 0.82, 0.86, 0.89, 0.90, 0.88, 0.85, 0.80, 0.72, 0.60],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          },
          {
            label: 'Generalization Gap',
            data: [0.21, 0.15, 0.09, 0.03, -0.02, -0.04, -0.07, -0.08, -0.07, -0.05],
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.4,
            borderDash: [5, 5]
          }
        ]
      }
    };
  }
  else if (paramName.includes('epoch') || paramName.includes('num_epochs') || paramName.includes('n_epochs')) {
    const labels = ['1', '5', '10', '20', '50', '100', '200', '500'];
    
    visualization = {
      type: 'line',
      title: 'Training Epochs Impact on Performance',
      xLabel: 'Number of Epochs',
      yLabel: 'Accuracy',
      data: {
        labels,
        datasets: [
          {
            label: 'Training Accuracy',
            data: [0.70, 0.85, 0.92, 0.97, 0.99, 0.995, 0.998, 0.999],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Accuracy',
            data: [0.68, 0.82, 0.86, 0.87, 0.86, 0.85, 0.84, 0.83],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          },
          {
            label: 'Overfitting Risk',
            data: [0.05, 0.15, 0.25, 0.4, 0.6, 0.75, 0.85, 0.95],
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.4,
            borderDash: [5, 5]
          }
        ]
      }
    };
  }
  else if (paramName.includes('weight_decay') || paramName.includes('l2') || paramName.includes('regularization')) {
    const labels = ['0.0', '0.0001', '0.001', '0.01', '0.1', '0.5', '1.0'];
    
    visualization = {
      type: 'line',
      title: 'Weight Decay (L2 Regularization) Impact',
      xLabel: 'Weight Decay Factor',
      yLabel: 'Performance Metrics',
      data: {
        labels,
        datasets: [
          {
            label: 'Training Loss',
            data: [0.05, 0.07, 0.10, 0.15, 0.25, 0.40, 0.60],
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Loss',
            data: [0.20, 0.15, 0.12, 0.14, 0.22, 0.45, 0.70],
            borderColor: 'rgba(54, 162, 235, 1)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            tension: 0.4
          },
          {
            label: 'Model Complexity',
            data: [1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.05],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4,
            borderDash: [5, 5]
          }
        ]
      }
    };
  }
  else if (paramName.includes('optimizer')) {
    // A bubble chart for optimizers using 3 dimensions of data
    // X-axis: Convergence Speed, Y-axis: Final Accuracy, Bubble size: Memory Requirements
    const optimizers = [
      { name: 'SGD', speed: 0.4, accuracy: 0.88, memory: 0.2, tuning: 0.85 },
      { name: 'SGD+Momentum', speed: 0.65, accuracy: 0.90, memory: 0.25, tuning: 0.75 },
      { name: 'Adam', speed: 0.9, accuracy: 0.92, memory: 0.6, tuning: 0.35 },
      { name: 'RMSprop', speed: 0.7, accuracy: 0.90, memory: 0.5, tuning: 0.45 },
      { name: 'AdaGrad', speed: 0.6, accuracy: 0.87, memory: 0.45, tuning: 0.55 },
      { name: 'AdamW', speed: 0.85, accuracy: 0.93, memory: 0.65, tuning: 0.40 },
      { name: 'LAMB', speed: 0.8, accuracy: 0.91, memory: 0.7, tuning: 0.5 }
    ];
    
    // Highlight the current optimizer if it's in our list
    const currentOptimizer = paramValue.toLowerCase();
    const pointBorderWidth = optimizers.map(opt => 
      opt.name.toLowerCase().includes(currentOptimizer) ? 4 : 2
    );
    
    // Create a set of gradient colors for the bubbles
    const colors = optimizers.map((opt, i) => {
      const hue = (i * 360 / optimizers.length) % 360;
      const highlighted = opt.name.toLowerCase().includes(currentOptimizer);
      return {
        bg: `hsla(${hue}, 70%, ${highlighted ? 60 : 50}%, 0.6)`,
        border: `hsla(${hue}, 70%, ${highlighted ? 45 : 35}%, 1)`
      };
    });
    
    // For a scatter/bubble chart
    visualization = {
      type: 'bubble',
      title: 'Optimizer Performance Comparison',
      xLabel: 'Convergence Speed',
      yLabel: 'Final Accuracy',
      data: {
        labels: optimizers.map(opt => opt.name),
        datasets: [{
          label: 'Optimizers (bubble size = memory usage)',
          data: optimizers.map((opt, i) => ({
            x: opt.speed,
            y: opt.accuracy,
            r: opt.memory * 30  // Scale for nice bubble size
          })),
          backgroundColor: colors.map(c => c.bg),
          borderColor: colors.map(c => c.border),
          borderWidth: pointBorderWidth,
          // Add hover effects
          hoverBackgroundColor: colors.map(c => c.bg.replace('0.6', '0.8')),
          hoverBorderWidth: pointBorderWidth.map(w => w + 2),
        }]
      },
      options: {
        plugins: {
          tooltip: {
            callbacks: {
              label: function(context) {
                const index = context.dataIndex;
                const opt = optimizers[index];
                return [
                  `${opt.name}`,
                  `Convergence Speed: ${(opt.speed * 100).toFixed(0)}%`,
                  `Accuracy: ${(opt.accuracy * 100).toFixed(1)}%`,
                  `Memory Usage: ${(opt.memory * 100).toFixed(0)}%`,
                  `Tuning Difficulty: ${(opt.tuning * 100).toFixed(0)}%`
                ];
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Convergence Speed'
            },
            min: 0.2,
            max: 1.0
          },
          y: {
            title: {
              display: true,
              text: 'Final Accuracy'
            },
            min: 0.8,
            max: 0.95
          }
        },
        animation: {
          duration: 2000
        }
      }
    };
  }
  else if (paramName.includes('momentum')) {
    const labels = ['0.0', '0.3', '0.5', '0.7', '0.9', '0.95', '0.99'];
    
    visualization = {
      type: 'line',
      title: 'Momentum Impact on Training',
      xLabel: 'Momentum Value',
      yLabel: 'Performance',
      data: {
        labels,
        datasets: [
          {
            label: 'Convergence Speed',
            data: [0.3, 0.5, 0.6, 0.7, 0.9, 0.93, 0.85],
            borderColor: 'rgba(255, 159, 64, 1)',
            backgroundColor: 'rgba(255, 159, 64, 0.2)',
            tension: 0.4
          },
          {
            label: 'Stability',
            data: [0.9, 0.85, 0.82, 0.8, 0.7, 0.65, 0.6],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          },
          {
            label: 'Ability to Escape Local Minima',
            data: [0.1, 0.3, 0.45, 0.6, 0.8, 0.9, 0.95],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4,
            borderDash: [5, 5]
          }
        ]
      }
    };
  }
  else if (paramName.includes('activation')) {
    // For activation functions, use a radar chart to show multiple dimensions
    const activationFunctions = ['ReLU', 'Leaky ReLU', 'ELU', 'SELU', 'Tanh', 'Sigmoid', 'GELU', 'Swish'];
    
    // Highlight the current activation function if possible
    const currentActivation = paramValue.toUpperCase();
    const pointRadiuses = activationFunctions.map(a => 
      a.toUpperCase().includes(currentActivation) ? 8 : 3
    );
    
    visualization = {
      type: 'radar',
      title: 'Activation Function Properties',
      xLabel: '',
      yLabel: '',
      data: {
        labels: ['Training Speed', 'Gradient Flow', 'Vanishing Gradient Resistance', 
                 'Representation Power', 'Biological Plausibility', 'Computational Efficiency'],
        datasets: activationFunctions.map((activation, index) => {
          // Create distinct but related colors for each activation
          const hue = (index * 45) % 360;
          const color = `hsla(${hue}, 70%, 60%, 0.7)`;
          const borderColor = `hsla(${hue}, 70%, 50%, 1)`;
          
          // Generate semi-random but characteristic values for each activation
          // These values are approximations of known properties
          let values;
          switch(activation) {
            case 'ReLU':
              values = [0.95, 0.70, 0.60, 0.75, 0.40, 0.95];
              break;
            case 'Leaky ReLU':
              values = [0.92, 0.85, 0.80, 0.82, 0.45, 0.90];
              break;
            case 'ELU':
              values = [0.88, 0.88, 0.85, 0.88, 0.50, 0.85];
              break;
            case 'SELU':
              values = [0.85, 0.90, 0.95, 0.91, 0.55, 0.80];
              break;
            case 'Tanh':
              values = [0.70, 0.80, 0.60, 0.87, 0.70, 0.75];
              break;
            case 'Sigmoid':
              values = [0.65, 0.65, 0.40, 0.80, 0.85, 0.70];
              break;
            case 'GELU':
              values = [0.90, 0.87, 0.82, 0.89, 0.60, 0.85];
              break;
            case 'Swish':
              values = [0.93, 0.89, 0.84, 0.90, 0.65, 0.87];
              break;
            default:
              values = [0.80, 0.80, 0.80, 0.80, 0.80, 0.80];
          }
          
          return {
            label: activation,
            data: values,
            backgroundColor: color,
            borderColor: borderColor,
            borderWidth: activation.toUpperCase().includes(currentActivation) ? 3 : 1,
            pointRadius: pointRadiuses[index],
            pointBackgroundColor: borderColor,
          };
        })
      },
      options: {
        scales: {
          r: {
            angleLines: {
              display: true,
              color: 'rgba(0, 0, 0, 0.1)',
            },
            suggestedMin: 0,
            suggestedMax: 1,
            ticks: {
              stepSize: 0.2
            },
            pointLabels: {
              font: {
                size: 10
              }
            }
          }
        },
        animation: {
          duration: 2000
        },
        plugins: {
          legend: {
            position: 'right'
          }
        }
      }
    };
  }
  else if (paramName.includes('hidden_units') || paramName.includes('units') || paramName.includes('hidden_size')) {
    const labels = ['16', '32', '64', '128', '256', '512', '1024', '2048'];
    
    visualization = {
      type: 'line',
      title: 'Hidden Units Impact on Model Capacity',
      xLabel: 'Number of Hidden Units',
      yLabel: 'Performance Metrics',
      data: {
        labels,
        datasets: [
          {
            label: 'Training Accuracy',
            data: [0.75, 0.82, 0.87, 0.91, 0.94, 0.96, 0.98, 0.99],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Accuracy',
            data: [0.74, 0.80, 0.85, 0.88, 0.89, 0.89, 0.88, 0.87],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          },
          {
            label: 'Memory Usage (relative)',
            data: [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0],
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.4,
            borderDash: [5, 5]
          }
        ]
      }
    };
  }
  else if (paramName.includes('kernel_size')) {
    // For kernel size, create a visualization that highlights the receptive field
    // and computational cost trade-offs in a more dynamic way
    const kernelSizes = ['1x1', '3x3', '5x5', '7x7', '9x9', '11x11'];
    
    // Find the closest kernel size to highlight
    const currSize = parseInt(paramValue) || 3;
    let closestIndex = 0;
    let smallestDiff = Infinity;
    
    kernelSizes.forEach((size, idx) => {
      const sizeNum = parseInt(size.split('x')[0]);
      const diff = Math.abs(sizeNum - currSize);
      if (diff < smallestDiff) {
        smallestDiff = diff;
        closestIndex = idx;
      }
    });

    // Create a combined chart with feature extraction as a line and computational cost as bars
    const baseData = {
      featureExtraction: [0.60, 0.85, 0.92, 0.95, 0.94, 0.91],
      computationalCost: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
      receptiveField: [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    };
    
    // Create a polarArea chart for a more visual representation of kernel size effects
    const polarData = kernelSizes.map((size, idx) => {
      return {
        size,
        extraction: baseData.featureExtraction[idx],
        cost: baseData.computationalCost[idx],
        field: baseData.receptiveField[idx],
        overall: (baseData.featureExtraction[idx] * 0.8) - (baseData.computationalCost[idx] * 0.4)
      };
    });
    
    // Create background colors with the highlighted kernel size
    const bgColors = kernelSizes.map((_, idx) => {
      const alpha = idx === closestIndex ? 0.8 : 0.5;
      return `rgba(54, 162, 235, ${alpha})`;
    });
    
    visualization = {
      type: 'polarArea',
      title: 'Kernel Size Impact on CNN',
      xLabel: '',
      yLabel: '',
      data: {
        labels: kernelSizes.map(size => `${size} Kernel`),
        datasets: [
          {
            label: 'Feature Extraction Capability',
            data: polarData.map(d => d.extraction * 100),
            backgroundColor: bgColors,
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: kernelSizes.map((_, idx) => idx === closestIndex ? 2 : 1)
          }
        ]
      },
      options: {
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            ticks: {
              stepSize: 20
            }
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: function(context) {
                const idx = context.dataIndex;
                const data = polarData[idx];
                return [
                  `${data.size} Kernel:`,
                  `Feature Quality: ${(data.extraction * 100).toFixed(0)}%`,
                  `Computational Cost: ${(data.cost * 100).toFixed(0)}%`,
                  `Receptive Field Size: ${(data.field * 100).toFixed(0)}%`,
                ];
              }
            }
          },
          legend: {
            display: false
          }
        },
        animation: {
          animateRotate: true,
          animateScale: true,
          duration: 2000
        }
      }
    };
  }
  else {
    // Generic visualization for any other parameter
    const baseValue = numValue || 0.5;
    const spread = baseValue * 0.8;
    const min = Math.max(0, baseValue - spread);
    const max = baseValue + spread;
    const step = (max - min) / 4;
    
    const labels = [];
    for (let i = 0; i <= 4; i++) {
      labels.push((min + i * step).toFixed(4));
    }
    
    // Generic performance curve (bell curve around optimal value)
    const performanceData = labels.map((val, index) => {
      const distance = Math.abs(parseFloat(val) - baseValue);
      return 0.9 - 0.2 * (distance / spread) ** 2;
    });
    
    // Generic cost curve (increasing with parameter value)
    const costData = labels.map((val) => {
      const normalizedVal = (parseFloat(val) - min) / (max - min);
      return 0.3 + 0.5 * normalizedVal;
    });
    
    // Generic complexity curve
    const complexityData = labels.map((val) => {
      const normalizedVal = (parseFloat(val) - min) / (max - min);
      return 0.4 + 0.4 * normalizedVal;
    });
    
    visualization = {
      type: 'line',
      title: `${paramName} Impact on Model Performance`,
      xLabel: `${paramName} Value`,
      yLabel: 'Relative Performance',
      data: {
        labels,
        datasets: [
          {
            label: 'Model Performance',
            data: performanceData,
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4
          },
          {
            label: 'Computational Cost',
            data: costData,
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.4
          },
          {
            label: 'Model Complexity',
            data: complexityData,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4,
            borderDash: [5, 5]
          }
        ]
      }
    };
  }
  
  return visualization;
}