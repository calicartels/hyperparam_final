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
}

interface VisualizationData {
  type: 'line' | 'bar';
  title: string;
  xLabel: string;
  yLabel: string;
  data: {
    labels: string[];
    datasets: DatasetEntry[];
  };
}

export function HyperparameterVisualizations({ 
  paramName, 
  paramValue,
  framework 
}: HyperparameterVisualizationProps) {
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;
    
    // Clean up previous chart instance
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }
    
    const visualization = getVisualizationData(paramName, paramValue);
    
    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;
    
    chartInstanceRef.current = new Chart(ctx, {
      type: visualization.type,
      data: visualization.data,
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: visualization.title
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return ` ${context.dataset.label}: ${context.parsed.y}`;
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: visualization.xLabel
            }
          },
          y: {
            title: {
              display: true,
              text: visualization.yLabel
            },
            beginAtZero: true,
          }
        }
      }
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
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
          <p>This visualization shows how different values for <strong>{paramName}</strong> may affect your model's performance.
             The optimal range depends on your specific dataset and problem.</p>
        </div>
      </CardContent>
    </Card>
  );
}

// Function to generate visualization data based on hyperparameter name and value
function getVisualizationData(paramName: string, paramValue: string): VisualizationData {
  // Convert parameter value to number if possible
  const numValue = parseFloat(paramValue);
  
  // Default visualization configuration
  let visualization: VisualizationData = {
    type: 'line',
    title: 'Parameter Impact',
    xLabel: 'Value',
    yLabel: 'Performance',
    data: {
      labels: [] as string[],
      datasets: [] as DatasetEntry[]
    }
  };
  
  // Generate different visualizations based on parameter name
  if (paramName.includes('learning_rate') || paramName.includes('lr')) {
    const labels = [
      '0.0001', '0.001', '0.01', '0.1', '1.0'
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
            data: [0.65, 0.78, 0.88, 0.82, 0.70],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Accuracy',
            data: [0.64, 0.76, 0.84, 0.76, 0.62],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          }
        ]
      }
    };
  } 
  else if (paramName.includes('batch_size')) {
    const labels = ['8', '16', '32', '64', '128', '256'];
    
    visualization = {
      type: 'line',
      title: 'Batch Size Impact on Training',
      xLabel: 'Batch Size',
      yLabel: 'Metric Value',
      data: {
        labels,
        datasets: [
          {
            label: 'Training Time',
            data: [1.0, 0.8, 0.6, 0.4, 0.3, 0.25],
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.2,
            yAxisID: 'y'
          },
          {
            label: 'Memory Usage',
            data: [0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
            borderColor: 'rgba(54, 162, 235, 1)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            tension: 0.2,
            yAxisID: 'y'
          }
        ]
      }
    };
  }
  else if (paramName.includes('dropout')) {
    const labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'];
    
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
            data: [0.99, 0.97, 0.95, 0.92, 0.88, 0.84, 0.78, 0.72],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Accuracy',
            data: [0.82, 0.86, 0.89, 0.90, 0.88, 0.85, 0.80, 0.72],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          }
        ]
      }
    };
  }
  else if (paramName.includes('epoch') || paramName.includes('num_epochs') || paramName.includes('n_epochs')) {
    const labels = ['1', '5', '10', '20', '50', '100'];
    
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
            data: [0.70, 0.85, 0.92, 0.97, 0.99, 0.995],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Accuracy',
            data: [0.68, 0.82, 0.86, 0.87, 0.86, 0.85],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          }
        ]
      }
    };
  }
  else if (paramName.includes('weight_decay') || paramName.includes('l2') || paramName.includes('regularization')) {
    const labels = ['0.0', '0.0001', '0.001', '0.01', '0.1'];
    
    visualization = {
      type: 'line',
      title: 'Weight Decay (L2 Regularization) Impact',
      xLabel: 'Weight Decay Factor',
      yLabel: 'Loss',
      data: {
        labels,
        datasets: [
          {
            label: 'Training Loss',
            data: [0.05, 0.07, 0.10, 0.15, 0.25],
            borderColor: 'rgba(255, 99, 132, 1)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.4
          },
          {
            label: 'Validation Loss',
            data: [0.20, 0.15, 0.12, 0.14, 0.22],
            borderColor: 'rgba(54, 162, 235, 1)',
            backgroundColor: 'rgba(54, 162, 235, 0.2)',
            tension: 0.4
          }
        ]
      }
    };
  }
  else if (paramName.includes('optimizer')) {
    visualization = {
      type: 'bar',
      title: 'Optimizer Performance Comparison',
      xLabel: 'Optimizer',
      yLabel: 'Relative Performance',
      data: {
        labels: ['SGD', 'Adam', 'RMSprop', 'AdaGrad', 'AdamW'],
        datasets: [
          {
            label: 'Convergence Speed',
            data: [0.5, 0.9, 0.7, 0.6, 0.85],
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
          },
          {
            label: 'Final Accuracy',
            data: [0.88, 0.92, 0.90, 0.87, 0.93],
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1
          }
        ]
      }
    };
  }
  else if (paramName.includes('momentum')) {
    const labels = ['0.0', '0.3', '0.6', '0.9', '0.99'];
    
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
            data: [0.3, 0.5, 0.7, 0.9, 0.85],
            borderColor: 'rgba(255, 159, 64, 1)',
            backgroundColor: 'rgba(255, 159, 64, 0.2)',
            tension: 0.4
          },
          {
            label: 'Stability',
            data: [0.9, 0.85, 0.8, 0.7, 0.6],
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            tension: 0.4
          }
        ]
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
          }
        ]
      }
    };
  }
  
  return visualization;
}