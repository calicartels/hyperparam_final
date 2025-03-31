import React, { useEffect, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import Chart from 'chart.js/auto';

interface HyperparameterVisualizationProps {
  paramName: string;
  paramValue: string;
  framework?: string;
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
    
    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;
    
    // Generate appropriate data based on parameter type
    let labels: string[] = [];
    let datasets: any[] = [];
    let title = `${paramName} Impact`;
    let xLabel = `${paramName} Value`;
    let yLabel = 'Performance';
    let chartType: 'line' | 'bar' = 'line';
    
    if (paramName.includes('learning_rate') || paramName.includes('lr')) {
      labels = ['0.0001', '0.001', '0.01', '0.1', '1.0'];
      datasets = [
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
      ];
      title = 'Learning Rate Impact';
      xLabel = 'Learning Rate (log scale)';
      yLabel = 'Accuracy';
    } 
    else if (paramName.includes('batch_size')) {
      labels = ['8', '16', '32', '64', '128', '256'];
      datasets = [
        {
          label: 'Training Time',
          data: [1.0, 0.8, 0.6, 0.4, 0.3, 0.25],
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          tension: 0.2
        },
        {
          label: 'Memory Usage',
          data: [0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
          borderColor: 'rgba(54, 162, 235, 1)',
          backgroundColor: 'rgba(54, 162, 235, 0.2)',
          tension: 0.2
        }
      ];
      title = 'Batch Size Impact';
      xLabel = 'Batch Size';
    }
    else if (paramName.includes('dropout')) {
      labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'];
      datasets = [
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
      ];
      title = 'Dropout Rate Impact';
      xLabel = 'Dropout Rate';
      yLabel = 'Accuracy';
    }
    else if (paramName.includes('optimizer')) {
      labels = ['SGD', 'Adam', 'RMSprop', 'AdaGrad', 'AdamW'];
      datasets = [
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
      ];
      chartType = 'bar';
      title = 'Optimizer Comparison';
      xLabel = 'Optimizer';
      yLabel = 'Relative Performance';
    }
    else {
      // Generic visualization for any other parameter
      const numValue = parseFloat(paramValue) || 0.5;
      const min = Math.max(0, numValue * 0.5);
      const max = numValue * 1.5;
      const step = (max - min) / 4;
      
      for (let i = 0; i <= 4; i++) {
        labels.push((min + i * step).toFixed(4));
      }
      
      // Generic performance curve (bell curve around optimal value)
      const performanceData = labels.map((val, index) => {
        const distance = Math.abs(parseFloat(val) - numValue);
        return 0.9 - 0.2 * (distance / (max - min)) ** 2;
      });
      
      // Generic cost curve (increasing with parameter value)
      const costData = labels.map((val) => {
        const normalizedVal = (parseFloat(val) - min) / (max - min);
        return 0.3 + 0.5 * normalizedVal;
      });
      
      datasets = [
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
      ];
    }

    // Create chart
    chartInstanceRef.current = new Chart(ctx, {
      type: chartType,
      data: {
        labels,
        datasets
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: 'top',
          },
          title: {
            display: true,
            text: title
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: xLabel
            }
          },
          y: {
            title: {
              display: true,
              text: yLabel
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
          <p>This visualization shows how different values for <strong>{paramName}</strong> may affect model performance.
             The optimal range depends on your specific dataset and problem.</p>
        </div>
      </CardContent>
    </Card>
  );
}