import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import Chart from 'chart.js/auto';
import { Separator } from '@/components/ui/separator';
import { Info, RefreshCw } from 'lucide-react';

interface BenchmarkComparisonProps {
  paramName: string;
  paramValue: string;
  framework?: string;
}

// Define datasets
const DATASETS = [
  { id: 'mnist', name: 'MNIST', description: 'Hand-written digits', type: 'image', size: '70,000 images' },
  { id: 'cifar10', name: 'CIFAR-10', description: 'Object recognition', type: 'image', size: '60,000 images' },
  { id: 'imdb', name: 'IMDB Reviews', description: 'Sentiment analysis', type: 'text', size: '50,000 reviews' },
  { id: 'boston', name: 'Boston Housing', description: 'Price regression', type: 'tabular', size: '506 samples' },
];

// Define metrics
const METRICS = [
  { id: 'accuracy', name: 'Accuracy', description: 'Percentage of correct predictions', higherIsBetter: true },
  { id: 'loss', name: 'Loss', description: 'Model error rate', higherIsBetter: false },
  { id: 'f1', name: 'F1 Score', description: 'Harmonic mean of precision and recall', higherIsBetter: true },
  { id: 'train_time', name: 'Training Time', description: 'Time required to train (seconds)', higherIsBetter: false },
];

// Helper function to generate benchmark results based on hyperparameter
const generateBenchmarkResults = (paramName: string, paramValue: any, dataset: string) => {
  const baseAccuracy = {
    'mnist': 0.975,
    'cifar10': 0.82,
    'imdb': 0.88,
    'boston': 0.77, // R²
  }[dataset] || 0.85;
  
  const baseLoss = {
    'mnist': 0.08,
    'cifar10': 0.65,
    'imdb': 0.31,
    'boston': 10.5,
  }[dataset] || 0.25;
  
  const baseF1 = {
    'mnist': 0.973,
    'cifar10': 0.81,
    'imdb': 0.87,
    'boston': 0.76,
  }[dataset] || 0.84;
  
  const baseTrainTime = {
    'mnist': 45,
    'cifar10': 210,
    'imdb': 85,
    'boston': 8,
  }[dataset] || 60;
  
  // Generate variations based on hyperparameter type and value
  let accuracyMod = 0;
  let lossMod = 0;
  let f1Mod = 0;
  let trainTimeMod = 1;
  
  const value = parseFloat(paramValue);
  
  if (paramName.includes('learning_rate') || paramName.includes('lr')) {
    // For learning rate, optimal is usually around 0.001-0.01
    if (value < 0.0001) {
      // Too small, slow convergence
      accuracyMod = -0.05;
      lossMod = 0.1;
      f1Mod = -0.05;
      trainTimeMod = 2.5;
    } else if (value < 0.001) {
      // Still on the small side
      accuracyMod = -0.01;
      lossMod = 0.03;
      f1Mod = -0.01;
      trainTimeMod = 1.5;
    } else if (value <= 0.01) {
      // Sweet spot
      accuracyMod = 0.01;
      lossMod = -0.05;
      f1Mod = 0.01;
      trainTimeMod = 1;
    } else if (value <= 0.1) {
      // Getting high, potential overshoot
      accuracyMod = -0.03;
      lossMod = 0.1;
      f1Mod = -0.02;
      trainTimeMod = 0.9;
    } else {
      // Too high, unstable training
      accuracyMod = -0.15;
      lossMod = 0.4;
      f1Mod = -0.15;
      trainTimeMod = 1.2;
    }
  } 
  else if (paramName.includes('batch_size')) {
    // For batch size, effects depend on dataset
    if (value <= 16) {
      // Small batch size
      accuracyMod = 0.01;
      lossMod = -0.02;
      f1Mod = 0.01;
      trainTimeMod = 1.8;
    } else if (value <= 64) {
      // Medium batch size - often optimal
      accuracyMod = 0.02;
      lossMod = -0.05;
      f1Mod = 0.02;
      trainTimeMod = 1;
    } else if (value <= 128) {
      // Large batch size
      accuracyMod = 0;
      lossMod = -0.02;
      f1Mod = 0;
      trainTimeMod = 0.8;
    } else {
      // Very large batch size
      accuracyMod = -0.03;
      lossMod = 0.05;
      f1Mod = -0.03;
      trainTimeMod = 0.7;
    }
  }
  else if (paramName.includes('dropout')) {
    // For dropout, optimal is usually 0.2-0.5
    if (value === 0) {
      // No dropout
      accuracyMod = 0.02;
      lossMod = -0.05;
      f1Mod = 0.02;
      trainTimeMod = 0.95;
      
      // But worse generalization (lower test accuracy)
      if (dataset !== 'boston') { // For small datasets like Boston, dropout sometimes hurts
        accuracyMod = -0.03;
        f1Mod = -0.03;
      }
    } else if (value <= 0.2) {
      // Light dropout
      accuracyMod = 0.03;
      lossMod = -0.05;
      f1Mod = 0.03;
      trainTimeMod = 0.98;
    } else if (value <= 0.5) {
      // Medium dropout - often optimal
      accuracyMod = 0.025;
      lossMod = -0.02;
      f1Mod = 0.025;
      trainTimeMod = 1;
    } else if (value <= 0.7) {
      // Heavy dropout
      accuracyMod = -0.01;
      lossMod = 0.04;
      f1Mod = -0.01;
      trainTimeMod = 1.05;
    } else {
      // Extreme dropout
      accuracyMod = -0.1;
      lossMod = 0.2;
      f1Mod = -0.1;
      trainTimeMod = 1.15;
    }
  }
  else if (paramName.includes('epoch')) {
    // For epoch count
    const logEpochs = Math.log10(Math.max(1, value));
    accuracyMod = Math.min(0.1, logEpochs * 0.03);
    lossMod = Math.min(-0.15, -logEpochs * 0.04);
    f1Mod = Math.min(0.1, logEpochs * 0.03);
    trainTimeMod = Math.max(0.1, value / 50);
  }
  
  // Add some variability based on datasets
  if (dataset === 'cifar10') {
    // CIFAR-10 is more sensitive to hyperparameters
    accuracyMod *= 1.5;
    lossMod *= 1.5;
    f1Mod *= 1.5;
  } else if (dataset === 'boston') {
    // Boston is less sensitive (small dataset)
    accuracyMod *= 0.8;
    lossMod *= 0.8;
    f1Mod *= 0.8;
  }
  
  // Generate actual values
  const accuracy = Math.min(1, Math.max(0, baseAccuracy + accuracyMod));
  const loss = Math.max(0, baseLoss + lossMod);
  const f1 = Math.min(1, Math.max(0, baseF1 + f1Mod));
  const trainTime = Math.max(1, baseTrainTime * trainTimeMod);
  
  // Add slight random variations
  const randomFactor = 0.005;
  return {
    accuracy: accuracy * (1 + (Math.random() - 0.5) * randomFactor),
    loss: loss * (1 + (Math.random() - 0.5) * randomFactor),
    f1: f1 * (1 + (Math.random() - 0.5) * randomFactor),
    train_time: trainTime * (1 + (Math.random() - 0.5) * randomFactor)
  };
};

export function BenchmarkComparison({ 
  paramName, 
  paramValue,
  framework 
}: BenchmarkComparisonProps) {
  const [selectedDataset, setSelectedDataset] = useState('mnist');
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [isLoading, setIsLoading] = useState(false);
  const [benchmarkData, setBenchmarkData] = useState<any>(null);
  const [chartInstance, setChartInstance] = useState<Chart | null>(null);
  
  const chartRef = React.useRef<HTMLCanvasElement>(null);
  
  // Get comparison values based on the current parameter
  const getComparisonValues = () => {
    switch(paramName) {
      case 'learning_rate':
      case 'lr':
        return ['0.0001', '0.001', '0.01', '0.1'];
      case 'batch_size':
        return ['16', '32', '64', '128', '256'];
      case 'dropout':
        return ['0', '0.2', '0.5', '0.7'];
      case 'epochs':
        return ['10', '50', '100', '200'];
      default:
        // Generate some reasonable comparison values
        const numValue = parseFloat(paramValue) || 0.5;
        const minVal = Math.max(0, numValue * 0.5);
        const maxVal = numValue * 2;
        const step = (maxVal - minVal) / 3;
        return [
          minVal.toFixed(4),
          (minVal + step).toFixed(4),
          (minVal + 2 * step).toFixed(4),
          maxVal.toFixed(4)
        ];
    }
  };
  
  // Get benchmark data
  const runBenchmark = () => {
    setIsLoading(true);
    
    // In a real app, this would be an API call to a benchmark service
    // Here we'll simulate it with setTimeout and our generation function
    setTimeout(() => {
      const comparisonValues = [paramValue, ...getComparisonValues().filter(v => v !== paramValue)];
      
      const results: { [value: string]: any } = {};
      
      // Generate results for each value
      comparisonValues.forEach(value => {
        results[value] = generateBenchmarkResults(paramName, value, selectedDataset);
      });
      
      setBenchmarkData(results);
      setIsLoading(false);
    }, 1500);
  };
  
  // Run benchmark initially and when dataset changes
  useEffect(() => {
    runBenchmark();
  }, [selectedDataset, paramValue]);
  
  // Update chart when data or metric changes
  useEffect(() => {
    if (!benchmarkData || !chartRef.current) return;
    
    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;
    
    // Destroy previous chart if it exists
    if (chartInstance) {
      chartInstance.destroy();
    }
    
    // Prepare chart data
    const values = Object.keys(benchmarkData);
    const metricValues = values.map(val => benchmarkData[val][selectedMetric]);
    
    // Determine if higher values are better for this metric
    const metric = METRICS.find(m => m.id === selectedMetric);
    const higherIsBetter = metric?.higherIsBetter ?? true;
    
    // Highlight the current value
    const backgroundColor = values.map(val => 
      val === paramValue ? 'rgba(79, 70, 229, 0.8)' : 'rgba(100, 116, 139, 0.5)'
    );
    const borderColor = values.map(val => 
      val === paramValue ? 'rgba(79, 70, 229, 1)' : 'rgba(100, 116, 139, 0.8)'
    );
    
    // Create chart
    const newChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: values,
        datasets: [{
          label: metric?.name || selectedMetric,
          data: metricValues,
          backgroundColor: backgroundColor,
          borderColor: borderColor,
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: false,
            // Set suggestedMin and suggestedMax based on data range
            suggestedMin: Math.min(...metricValues) * 0.95,
            suggestedMax: Math.max(...metricValues) * 1.05,
            title: {
              display: true,
              text: metric?.name || selectedMetric
            }
          },
          x: {
            title: {
              display: true,
              text: paramName
            }
          }
        },
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            callbacks: {
              afterLabel: function(context) {
                const value = context.parsed.y;
                if (selectedMetric === 'accuracy' || selectedMetric === 'f1') {
                  return `${(value * 100).toFixed(2)}%`;
                } else if (selectedMetric === 'train_time') {
                  return `${value.toFixed(1)} seconds`;
                }
                return value.toFixed(4);
              }
            }
          }
        }
      }
    });
    
    setChartInstance(newChart);
    
    // Cleanup
    return () => {
      newChart.destroy();
    };
  }, [benchmarkData, selectedMetric]);
  
  const selectedDatasetInfo = DATASETS.find(d => d.id === selectedDataset);
  const selectedMetricInfo = METRICS.find(m => m.id === selectedMetric);
  
  return (
    <Card className="w-full mt-6">
      <CardHeader>
        <CardTitle className="text-lg">Benchmark Comparison</CardTitle>
        <CardDescription>
          Compare {paramName} values across common datasets
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col space-y-4 sm:flex-row sm:space-y-0 sm:space-x-4">
          <div className="w-full sm:w-1/2">
            <label className="text-sm font-medium mb-1 block">Dataset</label>
            <Select value={selectedDataset} onValueChange={setSelectedDataset}>
              <SelectTrigger>
                <SelectValue placeholder="Select dataset" />
              </SelectTrigger>
              <SelectContent>
                {DATASETS.map(dataset => (
                  <SelectItem key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="w-full sm:w-1/2">
            <label className="text-sm font-medium mb-1 block">Metric</label>
            <Select value={selectedMetric} onValueChange={setSelectedMetric}>
              <SelectTrigger>
                <SelectValue placeholder="Select metric" />
              </SelectTrigger>
              <SelectContent>
                {METRICS.map(metric => (
                  <SelectItem key={metric.id} value={metric.id}>
                    {metric.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        
        <div className="py-2 flex items-center space-x-2">
          <Info className="h-4 w-4 text-blue-500" />
          <div className="text-xs text-gray-500">
            <span className="font-medium">{selectedDatasetInfo?.name}:</span> {selectedDatasetInfo?.description} ({selectedDatasetInfo?.size})
          </div>
          
          <Separator orientation="vertical" className="h-4" />
          
          <div className="text-xs text-gray-500">
            <span className="font-medium">{selectedMetricInfo?.name}:</span> {selectedMetricInfo?.higherIsBetter ? 'Higher is better' : 'Lower is better'}
          </div>
        </div>
        
        <div className="h-60 relative">
          {isLoading ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <RefreshCw className="h-8 w-8 animate-spin text-blue-500" />
              <span className="ml-2 text-sm text-gray-500">Running benchmark...</span>
            </div>
          ) : (
            <canvas ref={chartRef} />
          )}
        </div>
        
        <div className="pt-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={runBenchmark}
            disabled={isLoading}
            className="w-full sm:w-auto"
          >
            {isLoading ? 'Running...' : 'Run Benchmark Again'}
          </Button>
          
          <div className="mt-4 text-xs text-gray-500">
            <p>Note: This is a simulation based on typical performance patterns for different hyperparameter values.</p>
            <p>For real-world applications, you should benchmark on your specific datasets and models.</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}