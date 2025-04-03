import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { NetworkVisualization } from './NetworkVisualization';
import { HyperparameterVisualizations } from './HyperparameterVisualizations';
import { BenchmarkComparison } from './BenchmarkComparison';
import { Badge } from '@/components/ui/badge';
import { BadgeCustom } from '@/components/ui/badge-custom';
import { Info, ArrowRight, Maximize2, BarChart3, TrendingUp, Layers } from 'lucide-react';

export function HyperparameterPlayground() {
  const [activeTab, setActiveTab] = useState('learning-rate');
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(32);
  const [dropout, setDropout] = useState(0.5);
  const [epochs, setEpochs] = useState(10);
  const [optimizer, setOptimizer] = useState('adam');
  const [activeParameter, setActiveParameter] = useState({
    name: 'learning_rate',
    value: '0.001',
    framework: 'tensorflow'
  });

  // Update active parameter when tab or values change
  useEffect(() => {
    switch(activeTab) {
      case 'learning-rate':
        setActiveParameter({
          name: 'learning_rate',
          value: learningRate.toString(),
          framework: 'tensorflow'
        });
        break;
      case 'batch-size':
        setActiveParameter({
          name: 'batch_size',
          value: batchSize.toString(),
          framework: 'tensorflow'
        });
        break;
      case 'dropout':
        setActiveParameter({
          name: 'dropout',
          value: dropout.toString(),
          framework: 'tensorflow'
        });
        break;
      case 'epochs':
        setActiveParameter({
          name: 'epochs',
          value: epochs.toString(),
          framework: 'tensorflow'
        });
        break;
      case 'optimizer':
        setActiveParameter({
          name: 'optimizer',
          value: optimizer,
          framework: 'tensorflow'
        });
        break;
      default:
        break;
    }
  }, [activeTab, learningRate, batchSize, dropout, epochs, optimizer]);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Maximize2 className="h-5 w-5 text-primary" />
          Interactive Hyperparameter Playground
        </CardTitle>
        <CardDescription>
          Experiment with different hyperparameter values and see their effects on neural networks in real-time
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="learning-rate" value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="mb-4 w-full justify-start overflow-x-auto">
            <TabsTrigger value="learning-rate" className="flex items-center gap-1">
              <TrendingUp className="h-4 w-4" />
              Learning Rate
            </TabsTrigger>
            <TabsTrigger value="batch-size" className="flex items-center gap-1">
              <BarChart3 className="h-4 w-4" />
              Batch Size
            </TabsTrigger>
            <TabsTrigger value="dropout" className="flex items-center gap-1">
              <Layers className="h-4 w-4" />
              Dropout
            </TabsTrigger>
            <TabsTrigger value="epochs" className="flex items-center gap-1">
              <ArrowRight className="h-4 w-4" />
              Epochs
            </TabsTrigger>
            <TabsTrigger value="optimizer" className="flex items-center gap-1">
              <Info className="h-4 w-4" />
              Optimizer
            </TabsTrigger>
          </TabsList>

          <TabsContent value="learning-rate">
            <div className="space-y-4">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between">
                  <Label htmlFor="learning-rate-slider" className="text-sm font-medium">
                    Learning Rate: <span className="font-mono">{learningRate.toFixed(6)}</span>
                  </Label>
                  <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">High Impact</Badge>
                </div>
                <Slider
                  id="learning-rate-slider"
                  min={0.000001}
                  max={0.1}
                  step={0.000001}
                  value={[learningRate]}
                  onValueChange={(value) => setLearningRate(value[0])}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Slow Learning (0.000001)</span>
                  <span>Fast Learning (0.1)</span>
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  The learning rate controls how much to adjust the model weights during training. Higher values lead to faster learning but may overshoot optimal values.
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <BadgeCustom type="lower" onClick={() => setLearningRate(0.0001)}>0.0001</BadgeCustom>
                  <BadgeCustom type="higher" onClick={() => setLearningRate(0.01)}>0.01</BadgeCustom>
                  <BadgeCustom type="advanced" onClick={() => setLearningRate(0.001)}>0.001</BadgeCustom>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="batch-size">
            <div className="space-y-4">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between">
                  <Label htmlFor="batch-size-slider" className="text-sm font-medium">
                    Batch Size: <span className="font-mono">{batchSize}</span>
                  </Label>
                  <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
                </div>
                <Slider
                  id="batch-size-slider"
                  min={1}
                  max={256}
                  step={1}
                  value={[batchSize]}
                  onValueChange={(value) => setBatchSize(value[0])}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Small Batches (1)</span>
                  <span>Large Batches (256)</span>
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  The batch size determines how many training examples are processed together before updating model weights. Larger batches are more efficient but may lead to less optimal convergence.
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <BadgeCustom type="lower" onClick={() => setBatchSize(8)}>8</BadgeCustom>
                  <BadgeCustom type="higher" onClick={() => setBatchSize(128)}>128</BadgeCustom>
                  <BadgeCustom type="advanced" onClick={() => setBatchSize(32)}>32</BadgeCustom>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="dropout">
            <div className="space-y-4">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between">
                  <Label htmlFor="dropout-slider" className="text-sm font-medium">
                    Dropout Rate: <span className="font-mono">{dropout.toFixed(2)}</span>
                  </Label>
                  <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
                </div>
                <Slider
                  id="dropout-slider"
                  min={0}
                  max={0.9}
                  step={0.01}
                  value={[dropout]}
                  onValueChange={(value) => setDropout(value[0])}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>No Dropout (0.0)</span>
                  <span>High Dropout (0.9)</span>
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  Dropout randomly deactivates a percentage of neurons during training to prevent overfitting. Higher dropout rates enhance regularization but may require more training epochs.
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <BadgeCustom type="lower" onClick={() => setDropout(0.2)}>0.2</BadgeCustom>
                  <BadgeCustom type="higher" onClick={() => setDropout(0.7)}>0.7</BadgeCustom>
                  <BadgeCustom type="advanced" onClick={() => setDropout(0.5)}>0.5</BadgeCustom>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="epochs">
            <div className="space-y-4">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between">
                  <Label htmlFor="epochs-slider" className="text-sm font-medium">
                    Epochs: <span className="font-mono">{epochs}</span>
                  </Label>
                  <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
                </div>
                <Slider
                  id="epochs-slider"
                  min={1}
                  max={100}
                  step={1}
                  value={[epochs]}
                  onValueChange={(value) => setEpochs(value[0])}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Few Epochs (1)</span>
                  <span>Many Epochs (100)</span>
                </div>
                <div className="mt-2 text-sm text-gray-600">
                  An epoch represents one complete pass through the entire training dataset. More epochs generally lead to better learning, but too many can cause overfitting.
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <BadgeCustom type="lower" onClick={() => setEpochs(5)}>5</BadgeCustom>
                  <BadgeCustom type="higher" onClick={() => setEpochs(50)}>50</BadgeCustom>
                  <BadgeCustom type="advanced" onClick={() => setEpochs(20)}>20</BadgeCustom>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="optimizer">
            <div className="space-y-4">
              <div className="flex flex-col gap-2">
                <div className="flex justify-between">
                  <Label className="text-sm font-medium">
                    Optimizer: <span className="font-mono capitalize">{optimizer}</span>
                  </Label>
                  <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">High Impact</Badge>
                </div>
                <Select value={optimizer} onValueChange={setOptimizer}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select optimizer" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="sgd">SGD (Stochastic Gradient Descent)</SelectItem>
                    <SelectItem value="adam">Adam</SelectItem>
                    <SelectItem value="rmsprop">RMSprop</SelectItem>
                    <SelectItem value="adagrad">Adagrad</SelectItem>
                  </SelectContent>
                </Select>
                <div className="mt-2 text-sm text-gray-600">
                  The optimizer determines the specific algorithm used to update model weights during training. Each has different characteristics and performance profiles for various tasks.
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  <BadgeCustom type="lower" onClick={() => setOptimizer('sgd')}>SGD</BadgeCustom>
                  <BadgeCustom type="higher" onClick={() => setOptimizer('rmsprop')}>RMSprop</BadgeCustom>
                  <BadgeCustom type="advanced" onClick={() => setOptimizer('adam')}>Adam</BadgeCustom>
                </div>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
          <NetworkVisualization 
            paramName={activeParameter.name}
            paramValue={activeParameter.value}
            framework={activeParameter.framework}
          />
          
          <HyperparameterVisualizations
            paramName={activeParameter.name}
            paramValue={activeParameter.value}
            framework={activeParameter.framework}
          />
        </div>
        
        <div className="mt-6">
          <BenchmarkComparison
            paramName={activeParameter.name}
            paramValue={activeParameter.value}
            framework={activeParameter.framework}
          />
        </div>
      </CardContent>
    </Card>
  );
}