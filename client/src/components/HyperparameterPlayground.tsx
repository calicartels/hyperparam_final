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
import { 
  Info, ArrowRight, Maximize2, BarChart3, TrendingUp, Layers, 
  Code, Cpu, Zap, FunctionSquare, Activity, Box, Database
} from 'lucide-react';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';

// Parameter Category Types
type ParameterCategory = 
  | 'hyperparameters' 
  | 'architecture' 
  | 'layer-config' 
  | 'activation' 
  | 'loss' 
  | 'regularization';

export function HyperparameterPlayground() {
  // Current active category and tab
  const [activeCategory, setActiveCategory] = useState<ParameterCategory>('hyperparameters');
  const [activeTab, setActiveTab] = useState('learning-rate');
  
  // Traditional Hyperparameters
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(32);
  const [dropout, setDropout] = useState(0.5);
  const [epochs, setEpochs] = useState(10);
  const [optimizer, setOptimizer] = useState('adam');
  
  // Model Architecture Choices
  const [modelType, setModelType] = useState('sequential');
  const [hiddenLayers, setHiddenLayers] = useState(3);
  const [hiddenUnits, setHiddenUnits] = useState(128);
  
  // Layer Configuration
  const [kernelSize, setKernelSize] = useState(3);
  const [filters, setFilters] = useState(32);
  const [padding, setPadding] = useState('same');
  
  // Activation Functions
  const [activationFunc, setActivationFunc] = useState('relu');
  const [outputActivation, setOutputActivation] = useState('softmax');
  
  // Loss Functions
  const [lossFunction, setLossFunction] = useState('categorical_crossentropy');
  
  // Regularization
  const [l1Reg, setL1Reg] = useState(0.0);
  const [l2Reg, setL2Reg] = useState(0.01);
  const [batchNorm, setBatchNorm] = useState(true);
  
  // Track the active parameter to display in visualizations
  const [activeParameter, setActiveParameter] = useState({
    name: 'learning_rate',
    value: '0.001',
    framework: 'tensorflow'
  });

  // Update active parameter when category, tab or values change
  useEffect(() => {
    // Set the active parameter based on current category and tab
    switch(activeCategory) {
      case 'hyperparameters':
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
        }
        break;
        
      case 'architecture':
        switch(activeTab) {
          case 'model-type':
            setActiveParameter({
              name: 'model_architecture',
              value: modelType,
              framework: 'tensorflow'
            });
            break;
          case 'hidden-layers':
            setActiveParameter({
              name: 'num_hidden_layers',
              value: hiddenLayers.toString(),
              framework: 'tensorflow'
            });
            break;
          case 'hidden-units':
            setActiveParameter({
              name: 'hidden_units',
              value: hiddenUnits.toString(),
              framework: 'tensorflow'
            });
            break;
        }
        break;
        
      case 'layer-config':
        switch(activeTab) {
          case 'kernel-size':
            setActiveParameter({
              name: 'kernel_size',
              value: kernelSize.toString(),
              framework: 'tensorflow'
            });
            break;
          case 'filters':
            setActiveParameter({
              name: 'conv_filters',
              value: filters.toString(),
              framework: 'tensorflow'
            });
            break;
          case 'padding':
            setActiveParameter({
              name: 'padding',
              value: padding,
              framework: 'tensorflow'
            });
            break;
        }
        break;
        
      case 'activation':
        switch(activeTab) {
          case 'activation-function':
            setActiveParameter({
              name: 'activation_function',
              value: activationFunc,
              framework: 'tensorflow'
            });
            break;
          case 'output-activation':
            setActiveParameter({
              name: 'output_activation',
              value: outputActivation,
              framework: 'tensorflow'
            });
            break;
        }
        break;
        
      case 'loss':
        setActiveParameter({
          name: 'loss_function',
          value: lossFunction,
          framework: 'tensorflow'
        });
        break;
        
      case 'regularization':
        switch(activeTab) {
          case 'l1-regularization':
            setActiveParameter({
              name: 'l1_regularization',
              value: l1Reg.toString(),
              framework: 'tensorflow'
            });
            break;
          case 'l2-regularization':
            setActiveParameter({
              name: 'l2_regularization',
              value: l2Reg.toString(),
              framework: 'tensorflow'
            });
            break;
          case 'batch-normalization':
            setActiveParameter({
              name: 'batch_normalization',
              value: batchNorm ? 'True' : 'False',
              framework: 'tensorflow'
            });
            break;
        }
        break;
    }
  }, [
    activeCategory, activeTab, 
    learningRate, batchSize, dropout, epochs, optimizer,
    modelType, hiddenLayers, hiddenUnits,
    kernelSize, filters, padding,
    activationFunc, outputActivation,
    lossFunction,
    l1Reg, l2Reg, batchNorm
  ]);

  // Handle category change
  const handleCategoryChange = (category: ParameterCategory) => {
    setActiveCategory(category);
    
    // Set default tab for each category
    switch(category) {
      case 'hyperparameters':
        setActiveTab('learning-rate');
        break;
      case 'architecture':
        setActiveTab('model-type');
        break;
      case 'layer-config':
        setActiveTab('kernel-size');
        break;
      case 'activation':
        setActiveTab('activation-function');
        break;
      case 'loss':
        setActiveTab('loss-function');
        break;
      case 'regularization':
        setActiveTab('l2-regularization');
        break;
    }
  };

  // Render the tab content based on the active category
  const renderTabContent = () => {
    switch(activeCategory) {
      case 'hyperparameters':
        return renderHyperparameterTabs();
      case 'architecture':
        return renderArchitectureTabs();
      case 'layer-config':
        return renderLayerConfigTabs();
      case 'activation':
        return renderActivationTabs();
      case 'loss':
        return renderLossTabs();
      case 'regularization':
        return renderRegularizationTabs();
      default:
        return null;
    }
  };

  // Traditional Hyperparameter Tabs
  const renderHyperparameterTabs = () => (
    <>
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
    </>
  );

  // Architecture Tabs
  const renderArchitectureTabs = () => (
    <>
      <TabsList className="mb-4 w-full justify-start overflow-x-auto">
        <TabsTrigger value="model-type" className="flex items-center gap-1">
          <Box className="h-4 w-4" />
          Model Type
        </TabsTrigger>
        <TabsTrigger value="hidden-layers" className="flex items-center gap-1">
          <Layers className="h-4 w-4" />
          Hidden Layers
        </TabsTrigger>
        <TabsTrigger value="hidden-units" className="flex items-center gap-1">
          <Cpu className="h-4 w-4" />
          Hidden Units
        </TabsTrigger>
      </TabsList>

      <TabsContent value="model-type">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label className="text-sm font-medium">
                Model Architecture: <span className="font-mono capitalize">{modelType}</span>
              </Label>
              <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">High Impact</Badge>
            </div>
            <Select value={modelType} onValueChange={setModelType}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select model architecture" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="sequential">Sequential</SelectItem>
                <SelectItem value="functional">Functional API</SelectItem>
                <SelectItem value="subclassing">Model Subclassing</SelectItem>
              </SelectContent>
            </Select>
            <div className="mt-2 text-sm text-gray-600">
              The model architecture defines the overall structure and organization of your neural network. Different architectures allow for different levels of flexibility and complexity.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="lower" onClick={() => setModelType('sequential')}>Sequential</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setModelType('functional')}>Functional API</BadgeCustom>
              <BadgeCustom type="extreme" onClick={() => setModelType('subclassing')}>Model Subclassing</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="hidden-layers">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label htmlFor="hidden-layers-slider" className="text-sm font-medium">
                Number of Hidden Layers: <span className="font-mono">{hiddenLayers}</span>
              </Label>
              <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">High Impact</Badge>
            </div>
            <Slider
              id="hidden-layers-slider"
              min={1}
              max={20}
              step={1}
              value={[hiddenLayers]}
              onValueChange={(value) => setHiddenLayers(value[0])}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Shallow (1)</span>
              <span>Deep (20)</span>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              The number of hidden layers determines the depth of your neural network. Deeper networks can learn more complex patterns but require more data and computational resources.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="lower" onClick={() => setHiddenLayers(2)}>2</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setHiddenLayers(8)}>8</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setHiddenLayers(4)}>4</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="hidden-units">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label htmlFor="hidden-units-slider" className="text-sm font-medium">
                Hidden Units Per Layer: <span className="font-mono">{hiddenUnits}</span>
              </Label>
              <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">High Impact</Badge>
            </div>
            <Slider
              id="hidden-units-slider"
              min={8}
              max={1024}
              step={8}
              value={[hiddenUnits]}
              onValueChange={(value) => setHiddenUnits(value[0])}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Few (8)</span>
              <span>Many (1024)</span>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              The number of hidden units per layer determines the width of your neural network. More units increase the model's capacity to learn complex patterns but require more memory and computation.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="lower" onClick={() => setHiddenUnits(64)}>64</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setHiddenUnits(512)}>512</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setHiddenUnits(256)}>256</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>
    </>
  );

  // Layer Configuration Tabs
  const renderLayerConfigTabs = () => (
    <>
      <TabsList className="mb-4 w-full justify-start overflow-x-auto">
        <TabsTrigger value="kernel-size" className="flex items-center gap-1">
          <Box className="h-4 w-4" />
          Kernel Size
        </TabsTrigger>
        <TabsTrigger value="filters" className="flex items-center gap-1">
          <Layers className="h-4 w-4" />
          Filters
        </TabsTrigger>
        <TabsTrigger value="padding" className="flex items-center gap-1">
          <Maximize2 className="h-4 w-4" />
          Padding
        </TabsTrigger>
      </TabsList>

      <TabsContent value="kernel-size">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label htmlFor="kernel-size-slider" className="text-sm font-medium">
                Kernel Size: <span className="font-mono">{kernelSize}x{kernelSize}</span>
              </Label>
              <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
            </div>
            <Slider
              id="kernel-size-slider"
              min={1}
              max={11}
              step={2}
              value={[kernelSize]}
              onValueChange={(value) => setKernelSize(value[0])}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Small (1x1)</span>
              <span>Large (11x11)</span>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              The kernel size defines the area of focus (receptive field) for convolutional operations. Larger kernels capture more spatial context but require more computation and may lead to overfitting.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="lower" onClick={() => setKernelSize(1)}>1x1</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setKernelSize(5)}>5x5</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setKernelSize(3)}>3x3</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="filters">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label htmlFor="filters-slider" className="text-sm font-medium">
                Number of Filters: <span className="font-mono">{filters}</span>
              </Label>
              <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
            </div>
            <Slider
              id="filters-slider"
              min={1}
              max={256}
              step={1}
              value={[filters]}
              onValueChange={(value) => setFilters(value[0])}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Few (1)</span>
              <span>Many (256)</span>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              The number of filters in a convolutional layer determines how many different features the layer can detect. More filters increase the model's capacity to recognize complex patterns.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="lower" onClick={() => setFilters(16)}>16</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setFilters(64)}>64</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setFilters(32)}>32</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="padding">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label className="text-sm font-medium">
                Padding Type: <span className="font-mono capitalize">{padding}</span>
              </Label>
              <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
            </div>
            <Select value={padding} onValueChange={setPadding}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select padding type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="same">Same (Preserve Dimensions)</SelectItem>
                <SelectItem value="valid">Valid (No Padding)</SelectItem>
                <SelectItem value="causal">Causal (For Sequential Data)</SelectItem>
              </SelectContent>
            </Select>
            <div className="mt-2 text-sm text-gray-600">
              Padding determines how edges of the input are handled during convolution. "Same" padding preserves spatial dimensions, while "Valid" allows dimensions to shrink with each convolution.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="lower" onClick={() => setPadding('valid')}>Valid</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setPadding('causal')}>Causal</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setPadding('same')}>Same</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>
    </>
  );

  // Activation Function Tabs
  const renderActivationTabs = () => (
    <>
      <TabsList className="mb-4 w-full justify-start overflow-x-auto">
        <TabsTrigger value="activation-function" className="flex items-center gap-1">
          <Zap className="h-4 w-4" />
          Hidden Activations
        </TabsTrigger>
        <TabsTrigger value="output-activation" className="flex items-center gap-1">
          <Activity className="h-4 w-4" />
          Output Activation
        </TabsTrigger>
      </TabsList>

      <TabsContent value="activation-function">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label className="text-sm font-medium">
                Activation Function: <span className="font-mono capitalize">{activationFunc}</span>
              </Label>
              <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">High Impact</Badge>
            </div>
            <Select value={activationFunc} onValueChange={setActivationFunc}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select activation function" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="relu">ReLU (Rectified Linear Unit)</SelectItem>
                <SelectItem value="sigmoid">Sigmoid</SelectItem>
                <SelectItem value="tanh">Tanh</SelectItem>
                <SelectItem value="leaky_relu">Leaky ReLU</SelectItem>
                <SelectItem value="elu">ELU</SelectItem>
                <SelectItem value="selu">SELU</SelectItem>
                <SelectItem value="gelu">GELU</SelectItem>
              </SelectContent>
            </Select>
            <div className="mt-2 text-sm text-gray-600">
              Activation functions introduce non-linearity, allowing neural networks to learn complex patterns. Each function has different characteristics affecting gradient flow, training dynamics, and model capacity.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="advanced" onClick={() => setActivationFunc('relu')}>ReLU</BadgeCustom>
              <BadgeCustom type="lower" onClick={() => setActivationFunc('sigmoid')}>Sigmoid</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setActivationFunc('leaky_relu')}>Leaky ReLU</BadgeCustom>
              <BadgeCustom type="extreme" onClick={() => setActivationFunc('gelu')}>GELU</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="output-activation">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label className="text-sm font-medium">
                Output Activation: <span className="font-mono capitalize">{outputActivation}</span>
              </Label>
              <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">High Impact</Badge>
            </div>
            <Select value={outputActivation} onValueChange={setOutputActivation}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select output activation" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="softmax">Softmax (Multi-class Classification)</SelectItem>
                <SelectItem value="sigmoid">Sigmoid (Binary Classification)</SelectItem>
                <SelectItem value="linear">Linear (Regression)</SelectItem>
              </SelectContent>
            </Select>
            <div className="mt-2 text-sm text-gray-600">
              The output activation function determines how the model produces its final predictions, and should match your task type. Softmax for multi-class classification, sigmoid for binary, and linear for regression.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="advanced" onClick={() => setOutputActivation('softmax')}>Softmax</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setOutputActivation('sigmoid')}>Sigmoid</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setOutputActivation('linear')}>Linear</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>
    </>
  );

  // Loss Function Tabs
  const renderLossTabs = () => (
    <>
      <TabsList className="mb-4 w-full justify-start overflow-x-auto">
        <TabsTrigger value="loss-function" className="flex items-center gap-1">
          <FunctionSquare className="h-4 w-4" />
          Loss Function
        </TabsTrigger>
      </TabsList>

      <TabsContent value="loss-function">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label className="text-sm font-medium">
                Loss Function: <span className="font-mono capitalize">{lossFunction.replace('_', ' ')}</span>
              </Label>
              <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100">High Impact</Badge>
            </div>
            <Select value={lossFunction} onValueChange={setLossFunction}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select loss function" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="categorical_crossentropy">Categorical Cross-Entropy (Multi-class)</SelectItem>
                <SelectItem value="binary_crossentropy">Binary Cross-Entropy</SelectItem>
                <SelectItem value="mse">Mean Squared Error (Regression)</SelectItem>
                <SelectItem value="mae">Mean Absolute Error</SelectItem>
                <SelectItem value="huber">Huber Loss (Robust to Outliers)</SelectItem>
                <SelectItem value="kl_divergence">KL Divergence</SelectItem>
              </SelectContent>
            </Select>
            <div className="mt-2 text-sm text-gray-600">
              The loss function defines the optimization objective during training. It should match your task type and data characteristics. Different loss functions have different sensitivities to outliers and class imbalances.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="advanced" onClick={() => setLossFunction('categorical_crossentropy')}>Cat. CrossEntropy</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setLossFunction('mse')}>MSE</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setLossFunction('binary_crossentropy')}>Binary CrossEntropy</BadgeCustom>
              <BadgeCustom type="extreme" onClick={() => setLossFunction('huber')}>Huber</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>
    </>
  );

  // Regularization Tabs
  const renderRegularizationTabs = () => (
    <>
      <TabsList className="mb-4 w-full justify-start overflow-x-auto">
        <TabsTrigger value="l1-regularization" className="flex items-center gap-1">
          <Code className="h-4 w-4" />
          L1 Regularization
        </TabsTrigger>
        <TabsTrigger value="l2-regularization" className="flex items-center gap-1">
          <Code className="h-4 w-4" />
          L2 Regularization
        </TabsTrigger>
        <TabsTrigger value="batch-normalization" className="flex items-center gap-1">
          <Database className="h-4 w-4" />
          Batch Normalization
        </TabsTrigger>
      </TabsList>

      <TabsContent value="l1-regularization">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label htmlFor="l1-reg-slider" className="text-sm font-medium">
                L1 Regularization Strength: <span className="font-mono">{l1Reg.toFixed(4)}</span>
              </Label>
              <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
            </div>
            <Slider
              id="l1-reg-slider"
              min={0}
              max={0.1}
              step={0.0001}
              value={[l1Reg]}
              onValueChange={(value) => setL1Reg(value[0])}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>None (0.0)</span>
              <span>Strong (0.1)</span>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              L1 regularization (Lasso) adds a penalty proportional to the absolute value of weights, encouraging sparse models by driving some weights to zero. This helps with feature selection and simpler models.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="lower" onClick={() => setL1Reg(0)}>0.0</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setL1Reg(0.01)}>0.01</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setL1Reg(0.001)}>0.001</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="l2-regularization">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label htmlFor="l2-reg-slider" className="text-sm font-medium">
                L2 Regularization Strength: <span className="font-mono">{l2Reg.toFixed(4)}</span>
              </Label>
              <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
            </div>
            <Slider
              id="l2-reg-slider"
              min={0}
              max={0.1}
              step={0.0001}
              value={[l2Reg]}
              onValueChange={(value) => setL2Reg(value[0])}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>None (0.0)</span>
              <span>Strong (0.1)</span>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              L2 regularization (Ridge) adds a penalty proportional to the squared value of weights, discouraging large weights and encouraging more distributed weight values. This helps prevent overfitting.
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <BadgeCustom type="lower" onClick={() => setL2Reg(0)}>0.0</BadgeCustom>
              <BadgeCustom type="higher" onClick={() => setL2Reg(0.1)}>0.1</BadgeCustom>
              <BadgeCustom type="advanced" onClick={() => setL2Reg(0.01)}>0.01</BadgeCustom>
            </div>
          </div>
        </div>
      </TabsContent>

      <TabsContent value="batch-normalization">
        <div className="space-y-4">
          <div className="flex flex-col gap-2">
            <div className="flex justify-between">
              <Label className="text-sm font-medium">
                Batch Normalization: <span className="font-mono">{batchNorm ? 'Enabled' : 'Disabled'}</span>
              </Label>
              <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Medium Impact</Badge>
            </div>
            <div className="flex items-center space-x-4 mt-2">
              <Button 
                variant={batchNorm ? "default" : "outline"} 
                className="flex-1"
                onClick={() => setBatchNorm(true)}
              >
                Enabled
              </Button>
              <Button 
                variant={!batchNorm ? "default" : "outline"} 
                className="flex-1"
                onClick={() => setBatchNorm(false)}
              >
                Disabled
              </Button>
            </div>
            <div className="mt-2 text-sm text-gray-600">
              Batch normalization normalizes layer inputs by re-centering and re-scaling, which stabilizes and accelerates training by reducing internal covariate shift. It often reduces the need for careful weight initialization.
            </div>
          </div>
        </div>
      </TabsContent>
    </>
  );

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Maximize2 className="h-5 w-5 text-primary" />
          Interactive Parameter Playground
        </CardTitle>
        <CardDescription>
          Experiment with different model parameters and configurations to see their effects in real-time
        </CardDescription>
      </CardHeader>
      <CardContent>
        {/* Parameter Category Selection */}
        <div className="mb-6">
          <h3 className="text-sm font-medium mb-2">Parameter Category:</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
            <Button 
              variant={activeCategory === 'hyperparameters' ? "default" : "outline"} 
              className="flex items-center gap-1 text-sm justify-center"
              onClick={() => handleCategoryChange('hyperparameters')}
            >
              <TrendingUp className="h-4 w-4" />
              Hyperparameters
            </Button>
            <Button 
              variant={activeCategory === 'architecture' ? "default" : "outline"} 
              className="flex items-center gap-1 text-sm justify-center"
              onClick={() => handleCategoryChange('architecture')}
            >
              <Box className="h-4 w-4" />
              Architecture
            </Button>
            <Button 
              variant={activeCategory === 'layer-config' ? "default" : "outline"} 
              className="flex items-center gap-1 text-sm justify-center"
              onClick={() => handleCategoryChange('layer-config')}
            >
              <Layers className="h-4 w-4" />
              Layer Config
            </Button>
            <Button 
              variant={activeCategory === 'activation' ? "default" : "outline"} 
              className="flex items-center gap-1 text-sm justify-center"
              onClick={() => handleCategoryChange('activation')}
            >
              <Zap className="h-4 w-4" />
              Activation
            </Button>
            <Button 
              variant={activeCategory === 'loss' ? "default" : "outline"} 
              className="flex items-center gap-1 text-sm justify-center"
              onClick={() => handleCategoryChange('loss')}
            >
              <FunctionSquare className="h-4 w-4" />
              Loss Function
            </Button>
            <Button 
              variant={activeCategory === 'regularization' ? "default" : "outline"} 
              className="flex items-center gap-1 text-sm justify-center"
              onClick={() => handleCategoryChange('regularization')}
            >
              <Activity className="h-4 w-4" />
              Regularization
            </Button>
          </div>
        </div>

        {/* Parameter Configuration Tabs */}
        <Tabs defaultValue={activeTab} value={activeTab} onValueChange={setActiveTab} className="w-full">
          {renderTabContent()}
        </Tabs>

        {/* Visualizations */}
        <div className="mt-8 grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="w-full">
            <NetworkVisualization 
              paramName={activeParameter.name}
              paramValue={activeParameter.value}
              framework={activeParameter.framework}
            />
          </div>
          
          <div className="w-full">
            <HyperparameterVisualizations
              paramName={activeParameter.name}
              paramValue={activeParameter.value}
              framework={activeParameter.framework}
            />
          </div>
        </div>
        
        {/* Benchmark Comparison */}
        <div className="mt-8">
          <BenchmarkComparison
            paramName={activeParameter.name}
            paramValue={activeParameter.value}
            framework={activeParameter.framework}
          />
        </div>

        {/* Python Code Example */}
        <div className="mt-8">
          <Accordion type="single" collapsible className="w-full">
            <AccordionItem value="code-example">
              <AccordionTrigger className="text-md font-medium">
                Generated Code Example
              </AccordionTrigger>
              <AccordionContent>
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-md">
                  <pre className="text-xs overflow-x-auto">
                    <code className="language-python">
{`import tensorflow as tf
from tensorflow.keras.models import ${modelType === 'sequential' ? 'Sequential' : 'Model'}
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
${batchNorm ? 'from tensorflow.keras.layers import BatchNormalization' : ''}
from tensorflow.keras.optimizers import ${optimizer.charAt(0).toUpperCase() + optimizer.slice(1)}

# Create model
${modelType === 'sequential' ? 'model = Sequential()' : '# Functional API implementation would go here'}

# Add layers
${modelType === 'sequential' ? `
# First layer with input shape
model.add(Dense(${hiddenUnits}, activation='${activationFunc}', input_shape=(features,)))
${batchNorm ? 'model.add(BatchNormalization())' : ''}
${dropout > 0 ? `model.add(Dropout(${dropout}))` : ''}

# Hidden layers
${Array(hiddenLayers - 1).fill(0).map((_, i) => `model.add(Dense(${hiddenUnits}, activation='${activationFunc}'))\n${batchNorm ? 'model.add(BatchNormalization())\n' : ''}${dropout > 0 ? `model.add(Dropout(${dropout}))\n` : ''}`).join('')}

# Output layer
model.add(Dense(num_classes, activation='${outputActivation}'))
` : '# Layer implementation with Functional API would go here'}

# Compile model
model.compile(
    optimizer=${optimizer}(learning_rate=${learningRate}),
    loss='${lossFunction}',
    metrics=['accuracy']
)

# Add regularization if needed
${l1Reg > 0 || l2Reg > 0 ? `
# Note: In actual implementation, regularization would be added to each layer
# kernel_regularizer=tf.keras.regularizers.l1_l2(l1=${l1Reg}, l2=${l2Reg})
` : '# No regularization applied'}

# Train model
history = model.fit(
    x_train, 
    y_train,
    batch_size=${batchSize},
    epochs=${epochs},
    validation_split=0.2
)
`}
                    </code>
                  </pre>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </div>
      </CardContent>
    </Card>
  );
}