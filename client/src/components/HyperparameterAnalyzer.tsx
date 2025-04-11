import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Slider } from './ui/slider';
import { Code, BadgePlus, ChevronDown, ChevronUp, ArrowRight, Eye, Info, RefreshCw, TrendingUp, Layers } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

// Sample TensorFlow code for initial state
const INITIAL_CODE = `import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)`;

// Enhanced HyperparameterAnalyzer component
const HyperparameterAnalyzer = () => {
  // State for code input
  const [codeInput, setCodeInput] = useState(INITIAL_CODE);
  const [detectedParams, setDetectedParams] = useState([]);
  const [framework, setFramework] = useState('');
  const [expandedParams, setExpandedParams] = useState({});
  const [selectedParam, setSelectedParam] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('visualization');
  
  // Reference for canvas elements
  const networkCanvasRef = useRef(null);
  const chartCanvasRef = useRef(null);
  
  // State for playground controls
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(32);
  const [dropout, setDropout] = useState(0.5);
  const [epochs, setEpochs] = useState(10);
  const [optimizer, setOptimizer] = useState('adam');
  
  const { toast } = useToast();

  // Detect hyperparameters on code change
  useEffect(() => {
    if (codeInput.trim()) {
      analyzeCode();
    }
  }, [codeInput]);

  // Initialize visualizations when a parameter is selected
  useEffect(() => {
    if (selectedParam) {
      // Reset playground controls based on selected parameter
      if (selectedParam.key === 'learning_rate') {
        setLearningRate(parseFloat(selectedParam.value));
      } else if (selectedParam.key === 'batch_size') {
        setBatchSize(parseInt(selectedParam.value));
      } else if (selectedParam.key === 'dropout') {
        setDropout(parseFloat(selectedParam.value));
      } else if (selectedParam.key === 'epochs') {
        setEpochs(parseInt(selectedParam.value));
      }
      
      initNetworkVisualization();
      initChartVisualization();
    }
  }, [selectedParam]);

  // Function to analyze code and detect parameters
  const analyzeCode = () => {
    if (!codeInput.trim()) return;
    
    // Mock detection of hyperparameters
    const detectHyperparameters = (code) => {
      const patterns = {
        'learning_rate': /learning_rate\s*=\s*([\d.]+)/,
        'batch_size': /batch_size\s*=\s*(\d+)/,
        'dropout': /Dropout\(([\d.]+)\)/,
        'epochs': /epochs\s*=\s*(\d+)/,
        'optimizer': /optimizers\.([\w]+)/,
        'activation': /activation\s*=\s*['"](\w+)['"]/,
        'loss': /loss\s*=\s*['"]([^'"]+)['"]/,
        'hidden_units': /Dense\((\d+)/
      };
      
      const params = [];
      
      for (const [key, pattern] of Object.entries(patterns)) {
        const match = code.match(pattern);
        if (match) {
          const value = match[1];
          const position = {
            start: match.index,
            end: match.index + match[0].length
          };
          
          params.push({ key, value, position });
        }
      }
      
      return params;
    };
    
    // Detect framework based on imports
    const detectFramework = (code) => {
      if (code.includes('import tensorflow')) return 'TensorFlow';
      if (code.includes('import torch')) return 'PyTorch';
      if (code.includes('import keras')) return 'Keras';
      if (code.includes('from sklearn')) return 'scikit-learn';
      return 'Unknown';
    };
    
    const params = detectHyperparameters(codeInput);
    const detectedFramework = detectFramework(codeInput);
    
    setDetectedParams(params);
    setFramework(detectedFramework);
    
    // Reset expanded state for new parameters
    const initialExpandedState = {};
    params.forEach(param => {
      initialExpandedState[param.key] = false;
    });
    setExpandedParams(initialExpandedState);
    
    // Select first parameter by default if available
    if (params.length > 0 && !selectedParam) {
      handleSelectParameter(params[0]);
    }
    
    toast({
      title: "Code Analysis Complete",
      description: `Found ${params.length} parameters in ${detectedFramework} code.`,
      duration: 3000,
    });
  };

  // Toggle expanded state for parameter
  const toggleExpanded = (key) => {
    setExpandedParams(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  // Handle parameter selection
  const handleSelectParameter = (param) => {
    setSelectedParam(param);
    setIsLoading(true);
    
    // Generate parameter explanation
    setTimeout(() => {
      const explanations = {
        'learning_rate': {
          name: "Learning Rate",
          description: "Controls how quickly the model adapts to the problem by adjusting the size of weight and bias changes during training. Lower values provide more precise convergence but require more training iterations.",
          impact: "high",
          valueAnalysis: `The current value of ${param.value} is a standard choice for the Adam optimizer, providing a good balance between convergence speed and stability.`,
          alternatives: [
            { value: "0.01", description: "Faster learning but may cause instability or divergence in complex models", type: "higher" },
            { value: "0.0001", description: "Slower, more stable learning suitable for fine-tuning", type: "lower" },
            { value: "ReduceLROnPlateau", description: "Schedule that reduces learning rate when metrics plateau", type: "advanced" },
            { value: "CyclicLR", description: "Cycle between lower and upper learning rate boundaries", type: "extreme" }
          ],
          bestPractices: "Start with 0.001 for Adam, 0.01 for SGD. Monitor training loss curves for instability which indicates too high a learning rate. Decrease by factor of 10 when fine-tuning pre-trained models.",
          tradeoffs: "Higher learning rates speed up training but risk overshooting optima or diverging. Lower rates provide stable training but may get stuck in local minima or take too long to converge."
        },
        'batch_size': {
          name: "Batch Size",
          description: "The number of training examples processed together in one forward/backward pass. Affects memory usage, training speed, and the noise in the gradient updates.",
          impact: "medium",
          valueAnalysis: `The current batch size of ${param.value} provides a reasonable balance between training stability and memory requirements for most models.`,
          alternatives: [
            { value: "64", description: "Larger batches provide more stable gradient estimates but use more memory", type: "higher" },
            { value: "16", description: "Smaller batches introduce more noise which can improve generalization", type: "lower" },
            { value: "128", description: "Much larger batch size for systems with ample GPU memory", type: "higher" },
            { value: "Gradient Accumulation", description: "Technique to simulate larger batches on limited memory hardware", type: "advanced" }
          ],
          bestPractices: "Choose the largest batch size that fits in your GPU memory. Powers of 2 (16, 32, 64, 128) generally work well with modern hardware. If training performance plateaus, try reducing batch size.",
          tradeoffs: "Larger batches provide more stable and accurate gradient estimates but require more memory and may lead to poorer generalization. Smaller batches use less memory and can provide regularization effects but increase training variability."
        },
        'dropout': {
          name: "Dropout Rate",
          description: "A regularization technique that randomly sets a fraction of input units to 0 at each update during training to prevent overfitting.",
          impact: "medium",
          valueAnalysis: `A dropout rate of ${param.value} is a standard choice that typically works well for medium to large networks, providing effective regularization without severely impacting model capacity.`,
          alternatives: [
            { value: "0.2", description: "Lower dropout for smaller networks or when using other regularization", type: "lower" },
            { value: "0.8", description: "Higher dropout for very deep networks highly prone to overfitting", type: "higher" },
            { value: "0.0", description: "No dropout - useful for final fine-tuning stages", type: "extreme" },
            { value: "SpatialDropout", description: "Specialized version that drops entire feature maps in CNNs", type: "advanced" }
          ],
          bestPractices: "Apply dropout only during training, not inference. Place dropout layers between dense layers. Increase dropout rate for larger models. Consider coupling with batch normalization (before dropout layer).",
          tradeoffs: "Higher dropout rates provide stronger regularization but require longer training and may underfit. Lower rates may not sufficiently prevent overfitting. The optimal rate depends on model size, dataset size, and complexity."
        },
        'epochs': {
          name: "Number of Epochs",
          description: "An epoch is a complete pass through the entire training dataset. This parameter controls how many times the model will see the full training data.",
          impact: "medium",
          valueAnalysis: `Training for ${param.value} epochs provides a reasonable starting point, but the optimal number varies greatly depending on dataset size, model complexity, and learning rate.`,
          alternatives: [
            { value: "5", description: "Fewer epochs for large datasets or transfer learning", type: "lower" },
            { value: "50", description: "More epochs for complex tasks with adequate regularization", type: "higher" },
            { value: "EarlyStopping", description: "Dynamic approach that stops training when validation metrics stop improving", type: "advanced" },
            { value: "Checkpoint Ensemble", description: "Save models from multiple epochs to create an ensemble", type: "extreme" }
          ],
          bestPractices: "Use early stopping with a validation set to prevent overfitting. Save checkpoints of the best models. For transfer learning, use fewer epochs with a lower learning rate.",
          tradeoffs: "More epochs allow the model to learn more complex patterns but increase the risk of overfitting and computational cost. Fewer epochs may not allow sufficient learning but reduce the risk of overfitting."
        },
        'optimizer': {
          name: "Optimizer",
          description: "The algorithm used to update neural network weights based on the loss gradient. Different optimizers have different update rules that affect convergence behavior.",
          impact: "high",
          valueAnalysis: `The Adam optimizer is a popular default choice that adapts learning rates for each parameter, generally offering good performance across a wide range of problems.`,
          alternatives: [
            { value: "SGD", description: "Simple but requires careful tuning of learning rate and momentum", type: "advanced" },
            { value: "RMSprop", description: "Good for RNNs and problems with noisy gradients", type: "advanced" },
            { value: "AdamW", description: "Adam with decoupled weight decay for better regularization", type: "advanced" },
            { value: "Adafactor", description: "Memory-efficient alternative for very large models", type: "extreme" }
          ],
          bestPractices: "Adam is a good default choice. SGD with momentum often works better for CNNs with sufficient tuning. Consider switching optimizers when progress plateaus.",
          tradeoffs: "Adaptive optimizers like Adam converge faster but may generalize worse than well-tuned SGD. Different optimizers require different learning rates and hyperparameters."
        },
        'default': {
          name: "Parameter",
          description: "A configurable aspect of the machine learning model that affects its behavior, training process, or performance.",
          impact: "medium",
          valueAnalysis: `The current value of ${param.value} represents a specific configuration choice that affects model behavior.`,
          alternatives: [
            { value: "Lower value", description: "May be appropriate for simpler models or less complex data", type: "lower" },
            { value: "Higher value", description: "Could provide better model capacity at the cost of more computation", type: "higher" },
            { value: "Alternative approach", description: "Different configuration strategy that might work better for specific cases", type: "advanced" }
          ],
          bestPractices: "Experiment with different values through cross-validation. Consider the specific requirements of your dataset and task.",
          tradeoffs: "Most parameter choices involve trade-offs between model complexity, computational efficiency, and generalization performance."
        }
      };
      
      // Use specific explanation if available, otherwise default
      const paramExplanation = explanations[param.key] || explanations['default'];
      setExplanation(paramExplanation);
      setIsLoading(false);
    }, 500);
  };

  // Initialize network visualization
  const initNetworkVisualization = () => {
    if (!networkCanvasRef.current || !selectedParam) return;
    
    const canvas = networkCanvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Handle high-DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    
    ctx.scale(dpr, dpr);
    
    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height);
    
    // Draw network based on parameter
    const paramType = selectedParam.key;
    const paramValue = selectedParam.value;
    
    // Define network parameters
    let layerCount = 4;
    let neuronsPerLayer = [6, 8, 8, 4];
    let connectionOpacity = 0.3;
    let connectionThickness = 1;
    let neuronSize = 10;
    let dropoutRate = 0;
    
    // Adjust parameters based on the selected hyperparameter
    if (paramType === 'learning_rate') {
      const lr = parseFloat(paramValue);
      connectionOpacity = 0.2 + Math.min(lr * 200, 0.7);
      connectionThickness = 0.5 + Math.min(lr * 100, 2);
    } else if (paramType === 'batch_size') {
      const bs = parseInt(paramValue);
      const scaledSize = Math.log2(bs) / 2;
      neuronsPerLayer = [
        Math.max(3, Math.floor(scaledSize * 2)),
        Math.max(4, Math.floor(scaledSize * 3)),
        Math.max(4, Math.floor(scaledSize * 3)),
        Math.max(2, Math.floor(scaledSize * 1.5))
      ];
    } else if (paramType === 'dropout') {
      dropoutRate = parseFloat(paramValue);
    } else if (paramType === 'epochs') {
      const epochs = parseInt(paramValue);
      layerCount = Math.min(7, Math.max(3, Math.floor(Math.log10(epochs) * 2 + 3)));
      neuronsPerLayer = Array(layerCount).fill(0).map((_, i) => {
        if (i === 0) return 4; // Input layer
        if (i === layerCount - 1) return 3; // Output layer
        return 6; // Hidden layers
      });
    }
    
    // Calculate canvas dimensions
    const width = rect.width;
    const height = rect.height;
    
    // Colors
    const activeNeuronColor = '#4F46E5'; // Indigo
    const outputNeuronColor = '#EF4444'; // Red
    const inactiveNeuronColor = 'rgba(209, 213, 219, 0.5)'; // Light gray
    
    // Calculate neuron positions for each layer
    const layers = [];
    const layerSpacing = width / (layerCount + 1);
    
    for (let l = 0; l < layerCount; l++) {
      const layer = [];
      const neuronsInThisLayer = neuronsPerLayer[l] || 5;
      const isOutputLayer = l === layerCount - 1;
      
      // X position is based on layer index
      const x = (l + 1) * layerSpacing;
      
      // Calculate neuron positions for this layer
      for (let n = 0; n < neuronsInThisLayer; n++) {
        // Y position distributes neurons evenly
        const y = height * 0.1 + (n * (height * 0.8) / (neuronsInThisLayer - 1 || 1));
        
        // Determine if neuron is active (apply dropout to hidden layers only)
        const shouldApplyDropout = l > 0 && l < layerCount - 1;
        const active = !shouldApplyDropout || Math.random() > dropoutRate;
        
        layer.push({ x, y, active, isOutput: isOutputLayer });
      }
      
      layers.push(layer);
    }
    
    // Draw connections before neurons
    for (let l = 0; l < layers.length - 1; l++) {
      const fromLayer = layers[l];
      const toLayer = layers[l + 1];
      
      // Draw connections
      for (const fromNeuron of fromLayer) {
        if (!fromNeuron.active) continue;
        
        for (const toNeuron of toLayer) {
          if (!toNeuron.active) continue;
          
          // Draw connection line
          ctx.beginPath();
          ctx.moveTo(fromNeuron.x, fromNeuron.y);
          ctx.lineTo(toNeuron.x, toNeuron.y);
          ctx.strokeStyle = `rgba(79, 70, 229, ${connectionOpacity})`;
          ctx.lineWidth = connectionThickness;
          ctx.stroke();
        }
      }
    }
    
    // Draw neurons on top of connections
    for (let l = 0; l < layers.length; l++) {
      for (const neuron of layers[l]) {
        if (neuron.active) {
          // Active neuron
          ctx.beginPath();
          ctx.arc(neuron.x, neuron.y, neuronSize, 0, Math.PI * 2);
          ctx.fillStyle = neuron.isOutput ? outputNeuronColor : activeNeuronColor;
          ctx.fill();
        } else {
          // Inactive neuron (dropout)
          ctx.beginPath();
          ctx.arc(neuron.x, neuron.y, neuronSize, 0, Math.PI * 2);
          ctx.fillStyle = inactiveNeuronColor;
          ctx.fill();
          
          // Draw X mark to indicate dropout
          ctx.beginPath();
          ctx.moveTo(neuron.x - neuronSize / 2, neuron.y - neuronSize / 2);
          ctx.lineTo(neuron.x + neuronSize / 2, neuron.y + neuronSize / 2);
          ctx.moveTo(neuron.x + neuronSize / 2, neuron.y - neuronSize / 2);
          ctx.lineTo(neuron.x - neuronSize / 2, neuron.y + neuronSize / 2);
          ctx.strokeStyle = 'rgba(220, 38, 38, 0.5)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
    
    // Draw layer labels
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#6b7280';
    
    const layerNames = ['Input', ...Array(layerCount - 2).fill(0).map((_, i) => `Hidden ${i+1}`), 'Output'];
    for (let l = 0; l < layerCount; l++) {
      if (layers[l].length > 0) {
        ctx.fillText(layerNames[l], layers[l][0].x, 20);
      }
    }
  };
  
  // Initialize chart visualization
  const initChartVisualization = () => {
    if (!chartCanvasRef.current || !selectedParam) return;
    
    const canvas = chartCanvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Handle high-DPI displays
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    
    ctx.scale(dpr, dpr);
    
    // Clear canvas
    ctx.clearRect(0, 0, rect.width, rect.height);
    
    // Generate chart data based on parameter
    const paramType = selectedParam.key;
    const paramValue = selectedParam.value;
    
    let labels = [];
    let accuracyData = [];
    let secondaryData = [];
    
    const width = rect.width;
    const height = rect.height;
    const padding = 40;
    const chartWidth = width - (padding * 2);
    const chartHeight = height - (padding * 2);
    
    if (paramType === 'learning_rate') {
      labels = ['0.0001', '0.001', '0.01', '0.1', '1.0'];
      accuracyData = [0.65, 0.78, 0.88, 0.82, 0.70];
      secondaryData = [0.64, 0.76, 0.84, 0.76, 0.62];
    } else if (paramType === 'batch_size') {
      labels = ['8', '16', '32', '64', '128', '256'];
      accuracyData = [0.84, 0.85, 0.86, 0.85, 0.83, 0.81];
      secondaryData = [1.0, 0.8, 0.6, 0.4, 0.3, 0.25]; // Training time (relative)
    } else if (paramType === 'dropout') {
      labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'];
      accuracyData = [0.82, 0.86, 0.89, 0.90, 0.88, 0.85, 0.80, 0.72];
      secondaryData = [0.99, 0.97, 0.95, 0.92, 0.88, 0.84, 0.78, 0.72];
    } else if (paramType === 'epochs') {
      labels = ['1', '5', '10', '20', '50', '100'];
      accuracyData = [0.70, 0.82, 0.86, 0.87, 0.86, 0.85];
      secondaryData = [0.68, 0.85, 0.92, 0.97, 0.99, 0.995];
    } else {
      // Default chart for other parameters
      labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High'];
      accuracyData = [0.65, 0.78, 0.88, 0.82, 0.70];
      secondaryData = [0.5, 0.7, 0.9, 0.7, 0.5];
    }
    
    // Draw chart background
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(padding, padding, chartWidth, chartHeight);
    
    // Draw grid lines
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    const xStep = chartWidth / (labels.length - 1);
    for (let i = 0; i < labels.length; i++) {
      const x = padding + (i * xStep);
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, padding + chartHeight);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    const yStep = chartHeight / 4;
    for (let i = 0; i <= 4; i++) {
      const y = padding + (i * yStep);
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + chartWidth, y);
      ctx.stroke();
    }
    
    // Draw axes labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis labels
    for (let i = 0; i < labels.length; i++) {
      const x = padding + (i * xStep);
      ctx.fillText(labels[i], x, padding + chartHeight + 15);
    }
    
    // Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const y = padding + (chartHeight - i * yStep);
      ctx.fillText((i * 0.25).toFixed(2), padding - 10, y + 4);
    }
    
    // Draw axes titles
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    
    // X-axis title
    ctx.fillText(paramType.replace('_', ' ').charAt(0).toUpperCase() + paramType.replace('_', ' ').slice(1), width / 2, height - 5);
    
    // Y-axis title
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Performance', 0, 0);
    ctx.restore();
    
    // Function to convert data point to canvas coordinates
    const getPointCoords = (value, index) => {
      const x = padding + (index * xStep);
      const y = padding + chartHeight - (value * chartHeight);
      return { x, y };
    };
    
    // Draw first dataset line (validation accuracy)
    ctx.strokeStyle = '#4f46e5'; // Indigo
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    for (let i = 0; i < accuracyData.length; i++) {
      const { x, y } = getPointCoords(accuracyData[i], i);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    // Draw second dataset line (training accuracy)
    ctx.strokeStyle = '#ef4444'; // Red
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    for (let i = 0; i < secondaryData.length; i++) {
      const { x, y } = getPointCoords(secondaryData[i], i);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.stroke();
    
    // Draw points for first dataset
    ctx.fillStyle = '#4f46e5';
    for (let i = 0; i < accuracyData.length; i++) {
      const { x, y } = getPointCoords(accuracyData[i], i);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw points for second dataset
    ctx.fillStyle = '#ef4444';
    for (let i = 0; i < secondaryData.length; i++) {
      const { x, y } = getPointCoords(secondaryData[i], i);
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
    }
    
    // Draw legend
    const legendX = padding + 20;
    const legendY = padding + 20;
    
    // First dataset legend
    ctx.fillStyle = '#4f46e5';
    ctx.beginPath();
    ctx.rect(legendX, legendY, 15, 15);
    ctx.fill();
    
    ctx.fillStyle = '#111827';
    ctx.textAlign = 'left';
    ctx.fillText('Validation Accuracy', legendX + 20, legendY + 12);
    
    // Second dataset legend
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.rect(legendX, legendY + 25, 15, 15);
    ctx.fill();
    
    ctx.fillStyle = '#111827';
    ctx.fillText(paramType === 'batch_size' ? 'Training Speed' : 'Training Accuracy', legendX + 20, legendY + 37);
    
    // Highlight current value on chart
    const currentValueIndex = labels.indexOf(paramValue);
    if (currentValueIndex >= 0) {
      const { x, y } = getPointCoords(accuracyData[currentValueIndex], currentValueIndex);
      
      // Draw highlight circle
      ctx.strokeStyle = '#4f46e5';
      ctx.lineWidth = 2;
      ctx.fillStyle = 'rgba(79, 70, 229, 0.1)';
      ctx.beginPath();
      ctx.arc(x, y, 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Draw value label
      ctx.fillStyle = '#111827';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Current: ${paramValue}`, x, y - 15);
    }
  };

  // Get impact color
  const getImpactColor = (impact) => {
    switch(impact) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'medium':
        return 'bg-amber-100 text-amber-800 border-amber-300';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-300';
      default:
        return 'bg-blue-100 text-blue-800 border-blue-300';
    }
  };

  // Interactive Playground Slider for parameters
  const PlaygroundSlider = ({ value, onChange, min, max, step, label, description }) => (
    <div className="mb-6">
      <div className="flex justify-between mb-2">
        <label className="text-sm font-medium text-gray-700">{label}: <span className="font-mono">{typeof value === 'number' ? value.toString().includes('.') ? value.toFixed(6) : value : value}</span></label>
      </div>
      <Slider 
        value={[typeof value === 'number' ? value : 0]} 
        onValueChange={(values) => onChange(values[0])}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>Min: {min}</span>
        <span>Max: {max}</span>
      </div>
      {description && (
        <p className="text-xs text-gray-500 mt-2">{description}</p>
      )}
    </div>
  );

  return (
    <div className="container mx-auto py-6 px-4">
      <header className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Eye className="h-8 w-8 text-primary" />
          <span>HyperExplainer</span>
        </h1>
        <p className="text-muted-foreground">
          Automatically detect and understand hyperparameters in machine learning code
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Code Input Section */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="h-5 w-5" />
                Code Input
              </CardTitle>
              <CardDescription>
                Paste machine learning code to analyze
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={codeInput}
                onChange={(e) => setCodeInput(e.target.value)}
                className="font-mono text-sm h-96"
                placeholder="Paste your ML code here..."
              />
              <Button 
                onClick={analyzeCode} 
                className="w-full mt-4"
              >
                Analyze Code
              </Button>
              
              {/* Detected Parameters List */}
              {detectedParams.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-medium mb-2 flex items-center">
                    <BadgePlus className="h-4 w-4 mr-1" />
                    Detected Parameters
                    {framework && (
                      <Badge className="ml-2" variant="outline">
                        {framework}
                      </Badge>
                    )}
                  </h3>
                  <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                    {detectedParams.map((param, index) => (
                      <div 
                        key={index}
                        className={`border rounded-md overflow-hidden transition-all ${
                          selectedParam?.key === param.key ? 'ring-2 ring-primary' : ''
                        }`}
                      >
                        <div 
                          className="p-3 cursor-pointer flex justify-between items-center hover:bg-gray-50"
                          onClick={() => handleSelectParameter(param)}
                        >
                          <div>
                            <h4 className="font-medium text-sm">{param.key}</h4>
                            <div className="text-sm font-mono text-gray-600">{param.value}</div>
                          </div>
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleExpanded(param.key);
                            }}
                            className="p-1 rounded-full hover:bg-gray-200"
                          >
                            {expandedParams[param.key] ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </button>
                        </div>
                        
                        {expandedParams[param.key] && (
                          <div className="px-3 pb-3 text-xs text-gray-600 border-t pt-2">
                            <div className="flex items-center gap-1 mb-1">
                              <Info className="h-3 w-3" />
                              <span>Position: {param.position.start}-{param.position.end}</span>
                            </div>
                            <Button 
                              size="sm" 
                              variant="ghost" 
                              className="text-xs py-1 h-7 mt-1"
                              onClick={() => handleSelectParameter(param)}
                            >
                              View Details
                              <ArrowRight className="h-3 w-3 ml-1" />
                            </Button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        
        {/* Hyperparameter Explanation & Visualization */}
        <div className="lg:col-span-2">
          {selectedParam ? (
            <div className="space-y-6">
              {/* Explanation Card */}
              {explanation && (
                <Card>
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-xl font-bold">
                        {explanation.name}
                      </CardTitle>
                      <div className="flex items-center">
                        <Badge className={`${getImpactColor(explanation.impact)}`}>
                          {explanation.impact.charAt(0).toUpperCase() + explanation.impact.slice(1)} Impact
                        </Badge>
                      </div>
                    </div>
                    <CardDescription className="text-sm">
                      Current value: <code className="font-mono bg-gray-100 px-1 py-0.5 rounded">{selectedParam.value}</code>
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    <div>
                      <h3 className="text-sm font-medium mb-1">Description</h3>
                      <p className="text-sm text-gray-600">{explanation.description}</p>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium mb-1">Value Analysis</h3>
                      <p className="text-sm text-gray-600">{explanation.valueAnalysis}</p>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium mb-1">Alternative Values</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {explanation.alternatives.map((alt, index) => (
                          <div key={index} className="border rounded-lg p-2">
                            <div className="flex justify-between items-center mb-1">
                              <code className="text-sm font-mono bg-gray-100 px-1 rounded">
                                {alt.value}
                              </code>
                              <Badge 
                                className={
                                  alt.type === 'higher' ? 'bg-blue-100 text-blue-800' :
                                  alt.type === 'lower' ? 'bg-green-100 text-green-800' :
                                  alt.type === 'advanced' ? 'bg-purple-100 text-purple-800' :
                                  'bg-red-100 text-red-800'
                                }
                              >
                                {alt.type}
                              </Badge>
                            </div>
                            <p className="text-xs text-gray-600">{alt.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium mb-1">Best Practices</h3>
                      <p className="text-sm text-gray-600">{explanation.bestPractices}</p>
                    </div>
                    
                    <div>
                      <h3 className="text-sm font-medium mb-1">Trade-offs</h3>
                      <p className="text-sm text-gray-600">{explanation.tradeoffs}</p>
                    </div>
                  </CardContent>
                </Card>
              )}
              
              {/* Visualization Tabs */}
              <Tabs defaultValue="visualization" value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="mb-4 w-full">
                  <TabsTrigger value="visualization" className="flex items-center gap-1">
                    <TrendingUp className="h-4 w-4" />
                    Impact Visualization
                  </TabsTrigger>
                  <TabsTrigger value="network" className="flex items-center gap-1">
                    <Layers className="h-4 w-4" />
                    Neural Network
                  </TabsTrigger>
                  <TabsTrigger value="playground" className="flex items-center gap-1">
                    <RefreshCw className="h-4 w-4" />
                    Playground
                  </TabsTrigger>
                </TabsList>
                
                <TabsContent value="visualization" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Impact on Model Performance</CardTitle>
                      <CardDescription>
                        How {selectedParam.key.replace('_', ' ')} affects model behavior
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="w-full h-64 border rounded-md">
                        <canvas ref={chartCanvasRef} className="w-full h-full" />
                      </div>
                      <div className="mt-4 text-sm text-gray-500">
                        This visualization shows how different values for {selectedParam.key.replace('_', ' ')} affect model performance.
                        The optimal range depends on your specific dataset and model architecture.
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="network" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Neural Network Visualization</CardTitle>
                      <CardDescription>
                        Visual effect of {selectedParam.key.replace('_', ' ')} on network behavior
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="w-full h-64 border rounded-md">
                        <canvas ref={networkCanvasRef} className="w-full h-full" />
                      </div>
                      <div className="mt-4 text-sm text-gray-500">
                        This visualization shows a typical neural network architecture with the parameter applied.
                        {selectedParam.key === 'dropout' && " Inactive neurons (transparent) demonstrate the dropout effect."}
                        {selectedParam.key === 'learning_rate' && " Connection strength visualizes learning rate impact."}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
                
                <TabsContent value="playground" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Interactive Parameter Playground</CardTitle>
                      <CardDescription>
                        Experiment with different parameter values
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {selectedParam.key === 'learning_rate' && (
                        <PlaygroundSlider
                          value={learningRate}
                          onChange={setLearningRate}
                          min={0.00001}
                          max={0.1}
                          step={0.00001}
                          label="Learning Rate"
                          description="Controls how quickly the model adapts to the problem by adjusting weights."
                        />
                      )}
                      
                      {selectedParam.key === 'batch_size' && (
                        <PlaygroundSlider
                          value={batchSize}
                          onChange={setBatchSize}
                          min={1}
                          max={512}
                          step={1}
                          label="Batch Size"
                          description="The number of training examples processed in one iteration."
                        />
                      )}
                      
                      {selectedParam.key === 'dropout' && (
                        <PlaygroundSlider
                          value={dropout}
                          onChange={setDropout}
                          min={0}
                          max={0.9}
                          step={0.01}
                          label="Dropout Rate"
                          description="Fraction of the input units to drop during training to prevent overfitting."
                        />
                      )}
                      
                      {selectedParam.key === 'epochs' && (
                        <PlaygroundSlider
                          value={epochs}
                          onChange={setEpochs}
                          min={1}
                          max={100}
                          step={1}
                          label="Epochs"
                          description="Number of complete passes through the entire training dataset."
                        />
                      )}
                      
                      {selectedParam.key === 'optimizer' && (
                        <div className="mb-6">
                          <label className="text-sm font-medium text-gray-700 mb-2 block">
                            Optimizer: <span className="font-mono capitalize">{optimizer}</span>
                          </label>
                          <Select value={optimizer} onValueChange={setOptimizer}>
                            <SelectTrigger className="w-full">
                              <SelectValue placeholder="Select optimizer" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="adam">Adam</SelectItem>
                              <SelectItem value="sgd">SGD</SelectItem>
                              <SelectItem value="rmsprop">RMSprop</SelectItem>
                              <SelectItem value="adagrad">Adagrad</SelectItem>
                            </SelectContent>
                          </Select>
                          <p className="text-xs text-gray-500 mt-2">
                            The algorithm used to update the model weights during training.
                          </p>
                        </div>
                      )}
                      
                      {/* Apply Changes Button */}
                      <Button className="w-full mt-4" onClick={() => {
                        // Regenerate visualizations with new values
                        initNetworkVisualization();
                        initChartVisualization();
                        
                        toast({
                          title: "Parameters Updated",
                          description: "Visualizations have been refreshed with new values.",
                          duration: 3000,
                        });
                      }}>
                        Apply Changes
                      </Button>
                      
                      {/* Generated Code Preview */}
                      <div className="mt-6">
                        <h3 className="text-sm font-medium mb-2">Generated Code</h3>
                        <div className="bg-gray-50 rounded-md p-4">
                          <pre className="text-xs overflow-x-auto">
                            <code className="text-gray-800">{`import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(${dropout}),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.${optimizer.charAt(0).toUpperCase() + optimizer.slice(1)}(learning_rate=${learningRate})
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=${batchSize}, epochs=${epochs})`}</code>
                          </pre>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </div>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-10">
                <Info className="h-12 w-12 text-gray-300 mb-4" />
                <h3 className="text-xl font-medium text-gray-600 mb-2">No Parameter Selected</h3>
                <p className="text-gray-500 text-center max-w-md mb-4">
                  Enter your machine learning code and select a parameter to see detailed analysis and visualizations.
                </p>
                <Button onClick={analyzeCode}>Analyze Code</Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default HyperparameterAnalyzer;