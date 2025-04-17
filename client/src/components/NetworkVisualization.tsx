import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface NetworkVisualizationProps {
  paramName: string;
  paramValue: string;
  framework?: string;
}

export function NetworkVisualization({ 
  paramName, 
  paramValue,
  framework 
}: NetworkVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const isFirstRender = useRef(true);
  
  // Animation state
  const [showAnimation, setShowAnimation] = useState<boolean>(true);
  
  // Network type
  const [networkType, setNetworkType] = useState<string>('dense');
  
  // Get parameter range and configuration based on parameter type
  const getParameterRange = (): { 
    min: number; 
    max: number; 
    step: number; 
    defaultValue: number;
    visualizationType: string; // Type of visualization to use
    customSettings?: any;       // Any additional settings for this parameter type
  } => {
    // Create a unique visualization style for each major parameter type
    if (paramName.includes('learning_rate')) {
      return { 
        min: 0.0001, 
        max: 0.01, 
        step: 0.0001, 
        defaultValue: 0.001,
        visualizationType: 'flowField',  // Special visualization for learning rate
        customSettings: {
          particleCount: 150,          // Number of particles in the flow field
          particleSpeed: 0.5,          // Base speed of particles
          colorScheme: 'blueToRed',    // Color scheme for the particles
          flowIntensity: 3.0           // Intensity of the flow field
        }
      };
    } else if (paramName.includes('dropout')) {
      return { 
        min: 0, 
        max: 0.9, 
        step: 0.05, 
        defaultValue: 0.5,
        visualizationType: 'dropout',   // Specific visualization for dropout
        customSettings: {
          highlightDropped: true,      // Highlight neurons that are dropped
          dropoutPattern: 'random',    // Random pattern of dropout
          animateDropout: true         // Animate the dropout effect
        }
      };
    } else if (paramName.includes('batch_size')) {
      return { 
        min: 1, 
        max: 512, 
        step: 1, 
        defaultValue: 32,
        visualizationType: 'batchGroups', // Visualize batches of samples
        customSettings: {
          sampleVisibility: true,      // Show individual samples
          highlightBatches: true,      // Highlight batch boundaries
          colorByBatch: true           // Color samples by batch
        }
      };
    } else if (paramName.includes('momentum')) {
      return { 
        min: 0, 
        max: 0.99, 
        step: 0.01, 
        defaultValue: 0.9,
        visualizationType: 'momentum',  // Special visualization for momentum
        customSettings: {
          trailEffect: true,           // Show trailing effect for momentum
          particleInertia: true,       // Simulate inertia in particle motion
          velocityVectors: true        // Show velocity vectors
        }
      };
    } else if (paramName.includes('weight_decay') || paramName.includes('regularization')) {
      return { 
        min: 0, 
        max: 0.1, 
        step: 0.001, 
        defaultValue: 0.01,
        visualizationType: 'weightShrinkage', // Visualization for weight decay
        customSettings: {
          showWeightMagnitudes: true,   // Show the magnitude of weights
          shrinkageAnimation: true,     // Animate the shrinkage effect
          colorByWeight: true           // Color edges by weight value
        }
      };
    } else if (paramName.includes('epochs')) {
      return { 
        min: 1, 
        max: 100, 
        step: 1, 
        defaultValue: 10,
        visualizationType: 'trainingProgress', // Show training progress
        customSettings: {
          showLearningCurve: true,      // Show learning curve
          epochMarkers: true,           // Mark epochs on the visualization
          animateEpochs: true           // Animate through epochs
        }
      };
    } else if (paramName.includes('activation')) {
      return { 
        min: 0, 
        max: 1, 
        step: 0.01, 
        defaultValue: 0.5,
        visualizationType: 'activationFunction', // Show activation function
        customSettings: {
          showInputOutput: true,        // Show input/output mapping
          functionCurve: true,          // Show function curve
          gradientVisualization: true   // Visualize gradients
        }
      };
    } else if (paramName.includes('optimizer')) {
      return { 
        min: 0, 
        max: 1, 
        step: 0.01, 
        defaultValue: 0.5,
        visualizationType: 'optimizerPath',  // Show optimizer path
        customSettings: {
          showOptimizationPath: true,   // Show path of optimization
          contourLines: true,           // Show loss contours
          animateOptimization: true     // Animate optimization process
        }
      };
    } else if (paramName.includes('layer') || paramName.includes('hidden_units') || paramName.includes('units')) {
      return { 
        min: 1, 
        max: 256, 
        step: 1, 
        defaultValue: 64,
        visualizationType: 'networkArchitecture', // Specialized network architecture
        customSettings: {
          layerSizeAnimation: true,      // Animate changes in layer size
          layerConnectivity: 'complete', // Type of connectivity between layers
          showNodeRoles: true            // Show the role of each node
        }
      };
    } else {
      // Default range and generic visualization
      return { 
        min: 0, 
        max: 1, 
        step: 0.01, 
        defaultValue: 0.5,
        visualizationType: 'standard',    // Standard network visualization
        customSettings: {
          animateSignalFlow: true,       // Animate signal flow through network
          dynamicWeights: true           // Dynamically adjust weights
        }
      };
    }
  };
  
  // Get range and set initial value
  const range = getParameterRange();
  const [selectedValue, setSelectedValue] = useState<number>(
    paramValue ? parseFloat(paramValue) : range.defaultValue
  );
  
  // Handle canvas setup with ResizeObserver to ensure correct dimensions
  useEffect(() => {
    console.log('NetworkVisualization initial setup useEffect', { ref: canvasRef.current });
    
    if (!canvasRef.current) {
      console.error('Canvas ref is null in NetworkVisualization');
      return;
    }
    
    const setupCanvas = (canvas: HTMLCanvasElement) => {
      console.log('Setting up canvas for NetworkVisualization');
      
      // Handle high-DPI displays properly
      const dpr = window.devicePixelRatio || 1;
      console.log('Device pixel ratio:', dpr);
      
      // Get actual dimensions
      const rect = canvas.getBoundingClientRect();
      console.log('Canvas dimensions:', rect.width, 'x', rect.height);
      
      if (rect.width === 0 || rect.height === 0) {
        console.error('Canvas has zero width or height!');
        return false;
      }
      
      // Set the canvas attributes properly for high DPI displays
      canvas.width = Math.max(1, rect.width * dpr);
      canvas.height = Math.max(1, rect.height * dpr);
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Scale the context to compensate for high-DPI
        ctx.scale(dpr, dpr);
        console.log('Canvas 2D context initialized and scaled');
        return true;
      } else {
        console.error('Failed to get 2D context from canvas');
        return false;
      }
    };
    
    const canvas = canvasRef.current;
    
    // Use ResizeObserver to handle canvas sizing
    const resizeObserver = new ResizeObserver(entries => {
      console.log('ResizeObserver triggered', entries[0].contentRect);
      if (entries[0].contentRect.width > 0 && entries[0].contentRect.height > 0) {
        setupCanvas(canvas);
      }
    });
    
    // Observe the canvas
    resizeObserver.observe(canvas);
    
    // Initial setup with delay
    console.log('Setting up canvas with delay');
    setTimeout(() => {
      setupCanvas(canvas);
    }, 500);
    
    // Cleanup
    return () => {
      console.log('Cleaning up ResizeObserver');
      resizeObserver.disconnect();
    };
  }, []);
  
  // Convert parameter value to network parameters
  const getNetworkParams = () => {
    if (!canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    
    // Get visualization type and settings from parameter range
    const paramConfig = getParameterRange();
    const visualizationType = paramConfig.visualizationType;
    const customSettings = paramConfig.customSettings || {};
    
    // Base params
    const params = {
      layerCount: 4,
      neuronsPerLayer: [5, 7, 7, 3],
      connectionOpacity: 0.3,
      connectionThickness: 1,
      neuronSize: 7,
      layerSpacing: width / 5,
      canvasHeight: height,
      dropoutRate: 0,
      visualizationType, // Add visualization type
      customSettings,    // Add custom settings
      normalizedValue: (selectedValue - paramConfig.min) / (paramConfig.max - paramConfig.min) // Normalized 0-1 value
    };
    
    // Parameter-specific effects
    if (paramName.includes('learning_rate')) {
      // Learning rate affects connection strength/opacity
      const normalizedLR = params.normalizedValue;
      params.connectionOpacity = 0.2 + normalizedLR * 0.6;
      params.connectionThickness = 0.5 + normalizedLR * 1.5;
      
      // Special settings for flow field visualization
      params.customSettings = {
        ...params.customSettings,
        particleSpeed: 0.2 + normalizedLR * 0.8, // Speed increases with learning rate
        flowIntensity: 1.0 + normalizedLR * 5.0, // More chaotic at higher rates
      };
    } else if (paramName.includes('dropout')) {
      // Dropout directly affects dropout rate
      params.dropoutRate = selectedValue;
      
      // Special settings for dropout visualization
      params.customSettings = {
        ...params.customSettings,
        dropoutRate: selectedValue,
        dropoutPattern: selectedValue > 0.5 ? 'heavy' : 'light',
      };
    } else if (paramName.includes('batch_size')) {
      // Batch size affects number of neurons
      const normalizedBS = params.normalizedValue;
      const baseNeurons = 3;
      const maxNeurons = 10;
      const neuronCount = Math.round(baseNeurons + normalizedBS * (maxNeurons - baseNeurons));
      params.neuronsPerLayer = [neuronCount, neuronCount + 2, neuronCount + 2, Math.max(2, Math.round(neuronCount / 2))];
      
      // Custom batch settings
      params.customSettings = {
        ...params.customSettings,
        batchSize: Math.round(selectedValue),
        batchColor: `hsl(${Math.round(normalizedBS * 240)}, 80%, 60%)`,
      };
    } else if (paramName.includes('momentum')) {
      // Momentum affects connection thickness
      const normalizedMomentum = params.normalizedValue;
      params.connectionThickness = 0.5 + normalizedMomentum * 2.5;
      
      // Special settings for momentum visualization
      params.customSettings = {
        ...params.customSettings,
        trailLength: Math.round(5 + normalizedMomentum * 20), // Longer trails with higher momentum
        particleInertia: normalizedMomentum,
      };
    } else if (paramName.includes('weight_decay') || paramName.includes('regularization')) {
      // Weight decay inversely affects connection strength
      const normalizedWD = params.normalizedValue;
      params.connectionOpacity = 0.5 - normalizedWD * 0.3;
      
      // Custom weight decay settings
      params.customSettings = {
        ...params.customSettings,
        shrinkFactor: normalizedWD,
        weightVariability: 1.0 - normalizedWD * 0.7, // Less variability with higher decay
      };
    } else if (paramName.includes('epochs')) {
      // More epochs = more complex network
      const normalizedEpochs = params.normalizedValue;
      const minLayers = 3;
      const maxLayers = 6;
      params.layerCount = Math.round(minLayers + normalizedEpochs * (maxLayers - minLayers));
      params.neuronsPerLayer = Array(params.layerCount).fill(0).map((_, i) => {
        if (i === 0) return 5; // Input layer
        if (i === params.layerCount - 1) return 3; // Output layer
        return 7; // Hidden layers
      });
      
      // Custom epochs settings
      params.customSettings = {
        ...params.customSettings,
        currentEpoch: Math.round(selectedValue),
        convergenceRate: 1.0 - Math.exp(-normalizedEpochs * 3), // Converges faster with more epochs
      };
    } else if (paramName.includes('activation')) {
      // Activation functions get special visualization
      params.customSettings = {
        ...params.customSettings,
        activationType: paramName.includes('relu') ? 'relu' : 
                        paramName.includes('tanh') ? 'tanh' : 
                        paramName.includes('sigmoid') ? 'sigmoid' : 'linear',
        activationStrength: 0.2 + params.normalizedValue * 0.8,
      };
    } else if (paramName.includes('optimizer')) {
      // Optimizer gets special path visualization
      params.customSettings = {
        ...params.customSettings,
        optimizerType: paramName.includes('adam') ? 'adam' : 
                      paramName.includes('sgd') ? 'sgd' : 
                      paramName.includes('rmsprop') ? 'rmsprop' : 'generic',
        optimizerEfficiency: 0.3 + params.normalizedValue * 0.7,
      };
    } else if (paramName.includes('layer') || paramName.includes('hidden_units') || paramName.includes('units')) {
      // Units/layers affect network size
      const normalizedUnits = params.normalizedValue;
      const baseNeurons = 2;
      const maxNeurons = 12;
      const neuronCount = Math.round(baseNeurons + normalizedUnits * (maxNeurons - baseNeurons));
      
      // Dynamic network size based on units parameter
      params.neuronsPerLayer = [5, neuronCount, neuronCount, 3];
      
      // Custom settings
      params.customSettings = {
        ...params.customSettings,
        unitCount: Math.round(selectedValue),
        layerDensity: normalizedUnits,
      };
    }
    
    return params;
  };
  
  // Animation and drawing effect
  useEffect(() => {
    console.log('NetworkVisualization animation useEffect triggered', { 
      selectedValue, 
      networkType, 
      paramName,
      visualizationType 
    });
    
    if (!canvasRef.current) {
      console.error('Canvas ref is null in animation effect');
      return;
    }
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Failed to get 2D context in animation effect');
      return;
    }
    
    // Clear any existing animation
    if (animationRef.current) {
      console.log('Clearing existing animation');
      clearTimeout(animationRef.current);
      animationRef.current = null;
    }
    
    // Clear any stored state on canvas when visualization type changes
    if (canvas._particles || canvas._optimizerPath) {
      // Reset stored state when visualization type changes
      if (canvas._currentVisualizationType !== visualizationType) {
        console.log('Resetting canvas stored state for new visualization type');
        canvas._particles = null;
        canvas._optimizerPath = null;
        canvas._currentVisualizationType = visualizationType;
      }
    } else {
      canvas._currentVisualizationType = visualizationType;
    }
    
    // Account for device pixel ratio
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvas.width / dpr;
    const displayHeight = canvas.height / dpr;
    console.log('Canvas display dimensions:', displayWidth, 'x', displayHeight);
    
    // Get network parameters
    console.log('Getting network parameters');
    const params = getNetworkParams();
    if (!params) {
      console.error('Failed to get network parameters');
      return;
    }
    
    // Override the visualization type with the user-selected one
    params.visualizationType = visualizationType;
    console.log('Network parameters with visualization type:', params);
    
    // For static rendering when animation is off
    if (!showAnimation) {
      // Clear the canvas properly accounting for DPR
      ctx.clearRect(0, 0, displayWidth, displayHeight);
      drawNeuralNetwork(ctx, displayWidth, displayHeight, params, networkType, false);
      return;
    }
    
    // For animation, use a consistent approach with setTimeout
    let startTime = Date.now();
    
    const animate = () => {
      // Calculate elapsed time (very slow for educational purposes)
      const elapsed = (Date.now() - startTime) / 50000; // 50-second cycle
      
      // Clear the canvas properly accounting for DPR
      ctx.clearRect(0, 0, displayWidth, displayHeight);
      
      // Draw the network with animation
      drawNeuralNetwork(ctx, displayWidth, displayHeight, params, networkType, true, elapsed);
      
      // Loop with consistent timing
      animationRef.current = window.setTimeout(animate, 100); // 10fps is plenty for this visualization
    };
    
    // Start animation
    animate();
    
    // Cleanup
    return () => {
      if (animationRef.current) {
        clearTimeout(animationRef.current);
        animationRef.current = null;
      }
    };
  }, [selectedValue, networkType, paramName, showAnimation, visualizationType]);
  
  const handleValueChange = (value: number[] | number) => {
    if (Array.isArray(value)) {
      setSelectedValue(value[0]);
    } else {
      setSelectedValue(value);
    }
  };
  
  const toggleAnimation = () => {
    setShowAnimation(!showAnimation);
  };
  
  // State for visualization type selection
  const range = getParameterRange();
  const defaultVisualizationType = range.visualizationType;
  const [visualizationType, setVisualizationType] = useState<string>(defaultVisualizationType);
  
  // Get detailed information for different hyperparameters and visualizations
  const getParameterInfo = () => {
    // Define detailed descriptions for each parameter type
    const parameterInfo = {
      learning_rate: {
        name: "Learning Rate",
        description: "Controls how quickly the neural network updates its weights during training. It determines the step size at each iteration as the model moves toward minimizing the loss function.",
        valueAnalysis: (value: string) => {
          const numValue = parseFloat(value);
          if (numValue <= 0.0001) return "Very low learning rate (≤0.0001): Training will be extremely slow but stable. Good for fine-tuning pre-trained models or when dealing with complex loss landscapes. May get stuck in local minima.";
          if (numValue <= 0.001) return "Low learning rate (0.0001-0.001): Slow but stable training. This is often the default for Adam optimizer. Good for most applications as it balances convergence speed and stability.";
          if (numValue <= 0.01) return "Medium learning rate (0.001-0.01): Faster training, but may oscillate around minima. Often used with SGD with momentum. Requires careful monitoring of loss curves.";
          if (numValue <= 0.1) return "High learning rate (0.01-0.1): Very fast initial training but high risk of overshooting or divergence. May need learning rate schedules to reduce the rate over time.";
          return "Extremely high learning rate (>0.1): Very likely to cause training instability or divergence. Only usable in specific scenarios with robust optimization strategies like learning rate warmup.";
        },
        bestPractices: "• Start with 0.001 for Adam or 0.01 for SGD\n• Use learning rate schedulers to decrease the rate over time\n• For transfer learning, use 10-100x lower rates than training from scratch\n• Monitor loss curves for oscillations (too high) or plateaus (too low)\n• Consider learning rate warmup for very deep networks\n• Try learning rate finder techniques to identify optimal values",
        tradeoffs: "• Higher rates speed up training but risk overshooting optima or diverging\n• Lower rates provide stable training but may get stuck in local minima or take too long to converge\n• Different optimizers require different optimal learning rates (Adam typically needs 10x lower rates than SGD)\n• The optimal rate depends on batch size, model architecture and dataset complexity",
      },
      batch_size: {
        name: "Batch Size",
        description: "The number of training examples processed in a single forward/backward pass. Affects both the speed of training and the quality of the model's convergence.",
        valueAnalysis: (value: string) => {
          const numValue = parseInt(value);
          if (numValue <= 8) return "Very small batch size (≤8): Provides high training noise which can help escape local minima, but results in unstable gradients and slower overall training. Best for small datasets or limited memory situations.";
          if (numValue <= 32) return "Small batch size (9-32): Good balance between computational efficiency and training noise. This range often provides good generalization performance and is the most commonly used default.";
          if (numValue <= 128) return "Medium batch size (33-128): Provides more stable gradient estimates and better utilizes hardware acceleration. May require learning rate adjustments to maintain good convergence properties.";
          if (numValue <= 512) return "Large batch size (129-512): Highly efficient for hardware but may lead to poor generalization. Typically requires careful tuning of learning rates and often benefits from learning rate warmup.";
          return "Very large batch size (>512): Maximizes computational efficiency but often leads to optimization difficulties. Usually requires specialized optimization techniques like LARS or LAMB optimizers.";
        },
        bestPractices: "• Choose the largest batch size that fits in your GPU memory\n• Consider a linear scaling rule for learning rates when increasing batch size\n• Powers of 2 (16, 32, 64, 128) typically work best with hardware acceleration\n• For large batch training (>256), use specialized optimizers like LARS/LAMB\n• When fine-tuning, smaller batches often yield better results\n• Small datasets benefit from smaller batches to avoid overfitting",
        tradeoffs: "• Larger batches provide more stable gradients but may lead to poorer generalization\n• Smaller batches introduce noise that can help escape local minima\n• Larger batches make more efficient use of hardware acceleration\n• Memory requirements increase linearly with batch size\n• Training time per epoch decreases with larger batches, but may require more epochs to reach the same accuracy",
      },
      dropout: {
        name: "Dropout Rate",
        description: "A regularization technique that randomly deactivates a fraction of neurons during training, forcing the network to learn redundant representations and preventing co-adaptation of neurons.",
        valueAnalysis: (value: string) => {
          const numValue = parseFloat(value);
          if (numValue <= 0.1) return "Very low dropout (≤0.1): Minimal regularization effect. Suitable for small networks or when other regularization methods are in use. May not significantly impact overfitting.";
          if (numValue <= 0.3) return "Low dropout (0.1-0.3): Gentle regularization that preserves network capacity. Good for models that aren't severely overfitting or for deeper layers in a network.";
          if (numValue <= 0.5) return "Medium dropout (0.3-0.5): Standard range that works well for most networks. Provides significant regularization without severely limiting model capacity. The 0.5 value is a common default.";
          if (numValue <= 0.7) return "High dropout (0.5-0.7): Strong regularization for networks that are heavily overfitting. May require longer training time and higher learning rates to compensate for the reduced capacity.";
          return "Very high dropout (>0.7): Extreme regularization that severely restricts model capacity. Rarely used except in very specific cases. Networks may underfit if kept this high.";
        },
        bestPractices: "• Apply dropout only during training, not during inference\n• Higher dropout rates often require longer training time\n• Place dropout layers between dense layers, not directly before output layers\n• Consider using different dropout rates for different layers (higher for larger layers)\n• Combine with batch normalization (placing BN before dropout layer)\n• Adjust learning rate upward when using high dropout rates",
        tradeoffs: "• Higher dropout provides stronger regularization but may cause underfitting\n• Lower dropout may not sufficiently prevent overfitting\n• Dropout increases required training time as it effectively reduces network capacity\n• Specific layers may benefit from different dropout rates (input layer: 0.1-0.2, hidden layers: 0.3-0.5)\n• Alternative regularization techniques like weight decay might be more appropriate for some architectures",
      },
      epochs: {
        name: "Training Epochs",
        description: "An epoch represents one complete pass through the entire training dataset. The number of epochs controls how many times the model will see each training example during the learning process.",
        valueAnalysis: (value: string) => {
          const numValue = parseInt(value);
          if (numValue <= 5) return "Very few epochs (≤5): Suitable for transfer learning or very large datasets. The model may not have enough time to learn complex patterns, but avoids overfitting. Good for fine-tuning pre-trained models.";
          if (numValue <= 20) return "Few epochs (6-20): Common range for many tasks, especially with transfer learning or large datasets. Balances learning time with overfitting risk. Often sufficient with good initialization.";
          if (numValue <= 50) return "Medium epochs (21-50): Provides ample learning time for most models trained from scratch. Some risk of overfitting without proper regularization. Good when using techniques like early stopping.";
          if (numValue <= 100) return "Many epochs (51-100): Extended training time for complex problems or with strong regularization. Higher risk of overfitting unless using learning rate scheduling and monitoring validation metrics.";
          return "Very many epochs (>100): Extensive training typically only needed for very complex tasks, small datasets with heavy regularization, or specific architectures like self-supervised learning. High risk of overfitting or diminishing returns.";
        },
        bestPractices: "• Use early stopping with a validation set to prevent overfitting\n• Save checkpoints of best models during training\n• For transfer learning, fewer epochs (5-10) are typically sufficient\n• Monitor validation metrics throughout training\n• Consider learning rate schedules to reduce rates in later epochs\n• For small datasets, use more epochs with stronger regularization",
        tradeoffs: "• More epochs allow more learning time but increase overfitting risk\n• Fewer epochs save computational resources but may result in underfitting\n• The optimal number varies greatly based on dataset size, model complexity, and regularization\n• Some optimization techniques (like SWA) specifically benefit from extended training\n• Early stopping effectively makes the exact number less critical",
      },
      optimizer: {
        name: "Optimizer",
        description: "The algorithm used to update the network weights based on the gradient of the loss function. Different optimizers use different approaches to determine the step direction and size during training.",
        valueAnalysis: (value: string) => {
          const valueLower = value.toLowerCase();
          if (valueLower.includes('sgd')) return "SGD (Stochastic Gradient Descent): The simplest optimizer that follows the gradient direction directly. Can be very effective with proper momentum and learning rate schedules, but requires careful tuning. Often achieves best generalization performance when properly tuned.";
          if (valueLower.includes('adam')) return "Adam: Adaptive optimizer that maintains per-parameter learning rates. Converges quickly and works well across many tasks with minimal tuning. The most popular choice for many deep learning applications due to its robustness.";
          if (valueLower.includes('rmsprop')) return "RMSprop: Adaptive optimizer that handles non-stationary objectives well by using a moving average of squared gradients. Good performance on RNNs and problems with noisy gradients.";
          if (valueLower.includes('adagrad')) return "Adagrad: Adapts learning rates per-parameter, decreasing rates for frequently occurring features. Works well for sparse data but may decrease learning rates too aggressively over time.";
          return "Generic Optimizer: Each optimizer has specific strengths and weaknesses. Adaptive optimizers like Adam are easier to tune, while SGD often provides better generalization. The choice depends on your specific task, computational budget, and tuning time available.";
        },
        bestPractices: "• Adam is an excellent default choice for most applications\n• SGD+momentum often works better for CNNs with proper tuning\n• Match learning rate to the optimizer (Adam: ~0.001, SGD: ~0.01)\n• Consider switching optimizers when progress plateaus\n• For large batch training, use specialized variants like LARS or LAMB\n• Lower your learning rate when fine-tuning with any optimizer",
        tradeoffs: "• Adaptive optimizers (Adam, RMSprop) converge faster but may generalize worse\n• SGD typically requires more tuning but can achieve better final performance\n• Adam performs well across many tasks with minimal tuning\n• Different optimizers need different learning rate scales\n• Computational and memory requirements vary between optimizers\n• Some optimizers work better for specific architectures (RMSprop for RNNs, SGD for CNNs)",
      },
      activation: {
        name: "Activation Function",
        description: "Introduces non-linearity into neural networks, allowing them to learn complex patterns. Transforms the output of each neuron before passing it to the next layer.",
        valueAnalysis: (value: string) => {
          const valueLower = value.toLowerCase();
          if (valueLower.includes('relu')) return "ReLU (Rectified Linear Unit): The most widely used activation function. Simple, computationally efficient, and solves vanishing gradient problem for positive inputs. Has 'dying ReLU' problem where neurons can become permanently inactive.";
          if (valueLower.includes('leaky')) return "Leaky ReLU: Modification of ReLU that allows small negative values, preventing the dying ReLU problem. Provides all benefits of ReLU with more robustness, at minimal computational cost.";
          if (valueLower.includes('sigmoid')) return "Sigmoid: Classic activation that squashes values between 0 and 1. Prone to vanishing gradients and saturation. Useful mainly in output layers for binary classification or attention mechanisms.";
          if (valueLower.includes('tanh')) return "Tanh: Hyperbolic tangent that squashes values between -1 and 1. Similar to sigmoid but with stronger gradients and zero-centered output. Still prone to saturation but sometimes preferred for RNNs.";
          if (valueLower.includes('gelu')) return "GELU: Gaussian Error Linear Unit, used in modern architectures like Transformers. Smooth approximation to ReLU multiplied by gaussian CDF, balancing performance and gradient properties.";
          return "Specialized Activation: The choice of activation function significantly impacts network behavior. Modern practice favors ReLU and its variants (Leaky ReLU, ELU) for most hidden layers, with specialized functions for specific architectures.";
        },
        bestPractices: "• Use ReLU as the default choice for most networks\n• Consider Leaky ReLU, ELU or GELU to prevent dying neurons\n• For output layers, match activation to task (softmax for multi-class, sigmoid for binary, linear for regression)\n• Initialize weights appropriately for the chosen activation function\n• For very deep networks, consider SELU with proper initialization\n• For Transformers, GELU has become the standard choice",
        tradeoffs: "• ReLU is computationally efficient but can suffer from dead neurons\n• Sigmoid/Tanh provide bounded outputs but suffer from vanishing gradients\n• Leaky ReLU/ELU address dying neurons but add hyperparameters\n• More exotic activations (Swish, Mish) can improve performance but add complexity\n• The choice of activation affects weight initialization requirements\n• Some architectures work best with specific activations (GELU for Transformers)",
      },
      weight_decay: {
        name: "Weight Decay",
        description: "A regularization technique that adds a penalty for large weight values to the loss function, forcing the network to use smaller weights. Also known as L2 regularization.",
        valueAnalysis: (value: string) => {
          const numValue = parseFloat(value);
          if (numValue <= 0.0001) return "Very low weight decay (≤0.0001): Minimal regularization effect. Suitable when other regularization methods are sufficient or when preserving the full model capacity is important.";
          if (numValue <= 0.001) return "Low weight decay (0.0001-0.001): Gentle regularization that constrains weights without severely limiting model capacity. Good default range for many applications.";
          if (numValue <= 0.01) return "Medium weight decay (0.001-0.01): Standard range for most neural networks. Provides significant regularization while allowing the model to learn complex patterns. The value 0.01 is a common default.";
          if (numValue <= 0.1) return "High weight decay (0.01-0.1): Strong regularization for networks that are at high risk of overfitting. May limit model capacity significantly and require adjustment of other hyperparameters.";
          return "Very high weight decay (>0.1): Extreme regularization that severely restricts weights. Rarely used except in very specific cases. High risk of underfitting with such strong constraints.";
        },
        bestPractices: "• Start with a default of 0.01 for most networks\n• Add weight decay to all layers except bias terms and batch normalization parameters\n• For Adam optimizer, consider using AdamW which properly decouples weight decay\n• Increase weight decay for smaller datasets or less complex models\n• Combine with other regularization methods (dropout, data augmentation)\n• Monitor validation metrics to ensure you're not underfitting",
        tradeoffs: "• Higher weight decay provides stronger regularization but may cause underfitting\n• Lower weight decay may not sufficiently constrain model complexity\n• Weight decay affects learning rate dynamics - higher decay may require higher learning rates\n• Different layer types may benefit from different decay rates\n• L1 regularization provides sparsity while L2 (weight decay) encourages smaller weights overall\n• Adaptive optimizers like Adam require special implementations (AdamW) for proper weight decay",
      },
      kernel_size: {
        name: "Kernel Size",
        description: "In convolutional neural networks, kernel size determines the field of view (receptive field) of each filter. It defines how much of the input each neuron examines.",
        valueAnalysis: (value: string) => {
          const numValue = parseInt(value);
          if (numValue === 1) return "1×1 kernel: Used for dimensionality reduction/expansion (bottleneck architectures) or cross-channel interactions without spatial effects. No direct spatial feature extraction but very computationally efficient.";
          if (numValue <= 3) return "3×3 kernel: The most commonly used kernel size for most modern CNNs. Balances feature extraction capability with computational efficiency. Multiple stacked 3×3 layers can effectively replace larger kernels.";
          if (numValue <= 5) return "5×5 kernel: Medium-sized kernel that captures more spatial context in each layer. Offers wider receptive field than 3×3 but with higher computational cost. Often used in earlier network layers.";
          if (numValue <= 7) return "7×7 kernel: Large kernel that captures substantial spatial context. Often used in the first layer of networks like ResNet to quickly downsample and capture broad features from images.";
          return `${numValue}×${numValue} kernel: Very large kernel that captures extensive spatial relationships at significant computational cost. Rarely used except in first layers or specialized applications. Modern architectures often replace these with stacked smaller kernels.`;
        },
        bestPractices: "• Use 3×3 kernels as the default for most layers\n• Stack multiple 3×3 layers instead of using larger kernels\n• Consider larger kernels (5×5, 7×7) for the first layer\n• Use 1×1 kernels for channel-wise operations and bottlenecks\n• For modern architectures like MobileNet, consider depthwise separable convolutions\n• Balance kernel size with computational constraints and feature extraction needs",
        tradeoffs: "• Larger kernels capture more spatial context but are computationally expensive\n• Smaller kernels are more efficient but require more layers for the same receptive field\n• Stacking multiple small kernels (e.g., two 3×3 instead of one 5×5) introduces more non-linearity\n• Different kernel sizes extract features at different scales\n• Larger kernels have more parameters and may cause overfitting on smaller datasets\n• Odd-numbered kernels (3×3, 5×5) maintain spatial dimensions better than even-numbered ones",
      },
      layer: {
        name: "Hidden Layers",
        description: "The number of processing layers between input and output in a neural network. Determines the depth of the network and its capacity to learn hierarchical representations.",
        valueAnalysis: (value: string) => {
          const numValue = parseInt(value);
          if (numValue <= 2) return "Very shallow network (1-2 hidden layers): Limited capacity to learn complex features. Suitable for simple problems or as a baseline. May underfit on complex tasks but easy to train and interpret.";
          if (numValue <= 5) return "Shallow network (3-5 hidden layers): Moderate capacity that works well for many traditional machine learning tasks. Good balance between expressiveness and training difficulty.";
          if (numValue <= 10) return "Moderately deep network (6-10 hidden layers): Substantial capacity for feature learning. Appropriate for complex problems while avoiding some of the challenges of very deep networks.";
          if (numValue <= 20) return "Deep network (11-20 hidden layers): High capacity for learning complex feature hierarchies. May require techniques like residual connections, batch normalization, or careful initialization.";
          return "Very deep network (>20 hidden layers): Extreme capacity but challenging to train without architectural innovations like skip connections. Used in state-of-the-art vision and language models, but requires careful design to avoid optimization issues.";
        },
        bestPractices: "• Start with a moderate depth and increase if underfitting occurs\n• Use skip/residual connections for networks deeper than ~10 layers\n• Add batch normalization to improve training stability of deeper networks\n• Consider the problem complexity when choosing depth\n• For very deep networks, ensure gradient flow with techniques like residual connections\n• Use learning rate warmup for training very deep networks",
        tradeoffs: "• Deeper networks can learn more complex features but are harder to train\n• Shallower networks train faster and may generalize better on simpler tasks\n• Deeper networks require more data to train effectively\n• Vanishing/exploding gradients become more problematic with depth\n• Computational and memory requirements scale with depth\n• Very deep networks typically require architectural innovations (ResNets, Highway Networks)",
      },
      default: {
        name: "Hyperparameter",
        description: "A configuration setting that controls some aspect of the model's training or architecture. Finding optimal hyperparameters is crucial for model performance.",
        valueAnalysis: (value: string) => {
          return `Current value (${value}) represents a specific choice that impacts model behavior. The optimal value depends on your specific dataset, model architecture, and task requirements.`;
        },
        bestPractices: "• Use systematic hyperparameter tuning approaches\n• Consider the interactions between related hyperparameters\n• Track experiments to understand parameter sensitivity\n• Test a wide range of values on a logarithmic scale\n• Monitor validation performance to avoid overfitting\n• Consider computational constraints when selecting parameter values",
        tradeoffs: "• Most hyperparameter choices involve trade-offs between model capacity, training stability, and computational efficiency\n• Values that work well on one dataset may not transfer to others\n• Optimal values depend on model size, dataset size, and task complexity\n• Some parameters have strong interactions with others\n• Systematic search methods like grid search or Bayesian optimization can help find optimal values",
      }
    };

    // Extract the base parameter type from the parameter name
    let paramType = 'default';
    if (paramName.includes('learning_rate') || paramName.includes('lr')) paramType = 'learning_rate';
    else if (paramName.includes('batch_size')) paramType = 'batch_size';
    else if (paramName.includes('dropout')) paramType = 'dropout';
    else if (paramName.includes('epoch')) paramType = 'epochs';
    else if (paramName.includes('optimizer')) paramType = 'optimizer';
    else if (paramName.includes('activation')) paramType = 'activation';
    else if (paramName.includes('weight_decay') || paramName.includes('l2') || paramName.includes('regularization')) paramType = 'weight_decay';
    else if (paramName.includes('kernel')) paramType = 'kernel_size';
    else if (paramName.includes('layer') || paramName.includes('depth')) paramType = 'layer';

    return {
      ...parameterInfo[paramType],
      valueAnalysis: parameterInfo[paramType].valueAnalysis(paramValue)
    };
  };

  // Get descriptions for different visualization types
  const getVisualizationDescription = (type: string) => {
    switch (type) {
      case 'flowField':
        return "Simulation of weight updates as particles flowing through a gradient field. Learning rate controls particle speed and direction, showing how parameters navigate the loss landscape.";
      case 'dropout':
        return "Interactive demonstration of how dropout randomly deactivates neurons during training. This prevents co-adaptation of neurons and creates an ensemble effect, reducing overfitting.";
      case 'batchGroups':
        return "Visualization of mini-batch processing showing how samples are grouped for parallel computation. Illustrates the trade-off between batch size, memory usage, and optimization dynamics.";
      case 'momentum':
        return "Dynamic visualization of gradient-based optimization with momentum. Particles show how momentum accumulates velocity in consistent directions, helping overcome local minima and plateaus.";
      case 'weightShrinkage':
        return "Demonstrates how weight decay/regularization penalizes large weight values. The visualization shows weight constraints forcing simpler decision boundaries to prevent overfitting.";
      case 'trainingProgress':
        return "Time-series visualization of training progression across epochs. Shows how model performance improves over time, with potential overfitting effects in later stages.";
      case 'activationFunction':
        return "Interactive function plot showing how activation functions transform input signals. Demonstrates non-linearity properties, gradients, and the different behaviors of various activation types.";
      case 'optimizerPath':
        return "Comparison of optimization trajectories for different optimizers. Shows how adaptive methods like Adam differ from SGD in their approach to navigating the loss landscape.";
      case 'networkArchitecture':
        return "Structural visualization of neural network layer organization. Shows how depth and width affect model capacity, with attention to information flow through the network.";
      case 'standard':
      default:
        return "Classical neural network visualization showing neurons, connections, and signal propagation. This foundational view illustrates the basic structure and information flow in neural networks.";
    }
  };
  
  return (
    <Card className="w-full mt-4">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex justify-between items-center">
          <span>Neural Network Visualization</span>
          <div className="flex gap-2">
            <Select value={networkType} onValueChange={setNetworkType}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Network Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dense">Dense Network</SelectItem>
                <SelectItem value="cnn">Convolutional</SelectItem>
              </SelectContent>
            </Select>
            
            <Select value={visualizationType} onValueChange={setVisualizationType}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Visualization Type" />
              </SelectTrigger>
              <SelectContent>
                {/* Show parameter-specific visualization first */}
                <SelectItem value={defaultVisualizationType}>
                  {defaultVisualizationType.charAt(0).toUpperCase() + defaultVisualizationType.slice(1)} (Recommended)
                </SelectItem>
                
                {/* Then show other available visualizations */}
                {['standard', 'flowField', 'dropout', 'batchGroups', 'momentum', 
                  'weightShrinkage', 'trainingProgress', 'activationFunction',
                  'optimizerPath', 'networkArchitecture'].map(type => {
                  if (type !== defaultVisualizationType) {
                    return (
                      <SelectItem key={type} value={type}>
                        {type.charAt(0).toUpperCase() + type.slice(1)}
                      </SelectItem>
                    );
                  }
                  return null;
                })}
              </SelectContent>
            </Select>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full h-64 relative border border-gray-200 rounded-md">
          <canvas 
            ref={canvasRef} 
            className="w-full h-full" 
          />
        </div>
        
        <div className="mt-6 space-y-4">
          <div className="flex justify-between items-center">
            <Label htmlFor="value-slider" className="text-sm">
              {paramName}: <span className="font-mono">{selectedValue.toFixed(4)}</span>
            </Label>
            <Button 
              size="sm" 
              variant={showAnimation ? "default" : "outline"} 
              onClick={toggleAnimation}
            >
              {showAnimation ? "Pause Animation" : "Play Animation"}
            </Button>
          </div>
          <Slider
            id="value-slider"
            min={range.min}
            max={range.max}
            step={range.step}
            value={[selectedValue]}
            onValueChange={handleValueChange}
            className="w-full"
          />
        </div>
        
        <div className="mt-4 text-sm text-gray-600">
          <div className="bg-blue-50 p-3 rounded-md border border-blue-100">
            <p className="font-medium mb-1">
              Visualization: {visualizationType.charAt(0).toUpperCase() + visualizationType.slice(1)}
            </p>
            <p>{getVisualizationDescription(visualizationType)}</p>
          </div>
          
          {/* Get detailed parameter information */}
          {(() => {
            const paramInfo = getParameterInfo();
            return (
              <div className="mt-4 space-y-4">
                {/* Parameter Description */}
                <div className="bg-white p-3 rounded-md border border-gray-200">
                  <p className="font-medium text-gray-900 mb-1">{paramInfo.name}</p>
                  <p className="text-gray-700 mb-3">{paramInfo.description}</p>
                  
                  {/* Value Analysis */}
                  <div className="mt-3 pb-3 border-b border-gray-100">
                    <p className="font-medium text-gray-900 mb-1 text-sm">Current Value Analysis</p>
                    <p className="text-gray-700 text-sm">{paramInfo.valueAnalysis}</p>
                  </div>
                  
                  {/* Best Practices */}
                  <div className="mt-3 pb-3 border-b border-gray-100">
                    <p className="font-medium text-gray-900 mb-1 text-sm">Best Practices</p>
                    <div className="pl-1">
                      {paramInfo.bestPractices.split('\n').map((practice, index) => (
                        <div key={index} className="text-gray-700 text-xs mb-1 flex">
                          <span className="flex-shrink-0">{practice}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Trade-offs */}
                  <div className="mt-3">
                    <p className="font-medium text-gray-900 mb-1 text-sm">Trade-offs</p>
                    <div className="pl-1">
                      {paramInfo.tradeoffs.split('\n').map((tradeoff, index) => (
                        <div key={index} className="text-gray-700 text-xs mb-1 flex">
                          <span className="flex-shrink-0">{tradeoff}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      </CardContent>
    </Card>
  );
}

// Helper function to draw a neural network
function drawNeuralNetwork(
  ctx: CanvasRenderingContext2D, 
  width: number,
  height: number,
  params: any,
  networkType: string = 'dense',
  animate: boolean = false,
  time: number = 0
) {
  const { 
    layerCount, 
    neuronsPerLayer, 
    connectionOpacity, 
    connectionThickness,
    neuronSize = 10,
    layerSpacing,
    dropoutRate = 0,
    visualizationType = 'standard',
    customSettings = {},
    normalizedValue = 0.5
  } = params;
  
  // Colors
  const activeNeuronColor = '#4F46E5'; // Indigo
  const outputNeuronColor = '#EF4444'; // Red
  const inactiveNeuronColor = 'rgba(209, 213, 219, 0.5)'; // Light gray, semi-transparent
  
  // First handle special visualization types that don't use the standard network layout
  if (visualizationType === 'flowField' && animate) {
    // Learning rate visualization as a flow field
    drawFlowFieldVisualization(ctx, width, height, time, customSettings);
    return;
  } else if (visualizationType === 'activationFunction') {
    // Activation function visualization
    drawActivationFunctionVisualization(ctx, width, height, time, customSettings);
    return;
  } else if (visualizationType === 'optimizerPath' && animate) {
    // Optimizer path visualization (only in animation mode)
    drawOptimizerPathVisualization(ctx, width, height, time, customSettings);
    return;
  }
  
  // Calculate neuron positions for each layer
  const layers: Array<Array<{x: number, y: number, active: boolean, isOutput: boolean, id: number}>> = [];
  
  for (let l = 0; l < layerCount; l++) {
    const layer: Array<{x: number, y: number, active: boolean, isOutput: boolean, id: number}> = [];
    const neuronsInThisLayer = neuronsPerLayer[l] || 5;
    const isOutputLayer = l === layerCount - 1;
    
    // X position is based on layer index
    const x = (l + 1) * (width / (layerCount + 1));
    
    // Calculate neuron positions for this layer
    for (let n = 0; n < neuronsInThisLayer; n++) {
      // Y position distributes neurons evenly
      const y = height * 0.1 + (n * (height * 0.8) / (neuronsInThisLayer - 1 || 1));
      
      // Determine if neuron is active (apply dropout to hidden layers only)
      let active = true;
      if (visualizationType === 'dropout') {
        // Special dropout visualization - more deterministic pattern
        const shouldApplyDropout = l > 0 && l < layerCount - 1;
        if (shouldApplyDropout) {
          const dropoutPattern = customSettings.dropoutPattern || 'random';
          if (dropoutPattern === 'random') {
            active = Math.random() > dropoutRate;
          } else if (dropoutPattern === 'heavy') {
            // Heavier dropout on bottom neurons
            active = (n / neuronsInThisLayer) > dropoutRate;
          } else if (dropoutPattern === 'light') {
            // Lighter dropout on top neurons
            active = (1 - n / neuronsInThisLayer) > dropoutRate;
          }
        }
      } else {
        // Standard dropout
        const shouldApplyDropout = l > 0 && l < layerCount - 1;
        active = !shouldApplyDropout || Math.random() > dropoutRate;
      }
      
      layer.push({ x, y, active, isOutput: isOutputLayer, id: n });
    }
    
    layers.push(layer);
  }
  
  // Draw special background effects for some visualization types
  if (visualizationType === 'weightShrinkage') {
    // Draw weight magnitude background
    drawWeightShrinkageBackground(ctx, width, height, customSettings);
  } else if (visualizationType === 'trainingProgress') {
    // Draw training progress background
    drawTrainingProgressBackground(ctx, width, height, customSettings);
  } else if (visualizationType === 'batchGroups' && animate) {
    // Draw batch grouping background
    drawBatchGroupsBackground(ctx, width, height, time, customSettings);
  }
  
  // Draw connections before neurons so they appear behind
  for (let l = 0; l < layers.length - 1; l++) {
    const fromLayer = layers[l];
    const toLayer = layers[l + 1];
    
    // Different connection patterns based on network type
    if (networkType === 'dense') {
      // Fully connected: each neuron connects to all neurons in next layer
      for (const fromNeuron of fromLayer) {
        // Skip inactive neurons for drawing connections
        if (!fromNeuron.active) continue;
        
        for (const toNeuron of toLayer) {
          // Skip inactive neurons for drawing connections
          if (!toNeuron.active) continue;
          
          // Animation effect - pulse the connections with a slow, educational pace
          let opacity = connectionOpacity;
          let lineWidth = connectionThickness;
          let strokeStyle = `rgba(79, 70, 229, ${opacity})`;
          
          // Specialized connection styles based on visualization type
          if (visualizationType === 'momentum' && animate) {
            // Momentum visualization - flowing particles along connections
            drawMomentumConnection(ctx, fromNeuron, toNeuron, time, customSettings);
            continue; // Skip standard line drawing
          } else if (visualizationType === 'weightShrinkage') {
            // Weight decay visualization - thinner, variable connections
            const variability = customSettings.weightVariability || 0.3;
            const shrinkFactor = customSettings.shrinkFactor || 0.5;
            
            // Create variable weight connections
            const weightNoise = 1 - (shrinkFactor * 0.5) - (Math.random() * variability);
            lineWidth = connectionThickness * weightNoise;
            opacity = connectionOpacity * weightNoise;
            strokeStyle = `rgba(79, 70, 229, ${opacity})`;
          } else if (animate) {
            // Standard animation effect
            const distance = Math.sqrt(
              Math.pow(toNeuron.x - fromNeuron.x, 2) + 
              Math.pow(toNeuron.y - fromNeuron.y, 2)
            );
            
            // Create a slow wave effect based on distance and time
            const phase = (time * 0.5 + distance / 200) % 1;
            
            // Sinusoidal pulsing effect
            opacity = connectionOpacity * (0.3 + 0.7 * Math.sin(phase * Math.PI));
            strokeStyle = `rgba(79, 70, 229, ${opacity})`;
          }
          
          // Draw standard connection line
          ctx.beginPath();
          ctx.moveTo(fromNeuron.x, fromNeuron.y);
          ctx.lineTo(toNeuron.x, toNeuron.y);
          ctx.strokeStyle = strokeStyle;
          ctx.lineWidth = lineWidth;
          ctx.stroke();
        }
      }
    } else if (networkType === 'cnn') {
      // CNN: each neuron connects only to nearby neurons (simulating local receptive fields)
      const kernelSize = 3; // Simulate a 3x3 kernel
      
      for (let i = 0; i < fromLayer.length; i++) {
        const fromNeuron = fromLayer[i];
        if (!fromNeuron.active) continue;
        
        // Connect to nearby neurons in next layer (simulating convolution)
        for (let j = 0; j < toLayer.length; j++) {
          const toNeuron = toLayer[j];
          if (!toNeuron.active) continue;
          
          // Only connect if neurons are within "kernel" range
          const relativeIdx = i / fromLayer.length - j / toLayer.length;
          if (Math.abs(relativeIdx) <= kernelSize / 10) {
            let opacity = connectionOpacity;
            let lineWidth = connectionThickness;
            
            if (animate) {
              // Subtle wave effect for CNN
              const phase = (time * 0.3 + i / fromLayer.length) % 1;
              opacity = connectionOpacity * (0.3 + 0.7 * Math.sin(phase * Math.PI));
            }
            
            ctx.beginPath();
            ctx.moveTo(fromNeuron.x, fromNeuron.y);
            ctx.lineTo(toNeuron.x, toNeuron.y);
            ctx.strokeStyle = `rgba(79, 70, 229, ${opacity})`;
            ctx.lineWidth = lineWidth;
            ctx.stroke();
          }
        }
      }
    }
  }
  
  // Draw all neurons on top of connections
  for (let l = 0; l < layers.length; l++) {
    for (const neuron of layers[l]) {
      // Different appearance for active/inactive neurons
      if (neuron.active) {
        // Active neuron with specialized styling based on visualization type
        let neuronColor = neuron.isOutput ? outputNeuronColor : activeNeuronColor;
        let effectiveNeuronSize = neuronSize;
        
        if (visualizationType === 'networkArchitecture') {
          // For network architecture, color neurons by layer
          const layerProgress = l / (layerCount - 1);
          neuronColor = `hsl(${210 + layerProgress * 150}, 80%, 60%)`;
        } else if (visualizationType === 'batchGroups') {
          // For batch visualization, color neurons by batch
          const batchIndex = (neuron.id % (customSettings.batchSize || 4)) / (customSettings.batchSize || 4);
          neuronColor = `hsl(${batchIndex * 360}, 80%, 60%)`;
          
          // Pulse size based on animation
          if (animate) {
            const pulseFactor = 0.8 + 0.4 * Math.sin((time * 2 + neuron.id * 0.2) * Math.PI);
            effectiveNeuronSize = neuronSize * pulseFactor;
          }
        }
        
        // Draw active neuron
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, effectiveNeuronSize, 0, Math.PI * 2);
        ctx.fillStyle = neuronColor;
        ctx.fill();
        
        // For some visualizations, add special effects to neurons
        if (visualizationType === 'trainingProgress' && animate) {
          // Add training progress glow effect
          const glowOpacity = 0.3 + 0.7 * Math.sin(time * Math.PI * 2);
          ctx.beginPath();
          ctx.arc(neuron.x, neuron.y, effectiveNeuronSize * 1.8, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(79, 70, 229, ${glowOpacity * 0.3})`;
          ctx.fill();
        }
      } else {
        // Inactive neuron (dropout) - draw as hollow/transparent
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
  
  // Draw layer labels at the top
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#6b7280';
  
  // Draw layer names
  const layerNames = ['Input', ...Array(layerCount - 2).fill(0).map((_, i) => `Hidden ${i+1}`), 'Output'];
  for (let l = 0; l < layerCount; l++) {
    if (layers[l].length > 0) {
      ctx.fillText(layerNames[l], layers[l][0].x, 20);
    }
  }
}

// Helper function to draw momentum-based connections with particle flow
function drawMomentumConnection(
  ctx: CanvasRenderingContext2D,
  fromNeuron: any,
  toNeuron: any,
  time: number,
  settings: any
) {
  const trailLength = settings.trailLength || 10;
  const particleInertia = settings.particleInertia || 0.5;
  
  // Base connection
  ctx.beginPath();
  ctx.moveTo(fromNeuron.x, fromNeuron.y);
  ctx.lineTo(toNeuron.x, toNeuron.y);
  ctx.strokeStyle = `rgba(79, 70, 229, 0.2)`;
  ctx.lineWidth = 1;
  ctx.stroke();
  
  // Calculate direction vector
  const dx = toNeuron.x - fromNeuron.x;
  const dy = toNeuron.y - fromNeuron.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  
  // Draw flowing particles
  const particleCount = Math.max(3, Math.round(length / 15));
  
  for (let i = 0; i < particleCount; i++) {
    // Position along line, with momentum effect
    const basePos = (time * (0.3 + particleInertia * 0.5) + i / particleCount) % 1;
    const pos = easeInOutCubic(basePos); // Momentum-like easing
    
    const x = fromNeuron.x + dx * pos;
    const y = fromNeuron.y + dy * pos;
    
    // Particle size based on position (larger in middle)
    const sizeFactor = 1 - 2 * Math.abs(pos - 0.5);
    const particleSize = 2 + 3 * sizeFactor;
    
    // Draw particle
    ctx.beginPath();
    ctx.arc(x, y, particleSize, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(79, 70, 229, ${0.5 + 0.5 * sizeFactor})`;
    ctx.fill();
    
    // Draw trail based on momentum
    if (trailLength > 0) {
      const trailSteps = Math.min(pos * 20, trailLength);
      for (let t = 1; t <= trailSteps; t++) {
        const trailPos = Math.max(0, pos - t * 0.02 * particleInertia);
        const trailX = fromNeuron.x + dx * trailPos;
        const trailY = fromNeuron.y + dy * trailPos;
        const trailOpacity = (1 - t / trailSteps) * 0.4;
        const trailSize = particleSize * (1 - t / trailSteps);
        
        ctx.beginPath();
        ctx.arc(trailX, trailY, trailSize, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(79, 70, 229, ${trailOpacity})`;
        ctx.fill();
      }
    }
  }
}

// Helper function to draw flow field visualization (for learning rate)
function drawFlowFieldVisualization(
  ctx: CanvasRenderingContext2D,
  width: number, 
  height: number,
  time: number,
  settings: any
) {
  const particleCount = settings.particleCount || 150;
  const particleSpeed = settings.particleSpeed || 0.5;
  const flowIntensity = settings.flowIntensity || 3.0;
  
  // Draw a header for the visualization
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#6b7280';
  ctx.fillText('Learning Rate Flow Visualization', width / 2, 20);
  
  // Semi-transparent background
  ctx.fillStyle = 'rgba(249, 250, 251, 0.7)';
  ctx.fillRect(0, 0, width, height);
  
  // Create particles if they don't exist
  if (!ctx.canvas._particles) {
    ctx.canvas._particles = Array(particleCount).fill(0).map(() => ({
      x: Math.random() * width,
      y: Math.random() * height,
      size: 1 + Math.random() * 3,
      vx: 0,
      vy: 0,
      color: `hsl(${Math.random() * 60 + 200}, 80%, 60%)`
    }));
  }
  
  // Update and draw particles
  for (const p of ctx.canvas._particles) {
    // Flow field influence (based on position and time)
    const noiseX = time * 0.1 + p.x / width * 5;
    const noiseY = time * 0.1 + p.y / height * 5;
    
    // Simple but effective flow field based on sine waves
    const fx = Math.sin(noiseX) * Math.cos(noiseY * 2) * flowIntensity;
    const fy = Math.cos(noiseX * 2) * Math.sin(noiseY) * flowIntensity;
    
    // Update velocity with inertia
    p.vx = p.vx * 0.9 + fx * 0.1 * particleSpeed;
    p.vy = p.vy * 0.9 + fy * 0.1 * particleSpeed;
    
    // Update position
    p.x += p.vx;
    p.y += p.vy;
    
    // Wrap around edges
    if (p.x < 0) p.x += width;
    if (p.x > width) p.x -= width;
    if (p.y < 0) p.y += height;
    if (p.y > height) p.y -= height;
    
    // Draw particle
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
    ctx.fillStyle = p.color;
    ctx.fill();
    
    // Draw trail
    if (p.vx !== 0 || p.vy !== 0) {
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x - p.vx * 5, p.y - p.vy * 5);
      ctx.strokeStyle = p.color.replace(')', ', 0.3)').replace('rgb', 'rgba');
      ctx.lineWidth = p.size / 2;
      ctx.stroke();
    }
  }
  
  // Draw learning rate explanation
  const speedText = particleSpeed < 0.3 ? 'Slow & Stable' : 
                   particleSpeed > 0.7 ? 'Fast & Potentially Unstable' : 
                   'Balanced';
  
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#374151';
  ctx.fillText(`Particle Motion: ${speedText}`, width / 2, height - 20);
  
  // Draw a basic legend
  drawLegend(ctx, [
    { label: 'Particles', color: '#4F46E5', description: 'Parameter Updates' },
    { label: 'Flow Field', color: '#9CA3AF', description: 'Loss Landscape' },
    { label: 'Speed', color: '#EF4444', description: 'Learning Rate' }
  ], width, 50);
}

// Helper function to draw activation function visualization
function drawActivationFunctionVisualization(
  ctx: CanvasRenderingContext2D,
  width: number, 
  height: number,
  time: number,
  settings: any
) {
  const activationType = settings.activationType || 'relu';
  const activationStrength = settings.activationStrength || 0.5;
  
  // Draw a header for the visualization
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#6b7280';
  ctx.fillText(`${activationType.toUpperCase()} Activation Function`, width / 2, 20);
  
  // Setup coordinate system
  const padding = 40;
  const graphWidth = width - padding * 2;
  const graphHeight = height - padding * 2;
  const originX = padding + graphWidth / 2;
  const originY = padding + graphHeight / 2;
  
  // Draw axes
  ctx.beginPath();
  ctx.moveTo(padding, originY);
  ctx.lineTo(padding + graphWidth, originY);
  ctx.moveTo(originX, padding);
  ctx.lineTo(originX, padding + graphHeight);
  ctx.strokeStyle = '#9CA3AF';
  ctx.lineWidth = 1;
  ctx.stroke();
  
  // Draw axes labels
  ctx.font = '12px sans-serif';
  ctx.fillStyle = '#6b7280';
  ctx.textAlign = 'center';
  ctx.fillText('Input', padding + graphWidth, originY - 5);
  ctx.textAlign = 'right';
  ctx.fillText('Output', originX - 5, padding);
  
  // Function to map x value to screen coordinate
  const mapX = (x: number) => originX + x * (graphWidth / 4);
  const mapY = (y: number) => originY - y * (graphHeight / 4);
  
  // Draw the activation function
  ctx.beginPath();
  
  // Generate different activation functions
  const activationFunction = (x: number) => {
    switch (activationType) {
      case 'relu':
        return Math.max(0, x * activationStrength);
      case 'sigmoid':
        return 1 / (1 + Math.exp(-x * activationStrength * 3));
      case 'tanh':
        return Math.tanh(x * activationStrength * 2);
      case 'linear':
        return x * activationStrength;
      default:
        return Math.max(0, x * activationStrength); // default to ReLU
    }
  };
  
  // Draw the function curve
  const step = 1;
  for (let x = -graphWidth / 2; x <= graphWidth / 2; x += step) {
    const screenX = mapX(x / (graphWidth / 4));
    const screenY = mapY(activationFunction(x / (graphWidth / 4)));
    
    if (x === -graphWidth / 2) {
      ctx.moveTo(screenX, screenY);
    } else {
      ctx.lineTo(screenX, screenY);
    }
  }
  
  ctx.strokeStyle = '#4F46E5';
  ctx.lineWidth = 2;
  ctx.stroke();
  
  // Draw an animated point moving along the function
  if (time !== undefined) {
    const animX = Math.sin(time * Math.PI) * 2;
    const animY = activationFunction(animX);
    
    // Input line
    ctx.beginPath();
    ctx.moveTo(originX, mapY(animY));
    ctx.lineTo(mapX(animX), mapY(animY));
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Output line
    ctx.beginPath();
    ctx.moveTo(mapX(animX), originY);
    ctx.lineTo(mapX(animX), mapY(animY));
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Point on curve
    ctx.beginPath();
    ctx.arc(mapX(animX), mapY(animY), 5, 0, Math.PI * 2);
    ctx.fillStyle = '#EF4444';
    ctx.fill();
    
    // Input/output values
    ctx.font = '10px monospace';
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'left';
    ctx.fillText(`Input: ${animX.toFixed(2)}`, padding, height - padding / 2);
    ctx.textAlign = 'right';
    ctx.fillText(`Output: ${animY.toFixed(2)}`, width - padding, height - padding / 2);
  }
}

// Helper function to draw optimizer path visualization
function drawOptimizerPathVisualization(
  ctx: CanvasRenderingContext2D,
  width: number, 
  height: number,
  time: number,
  settings: any
) {
  const optimizerType = settings.optimizerType || 'generic';
  const optimizerEfficiency = settings.optimizerEfficiency || 0.5;
  
  // Draw a header for the visualization
  ctx.font = '14px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = '#6b7280';
  ctx.fillText(`${optimizerType.toUpperCase()} Optimization Path`, width / 2, 20);
  
  // Create a loss landscape
  const padding = 40;
  const graphWidth = width - padding * 2;
  const graphHeight = height - padding * 2;
  
  // Draw contour lines of the loss function
  drawLossContours(ctx, padding, padding, graphWidth, graphHeight);
  
  // Calculate optimizer path based on type
  if (!ctx.canvas._optimizerPath) {
    // Start path at a random point away from minimum
    const startX = padding + graphWidth * 0.75;
    const startY = padding + graphHeight * 0.25;
    
    // Create different paths based on optimizer type
    const steps = 50;
    const path = [];
    
    let currentX = startX;
    let currentY = startY;
    let vx = 0;
    let vy = 0;
    
    // Generate path to the minimum point
    const targetX = padding + graphWidth * 0.5;
    const targetY = padding + graphHeight * 0.6;
    
    for (let i = 0; i < steps; i++) {
      path.push({ x: currentX, y: currentY });
      
      // Vector towards target
      const dx = targetX - currentX;
      const dy = targetY - currentY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      // Normalize and scale gradient
      const gradientX = dist > 0 ? dx / dist : 0;
      const gradientY = dist > 0 ? dy / dist : 0;
      
      // Different optimizers have different behaviors
      if (optimizerType === 'sgd') {
        // Add some noise (stochastic)
        const noiseX = (Math.random() - 0.5) * 0.6 * (1 - optimizerEfficiency);
        const noiseY = (Math.random() - 0.5) * 0.6 * (1 - optimizerEfficiency);
        
        // Simple update
        currentX += (gradientX + noiseX) * 10 * optimizerEfficiency / steps;
        currentY += (gradientY + noiseY) * 10 * optimizerEfficiency / steps;
      } else if (optimizerType === 'adam') {
        // More direct path with adaptive step sizes
        const t = i / steps;
        const adaptiveFactor = 1 / (1 + Math.exp(-10 * (t - 0.5))); // Sigmoid to simulate adaptation
        
        // Less noise, more efficient path
        const noiseX = (Math.random() - 0.5) * 0.2 * (1 - optimizerEfficiency);
        const noiseY = (Math.random() - 0.5) * 0.2 * (1 - optimizerEfficiency);
        
        currentX += (gradientX + noiseX) * 10 * optimizerEfficiency * adaptiveFactor / steps;
        currentY += (gradientY + noiseY) * 10 * optimizerEfficiency * adaptiveFactor / steps;
      } else if (optimizerType === 'rmsprop') {
        // Somewhere between SGD and Adam
        const noiseX = (Math.random() - 0.5) * 0.4 * (1 - optimizerEfficiency);
        const noiseY = (Math.random() - 0.5) * 0.4 * (1 - optimizerEfficiency);
        
        // With some adaptation
        const adaptiveFactor = Math.min(1, i / (steps * 0.4));
        
        currentX += (gradientX + noiseX) * 10 * optimizerEfficiency * adaptiveFactor / steps;
        currentY += (gradientY + noiseY) * 10 * optimizerEfficiency * adaptiveFactor / steps;
      } else {
        // Generic optimizer - basic gradient descent with momentum
        // Update velocity with momentum
        vx = vx * 0.8 + gradientX * 0.2 * optimizerEfficiency;
        vy = vy * 0.8 + gradientY * 0.2 * optimizerEfficiency;
        
        // Update position
        currentX += vx * 10 / steps;
        currentY += vy * 10 / steps;
      }
    }
    
    ctx.canvas._optimizerPath = path;
  }
  
  // Draw the path
  const path = ctx.canvas._optimizerPath;
  
  // Calculate how much of the path to show based on time
  const pathProgress = Math.min(1, time * 2 % 2); // 0 to 1 over 0.5 time units
  const visibleSteps = Math.max(1, Math.ceil(pathProgress * path.length));
  
  // Draw path segments
  ctx.beginPath();
  ctx.moveTo(path[0].x, path[0].y);
  for (let i = 1; i < visibleSteps; i++) {
    ctx.lineTo(path[i].x, path[i].y);
  }
  ctx.strokeStyle = '#4F46E5';
  ctx.lineWidth = 2;
  ctx.stroke();
  
  // Draw current position
  if (visibleSteps > 0 && visibleSteps <= path.length) {
    const currentPos = path[visibleSteps - 1];
    ctx.beginPath();
    ctx.arc(currentPos.x, currentPos.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#EF4444';
    ctx.fill();
  }
  
  // Draw target minimum
  ctx.beginPath();
  ctx.arc(padding + graphWidth * 0.5, padding + graphHeight * 0.6, 5, 0, Math.PI * 2);
  ctx.fillStyle = '#10B981';
  ctx.fill();
  ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
  ctx.lineWidth = 1;
  ctx.stroke();
  
  // Draw legend
  drawLegend(ctx, [
    { label: 'Path', color: '#4F46E5', description: 'Optimization Trajectory' },
    { label: 'Current', color: '#EF4444', description: 'Current Parameters' },
    { label: 'Minimum', color: '#10B981', description: 'Loss Minimum' }
  ], width, height - 30);
}

// Helper function to draw weight shrinkage background (for weight decay)
function drawWeightShrinkageBackground(
  ctx: CanvasRenderingContext2D,
  width: number, 
  height: number,
  settings: any
) {
  const shrinkFactor = settings.shrinkFactor || 0.5;
  
  // Draw subtle grid lines to show coordinate system for weights
  ctx.strokeStyle = 'rgba(156, 163, 175, 0.2)';
  ctx.lineWidth = 1;
  
  // Draw grid
  const gridSize = 20;
  const shrinkOriginX = width / 2;
  const shrinkOriginY = height / 2;
  
  // Draw L2 regularization circles (weight decay)
  const maxRadius = Math.min(width, height) * 0.4;
  const numCircles = 3;
  
  for (let i = 1; i <= numCircles; i++) {
    const radius = (i / numCircles) * maxRadius;
    ctx.beginPath();
    ctx.arc(shrinkOriginX, shrinkOriginY, radius, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(79, 70, 229, ${0.1 + i * 0.05})`;
    ctx.stroke();
  }
  
  // Label the L2 norm regions
  ctx.font = '10px sans-serif';
  ctx.fillStyle = 'rgba(79, 70, 229, 0.7)';
  ctx.textAlign = 'center';
  ctx.fillText('L2 Regularization Strength', shrinkOriginX, shrinkOriginY - maxRadius - 10);
}

// Helper function to draw training progress background (for epochs)
function drawTrainingProgressBackground(
  ctx: CanvasRenderingContext2D,
  width: number, 
  height: number,
  settings: any
) {
  const currentEpoch = settings.currentEpoch || 1;
  const convergenceRate = settings.convergenceRate || 0.5;
  
  // Draw a learning curve at the bottom
  const padding = 40;
  const graphWidth = width - padding * 2;
  const graphHeight = 40;
  const graphY = height - graphHeight - 10;
  
  // Draw graph axes
  ctx.beginPath();
  ctx.moveTo(padding, graphY);
  ctx.lineTo(padding + graphWidth, graphY);
  ctx.moveTo(padding, graphY - graphHeight);
  ctx.lineTo(padding, graphY);
  ctx.strokeStyle = 'rgba(156, 163, 175, 0.4)';
  ctx.lineWidth = 1;
  ctx.stroke();
  
  // Draw axis labels
  ctx.font = '10px sans-serif';
  ctx.fillStyle = 'rgba(107, 114, 128, 0.7)';
  ctx.textAlign = 'center';
  ctx.fillText('Epochs', padding + graphWidth / 2, graphY + 12);
  ctx.textAlign = 'right';
  ctx.fillText('Loss', padding - 2, graphY - graphHeight / 2);
  
  // Draw learning curve
  ctx.beginPath();
  for (let i = 0; i <= graphWidth; i++) {
    const progress = i / graphWidth;
    const x = padding + i;
    // Exponential decay curve based on convergence rate
    const y = graphY - graphHeight * Math.exp(-progress * 5 * convergenceRate);
    
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.strokeStyle = 'rgba(79, 70, 229, 0.5)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  
  // Mark current epoch
  const epochX = padding + (graphWidth * currentEpoch / 100);
  if (epochX <= padding + graphWidth) {
    const progress = epochX / (padding + graphWidth);
    const epochY = graphY - graphHeight * Math.exp(-progress * 5 * convergenceRate);
    
    ctx.beginPath();
    ctx.arc(epochX, epochY, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#EF4444';
    ctx.fill();
    
    ctx.beginPath();
    ctx.moveTo(epochX, graphY);
    ctx.lineTo(epochX, graphY - graphHeight);
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Show epoch number
    ctx.font = '10px sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.fillText(`Epoch ${currentEpoch}`, epochX, graphY + 25);
  }
}

// Helper function to draw batch groups background
function drawBatchGroupsBackground(
  ctx: CanvasRenderingContext2D,
  width: number, 
  height: number,
  time: number,
  settings: any
) {
  const batchSize = settings.batchSize || 32;
  const batchColor = settings.batchColor || 'hsl(210, 80%, 60%)';
  
  // Show data flowing into batches
  const padding = 20;
  const sampleSize = 4;
  const rows = Math.floor((height - padding * 2) / (sampleSize * 2));
  const cols = Math.floor(batchSize / rows) + 1;
  
  // Draw data samples organized into batches
  const batchWidth = (width - padding * 2) / cols;
  const totalBatches = 3; // Show 3 batches
  
  for (let b = 0; b < totalBatches; b++) {
    // Batch position with animation
    const batchOffset = (time * 0.5 + b / totalBatches) % 1;
    const batchX = padding + (width - padding * 2) * batchOffset;
    
    // Draw batch border
    ctx.strokeStyle = batchColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(
      batchX, 
      padding, 
      batchWidth, 
      height - padding * 2
    );
    
    // Draw batch label
    ctx.font = '10px sans-serif';
    ctx.fillStyle = batchColor;
    ctx.textAlign = 'center';
    ctx.fillText(`Batch ${b+1}`, batchX + batchWidth / 2, padding - 5);
    
    // Draw data points inside batch
    let sampleIndex = 0;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < Math.floor(batchSize / rows); col++) {
        if (sampleIndex >= batchSize) break;
        
        const x = batchX + col * (batchWidth / (batchSize / rows));
        const y = padding + row * ((height - padding * 2) / rows);
        
        // Draw sample point
        ctx.beginPath();
        ctx.arc(
          x + (batchWidth / (batchSize / rows)) / 2, 
          y + ((height - padding * 2) / rows) / 2, 
          sampleSize / 2, 
          0, 
          Math.PI * 2
        );
        ctx.fillStyle = `${batchColor.replace(')', ', 0.5)')}`;
        ctx.fill();
        
        sampleIndex++;
      }
    }
  }
}

// Helper function to draw contour lines for loss landscape
function drawLossContours(
  ctx: CanvasRenderingContext2D, 
  x: number, 
  y: number, 
  width: number, 
  height: number
) {
  // Simple function to generate a loss landscape value at x,y
  const lossFunction = (px: number, py: number) => {
    // Normalized coordinates (-1 to 1)
    const nx = (px - (x + width / 2)) / (width / 2);
    const ny = (py - (y + height / 2)) / (height / 2);
    
    // Simple bowl-shaped loss with some asymmetry
    return 0.5 * (nx * nx + 1.5 * ny * ny) + 0.1 * Math.sin(nx * 5) * Math.sin(ny * 5);
  };
  
  // Draw contour lines
  const contours = 8;
  
  for (let c = 1; c <= contours; c++) {
    const level = c / contours;
    
    // Draw a contour line by scanning the space
    const step = 5;
    let pathStarted = false;
    
    for (let scanY = 0; scanY < height; scanY += step) {
      let lastAbove = false;
      
      for (let scanX = 0; scanX < width; scanX += step) {
        const px = x + scanX;
        const py = y + scanY;
        const value = lossFunction(px, py);
        const above = value > level;
        
        // Check for level crossing
        if (scanX > 0 && above !== lastAbove) {
          // Found a point on the contour
          const ratio = (level - lossFunction(px - step, py)) / 
                        (lossFunction(px, py) - lossFunction(px - step, py));
          const cx = px - step + step * ratio;
          
          if (!pathStarted) {
            ctx.beginPath();
            ctx.moveTo(cx, py);
            pathStarted = true;
          } else {
            ctx.lineTo(cx, py);
          }
        }
        
        lastAbove = above;
      }
    }
    
    if (pathStarted) {
      ctx.strokeStyle = `rgba(156, 163, 175, ${0.3 - 0.2 * (c / contours)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
}

// Helper function to draw a legend
function drawLegend(
  ctx: CanvasRenderingContext2D,
  items: Array<{label: string, color: string, description: string}>,
  width: number,
  y: number
) {
  const padding = 10;
  const itemWidth = (width - padding * 2) / items.length;
  
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    const x = padding + i * itemWidth;
    
    // Draw color box
    ctx.fillStyle = item.color;
    ctx.fillRect(x, y - 8, 8, 8);
    
    // Draw label
    ctx.font = '10px sans-serif';
    ctx.fillStyle = '#374151';
    ctx.textAlign = 'left';
    ctx.fillText(`${item.label}: ${item.description}`, x + 12, y);
  }
}

// Easing function for momentum simulation
function easeInOutCubic(x: number): number {
  return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
}