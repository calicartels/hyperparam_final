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
  
  // Get parameter range
  const getParameterRange = (): { min: number; max: number; step: number; defaultValue: number } => {
    if (paramName.includes('learning_rate')) {
      return { min: 0.0001, max: 0.01, step: 0.0001, defaultValue: 0.001 };
    } else if (paramName.includes('dropout')) {
      return { min: 0, max: 0.9, step: 0.05, defaultValue: 0.5 };
    } else if (paramName.includes('batch_size')) {
      // Represent batch size as a continuous param for the slider
      return { min: 1, max: 512, step: 1, defaultValue: 32 };
    } else if (paramName.includes('momentum')) {
      return { min: 0, max: 0.99, step: 0.01, defaultValue: 0.9 };
    } else if (paramName.includes('weight_decay')) {
      return { min: 0, max: 0.1, step: 0.001, defaultValue: 0.01 };
    } else if (paramName.includes('epochs')) {
      return { min: 1, max: 100, step: 1, defaultValue: 10 };
    } else {
      // Default range
      return { min: 0, max: 1, step: 0.01, defaultValue: 0.5 };
    }
  };
  
  // Get range and set initial value
  const range = getParameterRange();
  const [selectedValue, setSelectedValue] = useState<number>(
    paramValue ? parseFloat(paramValue) : range.defaultValue
  );
  
  // Handle canvas setup - only ONCE on mount
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const setupCanvas = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      // Handle high-DPI displays properly
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      
      // Set the canvas attributes properly for high DPI displays
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        // Scale the context to compensate for high-DPI
        ctx.scale(dpr, dpr);
      }
    };
    
    // Set up canvas after a small delay to ensure dimensions are correct
    setTimeout(setupCanvas, 100);
  }, []);
  
  // Convert parameter value to network parameters
  const getNetworkParams = () => {
    if (!canvasRef.current) return null;
    
    const canvas = canvasRef.current;
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    
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
    };
    
    // Parameter-specific effects
    if (paramName.includes('learning_rate')) {
      // Learning rate affects connection strength/opacity
      const normalizedLR = (selectedValue - range.min) / (range.max - range.min);
      params.connectionOpacity = 0.2 + normalizedLR * 0.6;
      params.connectionThickness = 0.5 + normalizedLR * 1.5;
    } else if (paramName.includes('dropout')) {
      // Dropout directly affects dropout rate
      params.dropoutRate = selectedValue;
    } else if (paramName.includes('batch_size')) {
      // Batch size affects number of neurons
      const normalizedBS = (selectedValue - range.min) / (range.max - range.min);
      const baseNeurons = 3;
      const maxNeurons = 10;
      const neuronCount = Math.round(baseNeurons + normalizedBS * (maxNeurons - baseNeurons));
      params.neuronsPerLayer = [neuronCount, neuronCount + 2, neuronCount + 2, Math.max(2, Math.round(neuronCount / 2))];
    } else if (paramName.includes('momentum')) {
      // Momentum affects connection thickness
      const normalizedMomentum = (selectedValue - range.min) / (range.max - range.min);
      params.connectionThickness = 0.5 + normalizedMomentum * 2.5;
    } else if (paramName.includes('weight_decay')) {
      // Weight decay inversely affects connection strength
      const normalizedWD = (selectedValue - range.min) / (range.max - range.min);
      params.connectionOpacity = 0.5 - normalizedWD * 0.3;
    } else if (paramName.includes('epochs')) {
      // More epochs = more complex network
      const normalizedEpochs = (selectedValue - range.min) / (range.max - range.min);
      const minLayers = 3;
      const maxLayers = 6;
      params.layerCount = Math.round(minLayers + normalizedEpochs * (maxLayers - minLayers));
      params.neuronsPerLayer = Array(params.layerCount).fill(0).map((_, i) => {
        if (i === 0) return 5; // Input layer
        if (i === params.layerCount - 1) return 3; // Output layer
        return 7; // Hidden layers
      });
    }
    
    return params;
  };
  
  // Animation and drawing effect
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear any existing animation
    if (animationRef.current) {
      clearTimeout(animationRef.current);
      animationRef.current = null;
    }
    
    // Account for device pixel ratio
    const dpr = window.devicePixelRatio || 1;
    const displayWidth = canvas.width / dpr;
    const displayHeight = canvas.height / dpr;
    
    // Get network parameters
    const params = getNetworkParams();
    if (!params) return;
    
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
  }, [selectedValue, networkType, paramName, showAnimation]);
  
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
  
  return (
    <Card className="w-full mt-4">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex justify-between items-center">
          <span>Neural Network Architecture</span>
          <Select value={networkType} onValueChange={setNetworkType}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Network Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="dense">Dense (Fully Connected)</SelectItem>
              <SelectItem value="cnn">Convolutional (CNN)</SelectItem>
            </SelectContent>
          </Select>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full h-60 relative border border-gray-200 rounded-md">
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
          {paramName.includes('learning_rate') && (
            <>
              <p>This visualization shows how <strong>{paramName}</strong> affects signal propagation through the network:</p>
              <ul className="list-disc list-inside mt-2 text-xs space-y-1">
                <li>Higher learning rates (values like 0.01) show as brighter, more intense connections</li>
                <li>Lower learning rates (values like 0.0001) show as more subtle, controlled signal flow</li>
              </ul>
            </>
          )}
          
          {paramName.includes('dropout') && (
            <>
              <p>This visualization shows how <strong>{paramName}</strong> affects neural network training:</p>
              <ul className="list-disc list-inside mt-2 text-xs space-y-1">
                <li>Inactive neurons (transparent) demonstrate the dropout effect</li>
                <li>Higher dropout rates disable more neurons during training</li>
              </ul>
            </>
          )}
          
          {paramName.includes('batch_size') && (
            <>
              <p>This visualization shows how <strong>{paramName}</strong> affects network training:</p>
              <ul className="list-disc list-inside mt-2 text-xs space-y-1">
                <li>Larger batch sizes allow the network to process more examples at once</li>
                <li>The number of active neurons corresponds to batch size capacity</li>
              </ul>
            </>
          )}
          
          {(!paramName.includes('learning_rate') && !paramName.includes('dropout') && !paramName.includes('batch_size')) && (
            <p>This visualization shows how changing the <strong>{paramName}</strong> affects the neural network architecture and behavior.</p>
          )}
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
    dropoutRate = 0
  } = params;
  
  // Colors
  const activeNeuronColor = '#4F46E5'; // Indigo
  const outputNeuronColor = '#EF4444'; // Red
  const inactiveNeuronColor = 'rgba(209, 213, 219, 0.5)'; // Light gray, semi-transparent
  
  // Calculate neuron positions for each layer
  const layers: Array<Array<{x: number, y: number, active: boolean, isOutput: boolean}>> = [];
  
  for (let l = 0; l < layerCount; l++) {
    const layer: Array<{x: number, y: number, active: boolean, isOutput: boolean}> = [];
    const neuronsInThisLayer = neuronsPerLayer[l] || 5;
    const isOutputLayer = l === layerCount - 1;
    
    // X position is based on layer index
    const x = (l + 1) * (width / (layerCount + 1));
    
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
          if (animate) {
            // Calculate distance for wave effect
            const distance = Math.sqrt(
              Math.pow(toNeuron.x - fromNeuron.x, 2) + 
              Math.pow(toNeuron.y - fromNeuron.y, 2)
            );
            
            // Create a slow wave effect based on distance and time
            // Scale time by a small factor to slow down animation
            const phase = (time * 0.5 + distance / 200) % 1;
            
            // Sinusoidal pulsing effect
            opacity = connectionOpacity * (0.3 + 0.7 * Math.sin(phase * Math.PI));
          }
          
          // Draw connection line
          ctx.beginPath();
          ctx.moveTo(fromNeuron.x, fromNeuron.y);
          ctx.lineTo(toNeuron.x, toNeuron.y);
          ctx.strokeStyle = `rgba(79, 70, 229, ${opacity})`;
          ctx.lineWidth = connectionThickness;
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
            
            if (animate) {
              // Subtle wave effect for CNN
              const phase = (time * 0.3 + i / fromLayer.length) % 1;
              opacity = connectionOpacity * (0.3 + 0.7 * Math.sin(phase * Math.PI));
            }
            
            ctx.beginPath();
            ctx.moveTo(fromNeuron.x, fromNeuron.y);
            ctx.lineTo(toNeuron.x, toNeuron.y);
            ctx.strokeStyle = `rgba(79, 70, 229, ${opacity})`;
            ctx.lineWidth = connectionThickness;
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
        // Active neuron
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, neuronSize, 0, Math.PI * 2);
        ctx.fillStyle = neuron.isOutput ? outputNeuronColor : activeNeuronColor;
        ctx.fill();
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