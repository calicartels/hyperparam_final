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
  const [selectedValue, setSelectedValue] = useState<number>(parseFloat(paramValue) || 0);
  const [networkType, setNetworkType] = useState<string>("dense");
  const [animationFrame, setAnimationFrame] = useState<number | null>(null);
  const [showAnimation, setShowAnimation] = useState(false);
  
  const getValueRange = () => {
    if (paramName.includes('learning_rate') || paramName.includes('lr')) {
      return { min: 0.0001, max: 0.1, step: 0.0001, defaultValue: 0.001 };
    } else if (paramName.includes('batch_size')) {
      return { min: 8, max: 256, step: 8, defaultValue: 32 };
    } else if (paramName.includes('dropout')) {
      return { min: 0, max: 0.9, step: 0.1, defaultValue: 0.5 };
    } else if (paramName.includes('hidden') || paramName.includes('units') || paramName.includes('neurons')) {
      return { min: 8, max: 512, step: 8, defaultValue: 64 };
    } else if (paramName.includes('layers')) {
      return { min: 1, max: 10, step: 1, defaultValue: 3 };
    } else {
      return { min: 0, max: 1, step: 0.01, defaultValue: 0.5 };
    }
  };
  
  const range = getValueRange();
  const initialValue = parseFloat(paramValue) || range.defaultValue;
  
  useEffect(() => {
    // Initialize with the initial value from props
    setSelectedValue(initialValue);
  }, [paramValue]);

  // Draw the neural network
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Parameters for the neural network visualization
    const getNetworkParams = () => {
      if (paramName.includes('learning_rate') || paramName.includes('lr')) {
        // For learning rate, affect connection line thickness and saturation
        return {
          layerCount: 4,
          neuronsPerLayer: [5, 8, 8, 3],
          connectionOpacity: 0.3 + selectedValue * 7, // Higher learning rate = more saturated connections
          connectionThickness: 0.5 + selectedValue * 20, // Higher learning rate = thicker connections
          neuronSize: 12,
          layerSpacing: canvas.width / 5
        };
      } else if (paramName.includes('batch_size')) {
        // For batch size, affect number of visible neurons
        const batchSizeFactor = Math.min(1, selectedValue / 128);
        const neuronsPerLayerBase = [4, 6, 6, 2];
        return {
          layerCount: 4,
          neuronsPerLayer: neuronsPerLayerBase.map(n => Math.max(2, Math.round(n * (0.5 + batchSizeFactor)))),
          connectionOpacity: 0.5,
          connectionThickness: 1.5,
          neuronSize: 12,
          layerSpacing: canvas.width / 5
        };
      } else if (paramName.includes('dropout')) {
        // For dropout, show some neurons as inactive
        return {
          layerCount: 4,
          neuronsPerLayer: [5, 10, 10, 3],
          connectionOpacity: 0.5,
          connectionThickness: 1.5,
          neuronSize: 12,
          layerSpacing: canvas.width / 5,
          dropoutRate: selectedValue // Percentage of neurons to drop
        };
      } else if (paramName.includes('hidden') || paramName.includes('units') || paramName.includes('neurons')) {
        // For hidden units/neurons, affect the number of neurons in hidden layers
        const neuronsInHidden = Math.max(2, Math.round(selectedValue / 20));
        return {
          layerCount: 4,
          neuronsPerLayer: [5, neuronsInHidden, neuronsInHidden, 3],
          connectionOpacity: 0.5,
          connectionThickness: 1.5,
          neuronSize: 12,
          layerSpacing: canvas.width / 5
        };
      } else if (paramName.includes('layers')) {
        // For layer count, affect the number of layers
        const layerCount = Math.max(2, Math.round(selectedValue));
        const neuronsPerLayer = [5];
        
        // Add hidden layers
        for (let i = 0; i < layerCount; i++) {
          neuronsPerLayer.push(8);
        }
        
        // Add output layer
        neuronsPerLayer.push(3);
        
        return {
          layerCount: layerCount + 2, // input + hidden + output
          neuronsPerLayer,
          connectionOpacity: 0.5,
          connectionThickness: 1.5,
          neuronSize: 12,
          layerSpacing: canvas.width / (layerCount + 3) // Adjust spacing based on layer count
        };
      } else {
        // Default visualization
        return {
          layerCount: 4,
          neuronsPerLayer: [5, 8, 8, 3],
          connectionOpacity: 0.5,
          connectionThickness: 1.5,
          neuronSize: 12,
          layerSpacing: canvas.width / 5
        };
      }
    };
    
    const params = getNetworkParams();
    
    // For static rendering when animation is off
    if (!showAnimation) {
      drawNeuralNetwork(ctx, canvas, params, networkType, false);
      return;
    }
    
    // For animation, set up an animation loop
    let lastTimestamp = 0;
    
    const animateNetwork = (timestamp: number) => {
      // Calculate time delta
      if (!lastTimestamp) lastTimestamp = timestamp;
      const elapsed = timestamp - lastTimestamp;
      lastTimestamp = timestamp;
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw the network with animation
      drawNeuralNetwork(ctx, canvas, params, networkType, true, timestamp / 1000);
      
      // Continue animation loop
      const frame = requestAnimationFrame(animateNetwork);
      setAnimationFrame(frame);
    };
    
    // Start animation loop
    const frame = requestAnimationFrame(animateNetwork);
    setAnimationFrame(frame);
    
    // Clean up function
    return () => {
      if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [selectedValue, networkType, paramName, showAnimation]);
  
  // Clean up animation frame on unmount
  useEffect(() => {
    return () => {
      if (animationFrame !== null) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, []);
  
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
              <SelectItem value="rnn">Recurrent (RNN)</SelectItem>
              <SelectItem value="transformer">Transformer</SelectItem>
            </SelectContent>
          </Select>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="w-full h-60 relative">
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
              {showAnimation ? "Stop Animation" : "Animate"}
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
          <p>This visualization shows how changing the <strong>{paramName}</strong> affects the neural network architecture and behavior.</p>
          <p className="mt-1 text-xs text-gray-500">Note: This is a simplified representation for educational purposes.</p>
        </div>
      </CardContent>
    </Card>
  );
}

// Helper function to draw a neural network
function drawNeuralNetwork(
  ctx: CanvasRenderingContext2D, 
  canvas: HTMLCanvasElement, 
  params: any,
  networkType: string = 'dense',
  animate: boolean = false,
  customTime?: number
) {
  const { 
    layerCount, 
    neuronsPerLayer, 
    connectionOpacity, 
    connectionThickness,
    neuronSize,
    layerSpacing,
    dropoutRate = 0
  } = params;
  
  // Colors
  const activeNeuronColor = '#4F46E5';
  const inactiveNeuronColor = '#d1d5db';
  const connectionColor = `rgba(79, 70, 229, ${connectionOpacity})`;
  const activationColor = '#ef4444';
  
  // Set time variable for animation
  const time = animate ? (customTime !== undefined ? customTime : Date.now() / 1000) : 0;
  
  // Clear the canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Store neuron positions
  const neurons: {x: number, y: number, active: boolean}[][] = [];
  
  // Draw each layer
  for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
    const neuronsInLayer = neuronsPerLayer[layerIdx];
    const layerX = (layerIdx + 1) * layerSpacing;
    
    neurons[layerIdx] = [];
    
    // Draw neurons for this layer
    for (let neuronIdx = 0; neuronIdx < neuronsInLayer; neuronIdx++) {
      // Calculate vertical position with even spacing
      const layerHeight = canvas.height * 0.8;
      const margin = canvas.height * 0.1;
      const spacing = layerHeight / (neuronsInLayer - 1 || 1);
      const neuronY = neuronIdx * spacing + margin;
      
      // Determine if neuron is active (use dropout rate for hidden layers)
      const isActive = layerIdx === 0 || layerIdx === layerCount - 1 || 
                      Math.random() > dropoutRate;
      
      // Store neuron position
      neurons[layerIdx].push({
        x: layerX,
        y: neuronY,
        active: isActive
      });
    }
  }
  
  // Draw connections first (so they appear behind neurons)
  for (let layerIdx = 0; layerIdx < layerCount - 1; layerIdx++) {
    const fromLayer = neurons[layerIdx];
    const toLayer = neurons[layerIdx + 1];
    
    // Different connection patterns based on network type
    if (networkType === 'dense') {
      // Fully connected layers
      for (const fromNeuron of fromLayer) {
        if (!fromNeuron.active) continue; // Skip inactive neurons
        
        for (const toNeuron of toLayer) {
          if (!toNeuron.active) continue; // Skip inactive neurons
          
          // Animation effect - pulse the connections
          let opacity = connectionOpacity;
          if (animate) {
            const distance = Math.sqrt(
              Math.pow(toNeuron.x - fromNeuron.x, 2) + 
              Math.pow(toNeuron.y - fromNeuron.y, 2)
            );
            const phase = (time * 3 + distance / 30) % 1;
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
    } else if (networkType === 'cnn') {
      // CNN-style local connections
      const kernelSize = 3; // Simulate a 3x3 kernel
      
      for (let i = 0; i < fromLayer.length; i++) {
        const fromNeuron = fromLayer[i];
        if (!fromNeuron.active) continue; // Skip inactive neurons
        
        // Connect to nearby neurons in next layer (simulating convolution)
        for (let j = 0; j < toLayer.length; j++) {
          const toNeuron = toLayer[j];
          if (!toNeuron.active) continue; // Skip inactive neurons
          
          // Only connect if neurons are within "kernel" range
          const verticalDistance = Math.abs(i - j * (fromLayer.length / toLayer.length));
          if (verticalDistance <= kernelSize / 2) {
            let opacity = connectionOpacity;
            if (animate) {
              const phase = (time * 2 + i / fromLayer.length) % 1;
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
    } else if (networkType === 'rnn') {
      // RNN-style with recurrent connections
      for (const fromNeuron of fromLayer) {
        if (!fromNeuron.active) continue; // Skip inactive neurons
        
        // Connect to next layer
        for (const toNeuron of toLayer) {
          if (!toNeuron.active) continue; // Skip inactive neurons
          
          let opacity = connectionOpacity;
          if (animate) {
            const phase = (time * 2) % 1;
            opacity = connectionOpacity * (0.3 + 0.7 * Math.sin(phase * Math.PI));
          }
          
          ctx.beginPath();
          ctx.moveTo(fromNeuron.x, fromNeuron.y);
          ctx.lineTo(toNeuron.x, toNeuron.y);
          ctx.strokeStyle = `rgba(79, 70, 229, ${opacity})`;
          ctx.lineWidth = connectionThickness;
          ctx.stroke();
        }
        
        // Draw recurrent connections (loops)
        if (layerIdx > 0) {
          // Draw curved recurrent connection
          ctx.beginPath();
          ctx.moveTo(fromNeuron.x, fromNeuron.y);
          
          const controlX1 = fromNeuron.x - 40;
          const controlY1 = fromNeuron.y - 20;
          const controlX2 = fromNeuron.x - 40;
          const controlY2 = fromNeuron.y + 20;
          
          ctx.bezierCurveTo(
            controlX1, controlY1,
            controlX2, controlY2,
            fromNeuron.x, fromNeuron.y
          );
          
          let opacity = connectionOpacity * 0.7;
          if (animate) {
            const phase = (time * 3 + fromNeuron.y / canvas.height) % 1;
            opacity = connectionOpacity * 0.7 * (0.3 + 0.7 * Math.sin(phase * Math.PI));
          }
          
          ctx.strokeStyle = `rgba(79, 70, 229, ${opacity})`;
          ctx.lineWidth = connectionThickness * 0.8;
          ctx.stroke();
        }
      }
    } else if (networkType === 'transformer') {
      // Transformer-style attention connections
      // First draw direct connections
      for (const fromNeuron of fromLayer) {
        if (!fromNeuron.active) continue; // Skip inactive neurons
        
        for (const toNeuron of toLayer) {
          if (!toNeuron.active) continue; // Skip inactive neurons
          
          let opacity = connectionOpacity * 0.5; // More subtle direct connections
          
          ctx.beginPath();
          ctx.moveTo(fromNeuron.x, fromNeuron.y);
          ctx.lineTo(toNeuron.x, toNeuron.y);
          ctx.strokeStyle = `rgba(79, 70, 229, ${opacity})`;
          ctx.lineWidth = connectionThickness * 0.7;
          ctx.stroke();
        }
      }
      
      // Then draw attention connections between neurons in the same layer
      if (layerIdx > 0 && layerIdx < layerCount - 1) {
        for (let i = 0; i < fromLayer.length; i++) {
          const neuronA = fromLayer[i];
          if (!neuronA.active) continue;
          
          for (let j = i + 1; j < fromLayer.length; j++) {
            const neuronB = fromLayer[j];
            if (!neuronB.active) continue;
            
            // Attention connection
            let opacity = connectionOpacity * 0.3;
            if (animate) {
              const phase = (time * 1.5 + (i + j) / fromLayer.length) % 1;
              opacity = connectionOpacity * 0.3 * (0.1 + 0.9 * Math.sin(phase * Math.PI));
            }
            
            ctx.beginPath();
            ctx.moveTo(neuronA.x, neuronA.y);
            
            // Create a curved connection
            const midX = neuronA.x - 15; // Control point X
            const midY = (neuronA.y + neuronB.y) / 2; // Control point Y
            
            ctx.quadraticCurveTo(midX, midY, neuronB.x, neuronB.y);
            
            ctx.strokeStyle = `rgba(79, 70, 229, ${opacity})`;
            ctx.lineWidth = connectionThickness * 0.5;
            ctx.stroke();
          }
        }
      }
    }
  }
  
  // Draw neurons
  for (let layerIdx = 0; layerIdx < layerCount; layerIdx++) {
    for (const neuron of neurons[layerIdx]) {
      // Skip drawing some neurons in hidden layers based on dropout rate
      if (!neuron.active) {
        // Draw inactive neuron
        ctx.beginPath();
        ctx.arc(neuron.x, neuron.y, neuronSize, 0, Math.PI * 2);
        ctx.fillStyle = inactiveNeuronColor;
        ctx.fill();
        
        // Draw a cross to indicate dropped neuron
        ctx.beginPath();
        ctx.moveTo(neuron.x - neuronSize / 2, neuron.y - neuronSize / 2);
        ctx.lineTo(neuron.x + neuronSize / 2, neuron.y + neuronSize / 2);
        ctx.moveTo(neuron.x + neuronSize / 2, neuron.y - neuronSize / 2);
        ctx.lineTo(neuron.x - neuronSize / 2, neuron.y + neuronSize / 2);
        ctx.strokeStyle = '#9ca3af';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        continue;
      }
      
      // Draw active neuron
      ctx.beginPath();
      ctx.arc(neuron.x, neuron.y, neuronSize, 0, Math.PI * 2);
      
      // Neuron color - if animating, pulse the input and output layers
      let fillColor = activeNeuronColor;
      if (animate && (layerIdx === 0 || layerIdx === layerCount - 1)) {
        const alpha = Math.sin(time * 3 + neuron.y / 50) * 0.5 + 0.5;
        fillColor = layerIdx === 0 ? 
          `rgba(79, 70, 229, ${0.7 + 0.3 * alpha})` : 
          `rgba(239, 68, 68, ${0.7 + 0.3 * alpha})`;
      } else if (layerIdx === layerCount - 1) {
        // Output layer has a different color
        fillColor = activationColor;
      }
      
      ctx.fillStyle = fillColor;
      ctx.fill();
      
      // Add a subtle border
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
  
  // Text labels for layers
  ctx.font = '12px sans-serif';
  ctx.fillStyle = '#4b5563';
  ctx.textAlign = 'center';
  
  // Input layer
  ctx.fillText('Input', neurons[0][0].x, 15);
  
  // Hidden layers
  for (let i = 1; i < layerCount - 1; i++) {
    let label = 'Hidden';
    if (layerCount > 3) {
      label = `Hidden ${i}`;
    }
    ctx.fillText(label, neurons[i][0].x, 15);
  }
  
  // Output layer
  ctx.fillText('Output', neurons[layerCount - 1][0].x, 15);
}