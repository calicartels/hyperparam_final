// Common hyperparameters database
export type Alternative = {
  value: string;
  description: string;
  type: 'higher' | 'lower' | 'advanced' | 'extreme';
};

export type HyperparameterInfo = {
  name: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  framework: string;
  alternatives: Alternative[];
};

// A basic database of common hyperparameters for ML frameworks
export const hyperparametersDB: Record<string, HyperparameterInfo> = {
  learning_rate: {
    name: "Learning Rate",
    description: "Controls how much to adjust the model weights in response to the estimated error each time the model weights are updated.",
    impact: "high",
    framework: "PyTorch",
    alternatives: [
      { value: "0.01", description: "Faster learning, but may overshoot optimal solution", type: "higher" },
      { value: "0.0001", description: "Slower learning, but more stable convergence", type: "lower" },
      { value: "Scheduled", description: "Start high, decrease over time (e.g., ReduceLROnPlateau)", type: "advanced" }
    ]
  },
  batch_size: {
    name: "Batch Size",
    description: "Number of training examples used in one iteration. Affects memory usage and training speed.",
    impact: "medium",
    framework: "PyTorch",
    alternatives: [
      { value: "64", description: "More stable gradients, but higher memory usage", type: "higher" },
      { value: "16", description: "Less memory usage, but potentially less stable", type: "lower" },
      { value: "Power of 2", description: "Values like 32, 64, 128 utilize GPU memory better", type: "advanced" }
    ]
  },
  dropout_rate: {
    name: "Dropout Rate",
    description: "Probability of setting a neuron's output to zero during training, which helps prevent overfitting by making the network more robust.",
    impact: "medium",
    framework: "PyTorch",
    alternatives: [
      { value: "0.5", description: "More aggressive regularization, better for large networks", type: "higher" },
      { value: "0.1", description: "Milder regularization, for smaller networks or less overfitting", type: "lower" },
      { value: "0 (None)", description: "Disable dropout, useful for small datasets or final training", type: "extreme" }
    ]
  },
  num_epochs: {
    name: "Number of Epochs",
    description: "The number of complete passes through the training dataset. Affects how long the model trains and its final performance.",
    impact: "high",
    framework: "PyTorch",
    alternatives: [
      { value: "10", description: "Longer training time, potentially better model", type: "higher" },
      { value: "3", description: "Shorter training, useful for quick experiments", type: "lower" },
      { value: "Early Stopping", description: "Use validation performance to determine when to stop", type: "advanced" }
    ]
  },
  weight_decay: {
    name: "Weight Decay",
    description: "L2 regularization parameter that prevents the weights from growing too large, helping to reduce overfitting.",
    impact: "medium",
    framework: "PyTorch",
    alternatives: [
      { value: "0.01", description: "Stronger regularization effect", type: "higher" },
      { value: "0.0001", description: "Weaker regularization effect", type: "lower" },
      { value: "0", description: "No weight decay/regularization", type: "extreme" }
    ]
  },
  optimizer: {
    name: "Optimizer",
    description: "Algorithm or method used to adjust the attributes of the neural network to reduce the loss function.",
    impact: "high",
    framework: "PyTorch",
    alternatives: [
      { value: "SGD", description: "Simple but may require careful tuning of learning rate", type: "advanced" },
      { value: "RMSprop", description: "Good for RNNs and problems with noisy gradients", type: "advanced" },
      { value: "AdamW", description: "Adam with proper weight decay, often works well", type: "advanced" }
    ]
  },
  activation_function: {
    name: "Activation Function",
    description: "Non-linear function applied to the output of a neuron, allowing the network to learn complex patterns.",
    impact: "medium",
    framework: "PyTorch",
    alternatives: [
      { value: "Sigmoid", description: "Useful for binary classification output layer", type: "advanced" },
      { value: "Tanh", description: "Output range -1 to 1, often used in RNNs", type: "advanced" },
      { value: "Leaky ReLU", description: "Modified ReLU that allows small negative values", type: "advanced" }
    ]
  },
};

// Regex patterns to identify hyperparameters in various frameworks
export const hyperparameterPatterns = [
  // Optimizer learning rates
  { regex: /lr\s*=\s*([\d.]+)/, key: "learning_rate" },
  { regex: /learning_rate\s*=\s*([\d.]+)/, key: "learning_rate" },
  
  // Batch sizes
  { regex: /batch_size\s*=\s*(\d+)/, key: "batch_size" },
  { regex: /batchSize\s*=\s*(\d+)/, key: "batch_size" },
  
  // Dropout
  { regex: /dropout\s*\(\s*([\d.]+)\s*\)/, key: "dropout_rate" },
  { regex: /dropout\s*=\s*([\d.]+)/, key: "dropout_rate" },
  
  // Training epochs
  { regex: /epochs\s*=\s*(\d+)/, key: "num_epochs" },
  { regex: /num_epochs\s*=\s*(\d+)/, key: "num_epochs" },
  
  // Weight decay
  { regex: /weight_decay\s*=\s*([\d.]+)/, key: "weight_decay" },
  { regex: /decay\s*=\s*([\d.]+)/, key: "weight_decay" },
];

// Function to identify hyperparameters in code
export const identifyHyperparameters = (code: string): { 
  key: string;
  value: string;
  position: { start: number; end: number };
}[] => {
  const results = [];
  
  for (const pattern of hyperparameterPatterns) {
    let match;
    const regex = new RegExp(pattern.regex, 'g');
    
    while ((match = regex.exec(code)) !== null) {
      const value = match[1];
      const start = match.index;
      const end = match.index + match[0].length;
      
      results.push({
        key: pattern.key,
        value,
        position: { start, end }
      });
    }
  }
  
  return results;
};

// Function to get hyperparameter information
export const getHyperparameterInfo = (key: string): HyperparameterInfo | undefined => {
  return hyperparametersDB[key];
};

// Detect framework from imports in code
export const detectFramework = (code: string): string => {
  if (code.includes('import torch') || code.includes('from torch')) {
    return 'PyTorch';
  } else if (code.includes('import tensorflow') || code.includes('from tensorflow')) {
    return 'TensorFlow';
  } else if (code.includes('from keras') || code.includes('import keras')) {
    return 'Keras';
  } else if (code.includes('from sklearn') || code.includes('import sklearn')) {
    return 'scikit-learn';
  }
  return 'Unknown';
};
