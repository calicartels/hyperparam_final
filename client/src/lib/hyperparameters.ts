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

// A comprehensive database of common hyperparameters and configurable aspects for ML frameworks
export const hyperparametersDB: Record<string, HyperparameterInfo> = {
  // PyTorch Hyperparameters
  learning_rate: {
    name: "Learning Rate",
    description: "Controls how much to adjust the model weights in response to the estimated error each time the model weights are updated.",
    impact: "high",
    framework: "PyTorch",
    alternatives: [
      { value: "0.01", description: "Faster learning, but may overshoot optimal solution", type: "higher" },
      { value: "0.0001", description: "Slower learning, but more stable convergence", type: "lower" },
      { value: "Scheduled", description: "Start high, decrease over time (e.g., ReduceLROnPlateau)", type: "advanced" },
      { value: "Cyclic", description: "Cycle between high and low learning rates during training", type: "advanced" }
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
      { value: "Power of 2", description: "Values like 32, 64, 128 utilize GPU memory better", type: "advanced" },
      { value: "Gradient Accumulation", description: "Simulate larger batch sizes with limited memory", type: "advanced" }
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
      { value: "0 (None)", description: "Disable dropout, useful for small datasets or final training", type: "extreme" },
      { value: "Spatial Dropout", description: "Drops entire feature maps instead of individual neurons", type: "advanced" }
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
      { value: "Early Stopping", description: "Use validation performance to determine when to stop", type: "advanced" },
      { value: "Checkpoint Ensemble", description: "Save multiple checkpoints to create ensemble", type: "extreme" }
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
      { value: "0", description: "No weight decay/regularization", type: "extreme" },
      { value: "Decoupled Weight Decay", description: "Apply weight decay separately from optimization", type: "advanced" }
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
      { value: "AdamW", description: "Adam with proper weight decay, often works well", type: "advanced" },
      { value: "Lion", description: "Newer optimizer that can be more efficient than Adam", type: "extreme" }
    ]
  },
  activation_function: {
    name: "Activation Function",
    description: "Non-linear function applied to the output of a neuron, allowing the network to learn complex patterns.",
    impact: "medium",
    framework: "PyTorch",
    alternatives: [
      { value: "ReLU", description: "Fast computation, but can lead to 'dying ReLU' problem", type: "advanced" },
      { value: "Sigmoid", description: "Useful for binary classification output layer", type: "advanced" },
      { value: "Tanh", description: "Output range -1 to 1, often used in RNNs", type: "advanced" },
      { value: "GELU", description: "Smoother activation used in transformers", type: "advanced" }
    ]
  },
  
  // TensorFlow Hyperparameters
  tf_learning_rate: {
    name: "Learning Rate",
    description: "Controls how quickly the model adapts to the problem in TensorFlow models.",
    impact: "high",
    framework: "TensorFlow",
    alternatives: [
      { value: "0.01", description: "Standard starting point for simple models", type: "higher" },
      { value: "0.001", description: "More conservative, helps avoid divergence", type: "lower" },
      { value: "ExponentialDecay", description: "Decays the learning rate exponentially over time", type: "advanced" },
      { value: "WarmUpCosineDecay", description: "Warms up and then follows cosine schedule", type: "extreme" }
    ]
  },
  clipnorm: {
    name: "Gradient Clipping",
    description: "Limits the size of gradients to prevent exploding gradient problems.",
    impact: "medium",
    framework: "TensorFlow",
    alternatives: [
      { value: "1.0", description: "Standard clipping threshold for stable training", type: "advanced" },
      { value: "0.5", description: "More aggressive clipping for very unstable models", type: "lower" },
      { value: "5.0", description: "Looser constraint, allows larger updates", type: "higher" },
      { value: "None", description: "No gradient clipping, may lead to instability", type: "extreme" }
    ]
  },
  momentum: {
    name: "Momentum",
    description: "Accelerates gradient descent and dampens oscillations during optimization.",
    impact: "medium",
    framework: "TensorFlow",
    alternatives: [
      { value: "0.9", description: "Standard value, works well for most models", type: "advanced" },
      { value: "0.99", description: "Higher momentum for smoother updates", type: "higher" },
      { value: "0.5", description: "Lower momentum for more immediate response", type: "lower" },
      { value: "Nesterov", description: "Look-ahead momentum for better convergence", type: "advanced" }
    ]
  },
  
  // scikit-learn Hyperparameters
  n_estimators: {
    name: "Number of Estimators",
    description: "Number of trees in random forest, boosting, or other ensemble methods.",
    impact: "high",
    framework: "scikit-learn",
    alternatives: [
      { value: "100", description: "Standard starting point for most datasets", type: "advanced" },
      { value: "500", description: "More trees for potentially better performance", type: "higher" },
      { value: "50", description: "Faster training with potential performance tradeoff", type: "lower" },
      { value: "1000+", description: "Diminishing returns but may help with complex data", type: "extreme" }
    ]
  },
  max_depth: {
    name: "Maximum Depth",
    description: "Maximum depth of the tree in decision tree models.",
    impact: "high",
    framework: "scikit-learn",
    alternatives: [
      { value: "None", description: "Grow until all leaves are pure (risk of overfitting)", type: "extreme" },
      { value: "5", description: "Shallow trees to prevent overfitting", type: "lower" },
      { value: "10", description: "Medium depth for balance", type: "advanced" },
      { value: "20", description: "Deep trees for complex relationships", type: "higher" }
    ]
  },
  min_samples_split: {
    name: "Minimum Samples to Split",
    description: "Minimum number of samples required to split an internal node in a decision tree.",
    impact: "medium",
    framework: "scikit-learn",
    alternatives: [
      { value: "2", description: "Default: Allow splitting with just 2 samples", type: "lower" },
      { value: "10", description: "More conservative splitting", type: "higher" },
      { value: "5%", description: "Percentage-based splitting threshold", type: "advanced" },
      { value: "20", description: "Very conservative splitting for noisy data", type: "extreme" }
    ]
  },
  
  // Keras Specific
  kernel_initializer: {
    name: "Kernel Initializer",
    description: "Defines the way to set the initial random weights of layers in Keras models.",
    impact: "medium",
    framework: "Keras",
    alternatives: [
      { value: "glorot_uniform", description: "Default initializer, works well for most networks", type: "advanced" },
      { value: "he_normal", description: "Better for models using ReLU activations", type: "advanced" },
      { value: "orthogonal", description: "Can help with vanishing/exploding gradients", type: "advanced" },
      { value: "zeros", description: "Initialize weights to zero (rarely used)", type: "extreme" }
    ]
  },
  
  // Transformers
  attention_heads: {
    name: "Attention Heads",
    description: "Number of attention heads in transformer models.",
    impact: "high",
    framework: "Transformers",
    alternatives: [
      { value: "8", description: "Standard for medium-sized models", type: "advanced" },
      { value: "16", description: "More heads for capturing complex patterns", type: "higher" },
      { value: "4", description: "Fewer heads for faster computation", type: "lower" },
      { value: "Multi-Query Attention", description: "Alternative attention mechanism with shared keys/values", type: "extreme" }
    ]
  },
  
  // Generic
  random_seed: {
    name: "Random Seed",
    description: "Controls reproducibility of the model training process.",
    impact: "low",
    framework: "Generic",
    alternatives: [
      { value: "42", description: "Common default seed", type: "advanced" },
      { value: "None", description: "Different results each run", type: "extreme" },
      { value: "Multiple Seeds", description: "Run with different seeds and average results", type: "advanced" },
      { value: "Time-based", description: "Seed based on timestamp for uniqueness", type: "advanced" }
    ]
  },
  
  // New Additions
  embedding_dim: {
    name: "Embedding Dimension",
    description: "Size of the embedding vectors used to represent categorical features or words.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "128", description: "Medium size embeddings, work for many applications", type: "advanced" },
      { value: "300", description: "Larger embeddings for more complex relationships", type: "higher" },
      { value: "64", description: "Smaller embeddings for efficiency", type: "lower" },
      { value: "512+", description: "Very large embeddings for extremely complex tasks", type: "extreme" }
    ]
  },
  
  hidden_size: {
    name: "Hidden Layer Size",
    description: "Number of neurons in hidden layers of neural networks.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "128", description: "Medium-sized layer for balance", type: "advanced" },
      { value: "512", description: "Larger layer for more complex patterns", type: "higher" },
      { value: "64", description: "Smaller layer for efficiency", type: "lower" },
      { value: "Powers of 2", description: "Using sizes like 64, 128, 256 for better memory alignment", type: "advanced" }
    ]
  },
  
  // NEW: Model Architecture Choices
  model_architecture: {
    name: "Model Architecture",
    description: "The overall structure and type of machine learning model being used.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "Sequential", description: "Simple layer-by-layer architecture for straightforward tasks", type: "advanced" },
      { value: "Functional API", description: "More flexible architecture allowing complex layer connections", type: "advanced" },
      { value: "ResNet", description: "Deep CNN with residual connections to avoid vanishing gradients", type: "advanced" },
      { value: "Transformer", description: "Architecture using self-attention for sequence modeling", type: "advanced" }
    ]
  },
  
  // NEW: Loss Functions
  loss_function: {
    name: "Loss Function",
    description: "Function that measures how well the model performs during training, to be minimized.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "categorical_crossentropy", description: "Standard loss for multi-class classification", type: "advanced" },
      { value: "binary_crossentropy", description: "Loss for binary classification tasks", type: "advanced" },
      { value: "mse", description: "Mean Squared Error for regression tasks", type: "advanced" },
      { value: "custom", description: "Custom-designed loss for specific requirements", type: "extreme" }
    ]
  },
  
  // NEW: Normalization Parameters
  normalization_mean: {
    name: "Normalization Mean",
    description: "Mean values used for normalizing input data. Often channel-wise means for images.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "[0.485, 0.456, 0.406]", description: "ImageNet standard RGB means", type: "advanced" },
      { value: "[0.5, 0.5, 0.5]", description: "Simplified normalization to center around zero", type: "advanced" },
      { value: "Dataset-specific", description: "Calculate means from your specific dataset", type: "advanced" },
      { value: "None", description: "Skip mean normalization (not recommended)", type: "extreme" }
    ]
  },
  
  normalization_std: {
    name: "Normalization Standard Deviation",
    description: "Standard deviation values used for normalizing input data. Scales the data to improve training.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "[0.229, 0.224, 0.225]", description: "ImageNet standard RGB standard deviations", type: "advanced" },
      { value: "[0.5, 0.5, 0.5]", description: "Simplified scaling to range of approximately [-1, 1]", type: "advanced" },
      { value: "Dataset-specific", description: "Calculate standard deviations from your specific dataset", type: "advanced" },
      { value: "None", description: "Skip standard deviation normalization (not recommended)", type: "extreme" }
    ]
  },
  
  // NEW: Data Processing
  dataloader_workers: {
    name: "DataLoader Workers",
    description: "Number of subprocesses used for data loading. Affects training speed and CPU usage.",
    impact: "low",
    framework: "PyTorch",
    alternatives: [
      { value: "4", description: "Balanced choice for most systems", type: "advanced" },
      { value: "0", description: "No multiprocessing, useful for debugging", type: "lower" },
      { value: "8", description: "Higher parallelism for systems with many cores", type: "higher" },
      { value: "num_cpus", description: "Set to number of CPU cores available", type: "advanced" }
    ]
  },
  
  data_shuffle: {
    name: "Data Shuffling",
    description: "Determines whether training data is randomly shuffled between epochs.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "True", description: "Randomizes data order, recommended for most cases", type: "advanced" },
      { value: "False", description: "Keeps original data order, useful for time series", type: "advanced" },
      { value: "Custom Sampler", description: "Create custom sampling strategy for imbalanced data", type: "extreme" },
      { value: "Per-batch shuffle", description: "Only shuffle within batches, not across all data", type: "advanced" }
    ]
  },
  
  // NEW: Convolutional Network Parameters
  kernel_size: {
    name: "Kernel Size",
    description: "Size of the convolutional filter kernel. Determines the receptive field of the convolution.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "3", description: "Standard size for most CNNs, balance of local features", type: "advanced" },
      { value: "1", description: "Point-wise convolution for channel mixing", type: "lower" },
      { value: "5", description: "Larger receptive field for broader feature capture", type: "higher" },
      { value: "Variable sizes", description: "Mix different kernel sizes for multiple feature scales", type: "advanced" }
    ]
  },
  
  conv_filters: {
    name: "Convolutional Filters",
    description: "Number of filters/channels in a convolutional layer. Determines feature extraction capacity.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "64", description: "Standard starting point for many architectures", type: "advanced" },
      { value: "32", description: "Fewer parameters, faster but less expressive", type: "lower" },
      { value: "128", description: "More features, potentially better recognition", type: "higher" },
      { value: "Progressive growth", description: "Double filters in deeper layers", type: "advanced" }
    ]
  },
  
  conv_strides: {
    name: "Convolution Stride",
    description: "Step size when sliding the filter over the input. Affects output size and computation.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "1", description: "Standard stride, preserves spatial dimensions", type: "advanced" },
      { value: "2", description: "Downsamples by factor of 2, reduces computation", type: "advanced" },
      { value: "(2,2)", description: "Stride in both dimensions, common for downsampling", type: "advanced" },
      { value: "Dilated", description: "Use dilated convolutions for larger receptive field", type: "extreme" }
    ]
  },
  
  padding_type: {
    name: "Padding Type",
    description: "How the input is padded before convolution. Affects output size and edge treatment.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "same", description: "Output size same as input, padding as needed", type: "advanced" },
      { value: "valid", description: "No padding, output size reduced", type: "advanced" },
      { value: "causal", description: "Padding only on one side, for causal convolutions", type: "advanced" },
      { value: "reflect", description: "Reflection padding for better edge handling", type: "extreme" }
    ]
  },
  
  // NEW: Pooling Operations
  pool_size: {
    name: "Pooling Size",
    description: "Size of the window for pooling operations. Affects downsampling rate.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "2", description: "Standard pooling, halves dimensions", type: "advanced" },
      { value: "3", description: "Larger pooling, more aggressive downsampling", type: "higher" },
      { value: "Global", description: "Global pooling across entire feature map", type: "extreme" },
      { value: "Spatial Pyramid", description: "Multiple pooling sizes for multi-scale features", type: "advanced" }
    ]
  },
  
  // NEW: Regularization Methods
  l1_regularization: {
    name: "L1 Regularization",
    description: "Adds penalty proportional to the absolute value of weights, promoting sparsity.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "0.01", description: "Moderate sparsity encouragement", type: "advanced" },
      { value: "0.001", description: "Lighter regularization effect", type: "lower" },
      { value: "0.1", description: "Strong sparsity encouragement", type: "higher" },
      { value: "0", description: "No L1 regularization", type: "extreme" }
    ]
  },
  
  activity_regularization: {
    name: "Activity Regularization",
    description: "Applies regularization to the output of the layer instead of weights.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "l1", description: "Sparsity in activations, encourages focused features", type: "advanced" },
      { value: "l2", description: "Stabilizes activations, prevents extreme values", type: "advanced" },
      { value: "l1_l2", description: "Combined L1 and L2 regularization on activations", type: "advanced" },
      { value: "None", description: "No activity regularization", type: "extreme" }
    ]
  },
  
  // NEW: RNN Parameters
  rnn_return_sequences: {
    name: "Return Sequences",
    description: "Whether RNN layer should return output for each time step or just the final output.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "True", description: "Return output for each time step, for sequence tasks", type: "advanced" },
      { value: "False", description: "Return only final output, for classification tasks", type: "advanced" },
      { value: "Mixed approach", description: "Some layers return sequences, others don't", type: "advanced" },
      { value: "Attention over sequences", description: "Return sequences with attention mechanism", type: "extreme" }
    ]
  },
  
  bidirectional: {
    name: "Bidirectional RNN",
    description: "Whether RNN processes sequence in both forward and backward directions.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "True", description: "Process in both directions, better for context", type: "advanced" },
      { value: "False", description: "Process only forward, for causal or streaming data", type: "advanced" },
      { value: "Stacked unidirectional", description: "Stack multiple unidirectional layers", type: "advanced" },
      { value: "Mixed layers", description: "Some layers bidirectional, others not", type: "advanced" }
    ]
  },
  
  // NEW: Attention Mechanisms
  attention_dropout: {
    name: "Attention Dropout",
    description: "Dropout rate applied to attention weights in transformer models.",
    impact: "medium",
    framework: "Transformers",
    alternatives: [
      { value: "0.1", description: "Standard rate for most transformer models", type: "advanced" },
      { value: "0.2", description: "Higher dropout for stronger regularization", type: "higher" },
      { value: "0.0", description: "No dropout, may overfit on smaller datasets", type: "extreme" },
      { value: "Scheduled", description: "Vary dropout rate during training", type: "advanced" }
    ]
  },
  
  num_attention_layers: {
    name: "Number of Attention Layers",
    description: "Number of attention/transformer blocks stacked in the model.",
    impact: "high",
    framework: "Transformers",
    alternatives: [
      { value: "6", description: "Standard for medium-sized transformers", type: "advanced" },
      { value: "12", description: "Deeper model for more complex relationships", type: "higher" },
      { value: "3", description: "Lightweight model for simpler tasks", type: "lower" },
      { value: "Progressive depth", description: "Vary layer depth across model sections", type: "advanced" }
    ]
  },
  
  // NEW: Output Configuration
  num_classes: {
    name: "Number of Classes",
    description: "Number of output classes for classification tasks.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "Dataset-specific", description: "Must match your specific dataset classes", type: "advanced" },
      { value: "1", description: "For binary classification with sigmoid output", type: "advanced" },
      { value: "Hierarchical classes", description: "Organize classes in hierarchy", type: "extreme" },
      { value: "Multi-label", description: "Allow multiple classes per sample", type: "advanced" }
    ]
  },
  
  // NEW: Training Controls
  early_stopping_patience: {
    name: "Early Stopping Patience",
    description: "Number of epochs with no improvement after which training will stop.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "10", description: "Standard patience period for most models", type: "advanced" },
      { value: "5", description: "Shorter patience for faster training", type: "lower" },
      { value: "20", description: "Longer patience for more training time", type: "higher" },
      { value: "Adaptive", description: "Adjust patience based on improvement rate", type: "extreme" }
    ]
  },
  
  lr_reduction_factor: {
    name: "Learning Rate Reduction Factor",
    description: "Factor by which learning rate is reduced on plateau in adaptive schedules.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "0.1", description: "Standard reduction (10x smaller)", type: "advanced" },
      { value: "0.5", description: "More gradual reduction (2x smaller)", type: "higher" },
      { value: "0.01", description: "Aggressive reduction (100x smaller)", type: "lower" },
      { value: "Custom schedule", description: "Design custom LR reduction schedule", type: "extreme" }
    ]
  },
  
  // NEW: Transfer Learning
  feature_extraction: {
    name: "Feature Extraction",
    description: "Using pre-trained model for feature extraction by freezing early layers.",
    impact: "high",
    framework: "Generic",
    alternatives: [
      { value: "True", description: "Freeze base model, train only new layers", type: "advanced" },
      { value: "False", description: "Fine-tune entire model", type: "advanced" },
      { value: "Progressive unfreezing", description: "Gradually unfreeze layers during training", type: "advanced" },
      { value: "Discriminative LR", description: "Different learning rates for different layers", type: "extreme" }
    ]
  },
  
  // NEW: Batch Normalization
  batch_norm_momentum: {
    name: "Batch Normalization Momentum",
    description: "Momentum for the moving average in batch normalization layers.",
    impact: "medium",
    framework: "Generic",
    alternatives: [
      { value: "0.99", description: "Standard momentum, stable statistics", type: "advanced" },
      { value: "0.9", description: "More responsive to recent batches", type: "lower" },
      { value: "0.999", description: "More stable long-term statistics", type: "higher" },
      { value: "Scheduled", description: "Adjust momentum during training", type: "extreme" }
    ]
  },
  
  batch_norm_epsilon: {
    name: "Batch Normalization Epsilon",
    description: "Small constant added to variance to prevent division by zero.",
    impact: "low",
    framework: "Generic",
    alternatives: [
      { value: "1e-5", description: "Standard epsilon for most models", type: "advanced" },
      { value: "1e-3", description: "Larger epsilon for more stable training", type: "higher" },
      { value: "1e-7", description: "Smaller epsilon for more precise normalization", type: "lower" },
      { value: "Adaptive", description: "Adjust epsilon based on batch statistics", type: "extreme" }
    ]
  }
};

// Regex patterns to identify hyperparameters and configurable aspects in various frameworks
export const hyperparameterPatterns = [
  // Optimizer learning rates
  { regex: /lr\s*=\s*([\d.]+)/, key: "learning_rate" },
  { regex: /learning_rate\s*=\s*([\d.]+)/, key: "learning_rate" },
  { regex: /LearningRateScheduler\([\s\S]*?([\d.]+)/, key: "learning_rate" }, // Common in TF/Keras
  
  // Batch sizes
  { regex: /batch_size\s*=\s*(\d+)/, key: "batch_size" },
  { regex: /batchSize\s*=\s*(\d+)/, key: "batch_size" },
  { regex: /batch_size:\s*(\d+)/, key: "batch_size" }, // YAML configs
  
  // Dropout
  { regex: /dropout\s*\(\s*([\d.]+)\s*\)/, key: "dropout_rate" },
  { regex: /dropout\s*=\s*([\d.]+)/, key: "dropout_rate" },
  { regex: /Dropout\(([\d.]+)\)/, key: "dropout_rate" }, // Class initialization
  { regex: /rate\s*=\s*([\d.]+)/, key: "dropout_rate" }, // Dropout rate parameter
  
  // Training epochs
  { regex: /epochs\s*=\s*(\d+)/, key: "num_epochs" },
  { regex: /num_epochs\s*=\s*(\d+)/, key: "num_epochs" },
  { regex: /n_epochs\s*=\s*(\d+)/, key: "num_epochs" },
  { regex: /max_epochs\s*=\s*(\d+)/, key: "num_epochs" }, // PyTorch Lightning
  
  // Weight decay
  { regex: /weight_decay\s*=\s*([\d.]+)/, key: "weight_decay" },
  { regex: /decay\s*=\s*([\d.]+)/, key: "weight_decay" },
  { regex: /l2\s*=\s*([\d.]+)/, key: "weight_decay" }, // L2 regularization
  
  // TensorFlow specific
  { regex: /clipnorm\s*=\s*([\d.]+)/, key: "clipnorm" },
  { regex: /clip_value\s*=\s*([\d.]+)/, key: "clipnorm" },
  { regex: /momentum\s*=\s*([\d.]+)/, key: "momentum" },
  
  // Optimizers
  { regex: /optimizer\s*=\s*['"]?(\w+)['"]?/, key: "optimizer" },
  { regex: /Optimizer\(\s*['"]?(\w+)['"]?/, key: "optimizer" },
  { regex: /optimizer\.(\w+)\(/, key: "optimizer" }, // PyTorch optimizers
  { regex: /tf\.keras\.optimizers\.(\w+)\(/, key: "optimizer" }, // TF optimizers
  
  // scikit-learn specific
  { regex: /n_estimators\s*=\s*(\d+)/, key: "n_estimators" },
  { regex: /max_depth\s*=\s*(\d+|None)/, key: "max_depth" },
  { regex: /min_samples_split\s*=\s*(\d+)/, key: "min_samples_split" },
  
  // Keras specific
  { regex: /kernel_initializer\s*=\s*['"]?(\w+)['"]?/, key: "kernel_initializer" },
  
  // Transformer specific
  { regex: /num_heads\s*=\s*(\d+)/, key: "attention_heads" },
  { regex: /n_head\s*=\s*(\d+)/, key: "attention_heads" },
  { regex: /num_attention_heads\s*=\s*(\d+)/, key: "attention_heads" },
  
  // Generic
  { regex: /random_state\s*=\s*(\d+|None)/, key: "random_seed" },
  { regex: /seed\s*=\s*(\d+)/, key: "random_seed" },
  { regex: /embedding_dim\s*=\s*(\d+)/, key: "embedding_dim" },
  { regex: /emb_size\s*=\s*(\d+)/, key: "embedding_dim" },
  { regex: /d_model\s*=\s*(\d+)/, key: "embedding_dim" }, // Transformer embedding dim
  { regex: /hidden_size\s*=\s*(\d+)/, key: "hidden_size" },
  { regex: /hidden_dim\s*=\s*(\d+)/, key: "hidden_size" },
  { regex: /hidden_units\s*=\s*(\d+)/, key: "hidden_size" },
  
  // NEW: Model architecture choices
  { regex: /model\s*=\s*(\w+)/, key: "model_architecture" },
  { regex: /model_type\s*=\s*['"]?(\w+)['"]?/, key: "model_architecture" },
  { regex: /architecture\s*=\s*['"]?(\w+)['"]?/, key: "model_architecture" },
  { regex: /Sequential\(\[/, key: "model_architecture", value: "Sequential" }, // Detect Sequential models
  { regex: /class\s+(\w+Model)\(/, key: "model_architecture" }, // Custom model classes
  { regex: /models\.(\w+)\(/, key: "model_architecture" }, // Pre-built models like ResNet, etc.
  
  // NEW: Activation functions
  { regex: /activation\s*=\s*['"]?(\w+)['"]?/, key: "activation_function" },
  { regex: /activation_fn\s*=\s*['"]?(\w+)['"]?/, key: "activation_function" },
  { regex: /nn\.(\w+)\(/, key: "activation_function" }, // PyTorch activations
  { regex: /tf\.nn\.(\w+)/, key: "activation_function" }, // TensorFlow activations
  
  // NEW: Loss functions
  { regex: /loss\s*=\s*['"]?(\w+)['"]?/, key: "loss_function" },
  { regex: /loss_fn\s*=\s*(\w+)/, key: "loss_function" },
  { regex: /criterion\s*=\s*nn\.(\w+)/, key: "loss_function" }, // PyTorch losses
  { regex: /tf\.keras\.losses\.(\w+)/, key: "loss_function" }, // TF losses
  
  // NEW: Data preprocessing
  { regex: /mean\s*=\s*\[([^\]]+)\]/, key: "normalization_mean" },
  { regex: /std\s*=\s*\[([^\]]+)\]/, key: "normalization_std" },
  { regex: /num_workers\s*=\s*(\d+)/, key: "dataloader_workers" },
  { regex: /shuffle\s*=\s*(True|False)/, key: "data_shuffle" },
  
  // NEW: Layer configurations
  { regex: /kernel_size\s*=\s*(\d+|\(\d+,\s*\d+\))/, key: "kernel_size" },
  { regex: /filters\s*=\s*(\d+)/, key: "conv_filters" },
  { regex: /out_channels\s*=\s*(\d+)/, key: "conv_filters" }, // PyTorch equivalent
  { regex: /strides\s*=\s*(\d+|\(\d+,\s*\d+\))/, key: "conv_strides" },
  { regex: /stride\s*=\s*(\d+|\(\d+,\s*\d+\))/, key: "conv_strides" }, // PyTorch version
  { regex: /padding\s*=\s*['"]?(\w+|\d+)['"]?/, key: "padding_type" },
  
  // NEW: Pooling operations
  { regex: /pool_size\s*=\s*(\d+|\(\d+,\s*\d+\))/, key: "pool_size" },
  { regex: /MaxPool\w*\((\d+)/, key: "pool_size" }, // MaxPool with kernel size
  { regex: /AvgPool\w*\((\d+)/, key: "pool_size" }, // AvgPool with kernel size
  
  // NEW: Regularization methods
  { regex: /regularizers\.l1_l2\(l1\s*=\s*([\d.]+)/, key: "l1_regularization" },
  { regex: /regularizers\.l1\(([\d.]+)/, key: "l1_regularization" },
  { regex: /activity_regularizer\s*=\s*(\w+)/, key: "activity_regularization" },
  
  // NEW: Recurrent networks specific
  { regex: /return_sequences\s*=\s*(True|False)/, key: "rnn_return_sequences" },
  { regex: /bidirectional\s*=\s*(True|False)/, key: "bidirectional" },
  { regex: /Bidirectional\(/, key: "bidirectional", value: "True" },
  { regex: /stateful\s*=\s*(True|False)/, key: "rnn_stateful" },
  
  // NEW: Attention mechanisms
  { regex: /attention_probs_dropout_prob\s*=\s*([\d.]+)/, key: "attention_dropout" },
  { regex: /MultiheadAttention\([\s\S]*?dropout\s*=\s*([\d.]+)/, key: "attention_dropout" },
  { regex: /num_attention_layers\s*=\s*(\d+)/, key: "num_attention_layers" },
  
  // NEW: Input/output shapes
  { regex: /input_shape\s*=\s*\(([^\)]+)\)/, key: "input_shape" },
  { regex: /output_dim\s*=\s*(\d+)/, key: "output_dim" },
  { regex: /num_classes\s*=\s*(\d+)/, key: "num_classes" },
  { regex: /output_classes\s*=\s*(\d+)/, key: "num_classes" },
  
  // NEW: Callbacks and training configurations
  { regex: /EarlyStopping\([\s\S]*?patience\s*=\s*(\d+)/, key: "early_stopping_patience" },
  { regex: /patience\s*=\s*(\d+)/, key: "patience" },
  { regex: /ReduceLROnPlateau\([\s\S]*?factor\s*=\s*([\d.]+)/, key: "lr_reduction_factor" },
  { regex: /ModelCheckpoint\(/, key: "model_checkpointing", value: "True" },
  
  // NEW: Gradient handling
  { regex: /clip_grad_norm_\([\s\S]*?([\d.]+)/, key: "gradient_clipping" },
  { regex: /gradient_accumulation_steps\s*=\s*(\d+)/, key: "gradient_accumulation" },
  
  // NEW: Transfer learning
  { regex: /freeze\(/, key: "feature_extraction", value: "True" },
  { regex: /trainable\s*=\s*(True|False)/, key: "layer_trainable" },
  { regex: /requires_grad\s*=\s*(True|False)/, key: "requires_grad" },
  
  // NEW: Batch normalization
  { regex: /BatchNormalization\([\s\S]*?momentum\s*=\s*([\d.]+)/, key: "batch_norm_momentum" },
  { regex: /eps\s*=\s*([\d.]+)/, key: "batch_norm_epsilon" },
  { regex: /track_running_stats\s*=\s*(True|False)/, key: "track_stats" },
];

// Function to identify all configurable aspects (hyperparameters and other choices) in code
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
      // Use the captured value from regex, or the fixed value if provided in the pattern
      const value = pattern.value !== undefined ? pattern.value : match[1];
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

// New: Enhanced hyperparameter info that uses LLM when available
import { 
  getHyperparameterExplanation, 
  checkLLMStatus, 
  type HyperparameterExplanation, 
  generateFallbackExplanation 
} from './llmService';

// Function to get hyperparameter information with LLM enhancement
export const getEnhancedHyperparameterInfo = async (
  key: string, 
  value: string, 
  codeContext?: string
): Promise<HyperparameterInfo> => {
  // First check if we have it in the static database
  if (hyperparametersDB[key]) {
    return hyperparametersDB[key];
  }
  
  // If not in static DB, try to get from LLM
  try {
    // Check if LLM is available
    const llmStatus = await checkLLMStatus();
    
    if (llmStatus.available) {
      // Detect framework from the code context
      const framework = codeContext ? detectFramework(codeContext) : undefined;
      
      // Get explanation from LLM
      const response = await getHyperparameterExplanation({
        paramName: key,
        paramValue: value,
        framework,
        codeContext
      });
      
      if (response.success && response.explanation) {
        return convertLLMResponseToHyperparameterInfo(response.explanation);
      }
    }
    
    // If LLM fails or is unavailable, use fallback
    return convertLLMResponseToHyperparameterInfo(
      generateFallbackExplanation(key, value, detectFramework(codeContext || ""))
    );
  } catch (error) {
    console.error("Error getting hyperparameter info:", error);
    
    // Use fallback in case of any errors
    return convertLLMResponseToHyperparameterInfo(
      generateFallbackExplanation(key, value, detectFramework(codeContext || ""))
    );
  }
};

// Convert the LLM response to our internal HyperparameterInfo format
const convertLLMResponseToHyperparameterInfo = (llmResponse: HyperparameterExplanation): HyperparameterInfo => {
  return {
    name: llmResponse.name,
    description: llmResponse.description,
    impact: llmResponse.impact,
    framework: "Detected", // We don't have this in the LLM response format
    alternatives: llmResponse.alternatives,
  };
};

// Detect framework from imports in code
export const detectFramework = (code: string): string => {
  // Framework detection patterns
  const frameworks = [
    // Deep Learning Frameworks
    { name: 'PyTorch', patterns: ['import torch', 'from torch', 'torch.nn', 'torch.optim'] },
    { name: 'TensorFlow', patterns: ['import tensorflow', 'from tensorflow', 'tf.keras', 'tf.nn', 'tf.data'] },
    { name: 'Keras', patterns: ['from keras', 'import keras', 'keras.layers', 'keras.models'] },
    { name: 'JAX', patterns: ['import jax', 'from jax', 'jax.numpy', 'flax.linen'] },
    
    // ML Libraries
    { name: 'scikit-learn', patterns: ['from sklearn', 'import sklearn', 'sklearn.ensemble', 'sklearn.linear_model'] },
    { name: 'XGBoost', patterns: ['import xgboost', 'from xgboost', 'XGBClassifier', 'XGBRegressor'] },
    { name: 'LightGBM', patterns: ['import lightgbm', 'from lightgbm', 'LGBMClassifier', 'LGBMRegressor'] },
    
    // NLP Libraries
    { name: 'Transformers', patterns: ['import transformers', 'from transformers', 'AutoModel', 'pipeline('] },
    { name: 'spaCy', patterns: ['import spacy', 'from spacy', 'nlp = spacy.load'] },
    
    // Computer Vision
    { name: 'OpenCV', patterns: ['import cv2', 'from cv2', 'cv2.imread'] },
    
    // Statistical
    { name: 'StatsModels', patterns: ['import statsmodels', 'from statsmodels', 'statsmodels.api'] },
  ];
  
  // Check each framework's patterns
  for (const framework of frameworks) {
    if (framework.patterns.some(pattern => code.includes(pattern))) {
      return framework.name;
    }
  }
  
  // Check for common ML keywords if no specific framework is detected
  if (code.includes('fit(') || code.includes('predict(') || 
      code.includes('train_test_split') || code.includes('model.add')) {
    return 'Generic ML';
  }
  
  return 'Unknown';
};

// Enhanced functions to support more code analysis capabilities

// Get a list of frameworks detected in the code
export const getAllDetectedFrameworks = (code: string): string[] => {
  const frameworks = [
    { name: 'PyTorch', patterns: ['import torch', 'from torch', 'torch.nn', 'torch.optim'] },
    { name: 'TensorFlow', patterns: ['import tensorflow', 'from tensorflow', 'tf.keras', 'tf.nn'] },
    { name: 'Keras', patterns: ['from keras', 'import keras', 'keras.layers', 'keras.models'] },
    { name: 'scikit-learn', patterns: ['from sklearn', 'import sklearn', 'sklearn.ensemble'] },
    { name: 'JAX', patterns: ['import jax', 'from jax', 'jax.numpy', 'flax.linen'] },
    { name: 'XGBoost', patterns: ['import xgboost', 'from xgboost', 'XGBClassifier'] },
    { name: 'Transformers', patterns: ['import transformers', 'from transformers', 'AutoModel'] },
  ];
  
  return frameworks
    .filter(framework => framework.patterns.some(pattern => code.includes(pattern)))
    .map(framework => framework.name);
};

// Function to generate parameter comparison data
export const compareHyperparameters = (
  foundParams: { key: string; value: string }[]
): {
  key: string;
  currentValue: string;
  recommendation: string;
  improvement: string;
  impact: string;
}[] => {
  return foundParams.map(param => {
    const paramInfo = hyperparametersDB[param.key];
    if (!paramInfo) {
      return {
        key: param.key,
        currentValue: param.value,
        recommendation: "N/A",
        improvement: "Unknown",
        impact: "Unknown"
      };
    }
    
    // Find a recommended alternative
    let recommendation = "";
    let improvement = "";
    
    // Look for an "advanced" alternative first
    const advancedOption = paramInfo.alternatives.find(alt => alt.type === "advanced");
    if (advancedOption) {
      recommendation = advancedOption.value;
      improvement = advancedOption.description;
    } else if (paramInfo.alternatives.length > 0) {
      recommendation = paramInfo.alternatives[0].value;
      improvement = paramInfo.alternatives[0].description;
    }
    
    return {
      key: paramInfo.name,
      currentValue: param.value,
      recommendation,
      improvement,
      impact: paramInfo.impact
    };
  });
};

// Function to generate markdown documentation for hyperparameters
export const generateParameterDocs = (
  code: string,
  foundParams: { key: string; value: string }[]
): string => {
  const framework = detectFramework(code);
  const allFrameworks = getAllDetectedFrameworks(code);
  
  let mdDoc = `# Hyperparameter Documentation\n\n`;
  
  // Add framework information
  mdDoc += `## Framework Information\n\n`;
  mdDoc += `Primary Framework: ${framework}\n\n`;
  
  if (allFrameworks.length > 1) {
    mdDoc += `All Detected Frameworks: ${allFrameworks.join(', ')}\n\n`;
  }
  
  // Add hyperparameter details
  mdDoc += `## Hyperparameters\n\n`;
  
  foundParams.forEach(param => {
    const paramInfo = hyperparametersDB[param.key];
    if (!paramInfo) return;
    
    mdDoc += `### ${paramInfo.name}\n\n`;
    mdDoc += `**Current Value:** \`${param.value}\`\n\n`;
    mdDoc += `**Description:** ${paramInfo.description}\n\n`;
    mdDoc += `**Impact:** ${paramInfo.impact.charAt(0).toUpperCase() + paramInfo.impact.slice(1)}\n\n`;
    
    if (paramInfo.alternatives.length > 0) {
      mdDoc += `**Alternatives:**\n\n`;
      paramInfo.alternatives.forEach(alt => {
        mdDoc += `- **${alt.value}**: ${alt.description}\n`;
      });
      mdDoc += `\n`;
    }
  });
  
  // Add references section
  mdDoc += `## References\n\n`;
  mdDoc += `- [${framework} Documentation](https://example.com/${framework.toLowerCase()})\n`;
  mdDoc += `- [Hyperparameter Optimization Guide](https://example.com/hyperparameter-optimization)\n\n`;
  
  return mdDoc;
};

// Function to generate alternative code with different hyperparameters
export const generateAlternativeCode = (
  originalCode: string,
  paramToReplace: { key: string; value: string; position: { start: number; end: number } },
  newValue: string
): string => {
  // Get the original line containing the parameter
  const lines = originalCode.split('\n');
  let lineIndex = 0;
  let charCount = 0;
  let targetLine = 0;
  let lineStart = 0;
  
  // Find the line containing the parameter
  for (let i = 0; i < lines.length; i++) {
    lineStart = charCount;
    charCount += lines[i].length + 1; // +1 for newline
    
    if (paramToReplace.position.start >= lineStart && paramToReplace.position.start < charCount) {
      targetLine = i;
      break;
    }
  }
  
  // Original line and parameter text
  const originalLine = lines[targetLine];
  const paramText = originalCode.substring(paramToReplace.position.start, paramToReplace.position.end);
  
  // Replace the parameter value in the original line
  const newParamText = paramText.replace(paramToReplace.value, newValue);
  const newLine = originalLine.replace(paramText, newParamText);
  
  // Create modified code
  const modifiedLines = [...lines];
  modifiedLines[targetLine] = newLine;
  
  return modifiedLines.join('\n');
};

// Function to generate a complete alternative version of the code with recommended hyperparameters
export const generateOptimizedCode = (
  originalCode: string,
  params: { key: string; value: string; position: { start: number; end: number } }[]
): string => {
  let optimizedCode = originalCode;
  
  // Sort params in reverse order by position to avoid messing up indices
  const sortedParams = [...params].sort(
    (a, b) => b.position.start - a.position.start
  );
  
  for (const param of sortedParams) {
    const paramInfo = hyperparametersDB[param.key];
    if (!paramInfo || !paramInfo.alternatives.length) continue;
    
    // Choose the most "advanced" alternative
    const advancedAlt = paramInfo.alternatives.find(alt => alt.type === "advanced");
    if (advancedAlt) {
      optimizedCode = generateAlternativeCode(optimizedCode, param, advancedAlt.value);
    }
  }
  
  return optimizedCode;
};
