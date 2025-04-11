// src/lib/llmService.ts
import { apiRequest } from "./queryClient";

/**
 * Interface for the hyperparameter explanation request
 */
export interface ExplainHyperparameterRequest {
  paramName: string;
  paramValue: string;
  framework?: string;
  codeContext?: string;
}

/**
 * Interface for the alternative value suggestion
 */
export interface AlternativeValue {
  value: string;
  description: string;
  type: 'higher' | 'lower' | 'advanced' | 'extreme';
}

/**
 * Interface for the structured LLM explanation response
 */
export interface HyperparameterExplanation {
  name: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  valueAnalysis: string;
  alternatives: AlternativeValue[];
  bestPractices: string;
  tradeoffs: string;
}

/**
 * Interface for the full LLM API response
 */
export interface LLMExplanationResponse {
  success: boolean;
  explanation: HyperparameterExplanation;
  error?: string;
  rawResponse?: string;
  fallbackAvailable?: boolean;
}

/**
 * Interface for the LLM status response
 */
export interface LLMStatusResponse {
  available: boolean;
  provider: string;
  model: string;
  requiresAuth: boolean;
}

/**
 * Check if LLM service is available
 * @returns Promise with LLM status
 */
export async function checkLLMStatus(): Promise<LLMStatusResponse> {
  try {
    const response = await fetch('/api/llm/status');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to check LLM status:", error);
    return {
      available: false,
      provider: "Unknown",
      model: "Unknown",
      requiresAuth: true
    };
  }
}

/**
 * Get a hyperparameter explanation from the LLM service
 * @param request Hyperparameter explanation request
 * @returns Promise with explanation
 */
export async function getHyperparameterExplanation(
  request: ExplainHyperparameterRequest
): Promise<LLMExplanationResponse> {
  try {
    const response = await fetch('/api/llm/explain-hyperparameter', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error getting hyperparameter explanation:", error);
    return {
      success: false,
      explanation: generateFallbackExplanation(request.paramName, request.paramValue, request.framework),
      error: `Error: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}

/**
 * Get hyperparameter detection from the LLM service
 * @param code Code to analyze
 * @returns Promise with detected parameters
 */
export async function detectHyperparametersWithLLM(code: string): Promise<any> {
  try {
    const response = await fetch('/api/llm/detect-hyperparameters', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error("Error detecting hyperparameters:", error);
    return {
      success: false,
      parameters: [],
      error: `Error: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}

/**
 * Generate fallback content if LLM service is unavailable
 * @param paramName Hyperparameter name
 * @param paramValue Parameter value
 * @param framework Optional framework name
 * @returns Fallback explanation
 */
export function generateFallbackExplanation(
  paramName: string,
  paramValue: string,
  framework?: string
): HyperparameterExplanation {
  // Convert paramName to a more readable format
  const formattedName = paramName
    .replace(/_/g, ' ')
    .replace(/(\w)(\w*)/g, (_, first, rest) => first.toUpperCase() + rest.toLowerCase());
  
  // Map of parameter types to their descriptions
  const paramTypeDescriptions: Record<string, {
    description: string;
    impact: 'high' | 'medium' | 'low';
    valueAnalysis: string;
    alternatives: AlternativeValue[];
    bestPractices: string;
    tradeoffs: string;
  }> = {
    // Learning rate
    'learning_rate': {
      description: "Controls how quickly the model updates its weights during training based on the calculated gradients.",
      impact: 'high',
      valueAnalysis: `The value ${paramValue} is a common default that provides a good balance between convergence speed and stability.`,
      alternatives: [
        {
          value: "0.0001",
          description: "More conservative choice that may help with smoother convergence for complex models.",
          type: "lower"
        },
        {
          value: "0.01",
          description: "Faster learning but may cause instability or overshooting of optimal parameters.",
          type: "higher"
        },
        {
          value: "ReduceLROnPlateau",
          description: "Adaptive scheduling that reduces learning rate when metrics plateau.",
          type: "advanced"
        },
        {
          value: "CyclicLR",
          description: "Cycles between low and high learning rates to escape local minima.",
          type: "extreme"
        }
      ],
      bestPractices: "Start with 0.001 for Adam optimizer, or 0.01 for SGD. Monitor validation loss for signs of instability or slow convergence, and adjust accordingly.",
      tradeoffs: "Higher values speed up learning but risk overshooting or divergence. Lower values are more stable but may get stuck in local minima or train too slowly."
    },
    
    // Batch size
    'batch_size': {
      description: "The number of training examples utilized in one iteration of model training.",
      impact: 'medium',
      valueAnalysis: `A batch size of ${paramValue} balances memory requirements with training stability.`,
      alternatives: [
        {
          value: "16",
          description: "Smaller batches provide more frequent updates and better generalization.",
          type: "lower"
        },
        {
          value: "128",
          description: "Larger batches for more stable gradient estimates and better hardware utilization.",
          type: "higher"
        },
        {
          value: "Power of 2",
          description: "Values like 16, 32, 64, 128 optimize GPU memory usage.",
          type: "advanced"
        },
        {
          value: "Gradient Accumulation",
          description: "Simulate larger batches on limited memory hardware.",
          type: "advanced"
        }
      ],
      bestPractices: "Choose the largest batch size that fits in your GPU/TPU memory. Consider reducing batch size if training performance plateaus.",
      tradeoffs: "Larger batches provide more stable gradient estimates but may lead to poorer generalization. Smaller batches can provide regularization effects but increase training time."
    },
    
    // Dropout
    'dropout': {
      description: "A regularization technique that randomly sets a fraction of input units to 0 at each update during training.",
      impact: 'medium',
      valueAnalysis: `A dropout rate of ${paramValue} provides reasonable regularization for medium-sized networks.`,
      alternatives: [
        {
          value: "0.2",
          description: "Less aggressive dropout for simpler models or when using other regularization methods.",
          type: "lower"
        },
        {
          value: "0.7",
          description: "More aggressive dropout for very deep networks prone to overfitting.",
          type: "higher"
        },
        {
          value: "0.0",
          description: "No dropout - use when overfitting is not a concern or for final fine-tuning.",
          type: "extreme"
        },
        {
          value: "SpatialDropout",
          description: "Specialized dropout for convolutional networks, drops entire channels.",
          type: "advanced"
        }
      ],
      bestPractices: "Apply dropout only during training, not during inference. Higher dropout rates often require more training epochs.",
      tradeoffs: "Higher dropout provides stronger regularization but requires longer training and may result in underfitting. Lower dropout may not sufficiently prevent overfitting."
    },
    
    // Epochs
    'epochs': {
      description: "The number of complete passes through the training dataset.",
      impact: 'medium',
      valueAnalysis: `Training for ${paramValue} epochs provides a reasonable training duration for many tasks.`,
      alternatives: [
        {
          value: "5",
          description: "Fewer epochs for simple datasets or when fine-tuning pretrained models.",
          type: "lower"
        },
        {
          value: "100",
          description: "More epochs for complex tasks that require longer learning.",
          type: "higher"
        },
        {
          value: "Early Stopping",
          description: "Use validation performance to determine when to stop training.",
          type: "advanced"
        },
        {
          value: "ReduceLROnPlateau + patience",
          description: "Reduce learning rate when progress plateaus, then stop after no improvement.",
          type: "advanced"
        }
      ],
      bestPractices: "Use early stopping with a validation set to prevent overfitting. Monitor validation metrics to determine optimal training duration.",
      tradeoffs: "More epochs allow better convergence but increase risk of overfitting and computational cost. Fewer epochs may not allow the model to fully learn the data patterns."
    },
    
    // Activation functions
    'activation': {
      description: "Non-linear function applied to the output of a layer to introduce complex patterns into the model.",
      impact: 'medium',
      valueAnalysis: `The '${paramValue}' activation function is a common choice that works well for many neural network architectures.`,
      alternatives: [
        {
          value: "sigmoid",
          description: "Useful for binary classification output layers, maps to range [0,1].",
          type: "advanced"
        },
        {
          value: "tanh",
          description: "Maps to range [-1,1], often used in RNNs.",
          type: "advanced"
        },
        {
          value: "leaky_relu",
          description: "Variant of ReLU that allows small negative values, preventing 'dying ReLU' problem.",
          type: "advanced"
        },
        {
          value: "gelu",
          description: "Gaussian Error Linear Unit, smoother than ReLU, used in transformers.",
          type: "extreme"
        }
      ],
      bestPractices: "Use ReLU or variants for hidden layers in most networks. For output layers, use activation appropriate for the task (softmax for multi-class, sigmoid for binary).",
      tradeoffs: "Different activations have different properties regarding gradient flow, computational efficiency, and expressiveness."
    },
    
    // Optimizer
    'optimizer': {
      description: "Algorithm used to update network weights to minimize the loss function.",
      impact: 'high',
      valueAnalysis: `The '${paramValue}' optimizer is a solid choice that adapts learning rates per parameter.`,
      alternatives: [
        {
          value: "SGD",
          description: "Simple but may require careful tuning of learning rate and momentum.",
          type: "advanced"
        },
        {
          value: "RMSprop",
          description: "Good for RNNs and problems with noisy gradients.",
          type: "advanced"
        },
        {
          value: "AdamW",
          description: "Adam with proper weight decay, often works well for large models.",
          type: "advanced"
        },
        {
          value: "LARS/LAMB",
          description: "Specialized optimizers for large batch training.",
          type: "extreme"
        }
      ],
      bestPractices: "Adam is a good default choice. SGD with momentum often works better for computer vision tasks with sufficient tuning.",
      tradeoffs: "Adaptive optimizers like Adam converge faster but may generalize worse than well-tuned SGD. Different optimizers require different learning rates."
    },
    
    // Loss function
    'loss': {
      description: "Function that measures how well the model's predictions match the ground truth.",
      impact: 'high',
      valueAnalysis: `The '${paramValue}' loss function is appropriate for multi-class classification tasks.`,
      alternatives: [
        {
          value: "binary_crossentropy",
          description: "For binary classification tasks where output is probability.",
          type: "advanced"
        },
        {
          value: "mse",
          description: "Mean Squared Error for regression tasks.",
          type: "advanced"
        },
        {
          value: "focal_loss",
          description: "Modified cross-entropy that focuses on hard examples, good for imbalanced datasets.",
          type: "advanced"
        },
        {
          value: "custom_loss",
          description: "Implement problem-specific loss function for specialized requirements.",
          type: "extreme"
        }
      ],
      bestPractices: "Match loss function to the task type. Use categorical_crossentropy for multi-class, binary_crossentropy for binary, and MSE for regression.",
      tradeoffs: "Different loss functions prioritize different aspects of model performance. Some are more sensitive to outliers or imbalanced classes."
    },
    
    // Default fallback
    'default': {
      description: "A configurable aspect of the machine learning model that affects its behavior and performance.",
      impact: 'medium',
      valueAnalysis: `The value ${paramValue} represents a specific configuration choice for this parameter.`,
      alternatives: [
        {
          value: "Lower value",
          description: "A reduced value might be appropriate for simpler models or to reduce computational requirements.",
          type: "lower"
        },
        {
          value: "Higher value",
          description: "An increased value could provide better model capacity at the cost of more computation.",
          type: "higher"
        },
        {
          value: "Alternative approach",
          description: "Consider a different approach depending on your specific use case.",
          type: "advanced"
        }
      ],
      bestPractices: "Experiment with different values through cross-validation. Consider the specific requirements of your dataset and task.",
      tradeoffs: "Most hyperparameter choices involve trade-offs between model complexity, computational efficiency, and generalization performance."
    }
  };
  
  // Determine if we have a specific description for this parameter type
  let parameterInfo = paramTypeDescriptions[paramName] || paramTypeDescriptions['default'];
  
  // Generate the explanation
  return {
    name: formattedName,
    description: parameterInfo.description,
    impact: parameterInfo.impact,
    valueAnalysis: parameterInfo.valueAnalysis,
    alternatives: parameterInfo.alternatives,
    bestPractices: parameterInfo.bestPractices,
    tradeoffs: parameterInfo.tradeoffs
  };
}