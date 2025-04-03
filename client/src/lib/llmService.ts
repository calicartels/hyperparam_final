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
  const response = await apiRequest<LLMStatusResponse>({
    method: "GET",
    url: "/api/llm/status",
    on401: "returnNull",
  });
  
  return response;
}

/**
 * Get a hyperparameter explanation from the LLM service
 * @param request Hyperparameter explanation request
 * @returns Promise with explanation
 */
export async function getHyperparameterExplanation(
  request: ExplainHyperparameterRequest
): Promise<LLMExplanationResponse> {
  const response = await apiRequest<LLMExplanationResponse>({
    method: "POST",
    url: "/api/llm/explain-hyperparameter",
    body: request,
    on401: "returnNull",
  });
  
  return response;
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
    // Model architecture parameters
    'model_architecture': {
      description: "The overall structure and type of the machine learning model being used.",
      impact: 'high',
      valueAnalysis: `The "${paramValue}" architecture is a design choice that defines how the model processes data.`,
      alternatives: [
        {
          value: "Sequential",
          description: "Simple layer-by-layer architecture for straightforward tasks",
          type: "advanced"
        },
        {
          value: "Functional API",
          description: "More flexible architecture allowing complex layer connections",
          type: "advanced"
        }
      ],
      bestPractices: "Choose an architecture that matches the complexity of your problem. Simpler architectures are easier to debug and often sufficient for many tasks.",
      tradeoffs: "More complex architectures can capture more intricate patterns but require more data, compute resources, and are harder to debug."
    },
    
    // Loss function parameters
    'loss_function': {
      description: "The function used to measure how well the model performs, which the training process aims to minimize.",
      impact: 'high',
      valueAnalysis: `The "${paramValue}" loss function is appropriate for certain types of prediction tasks.`,
      alternatives: [
        {
          value: "categorical_crossentropy",
          description: "Standard loss for multi-class classification",
          type: "advanced"
        },
        {
          value: "mse",
          description: "Mean Squared Error for regression tasks",
          type: "advanced"
        }
      ],
      bestPractices: "Match your loss function to your task type: classification, regression, ranking, etc.",
      tradeoffs: "Some loss functions are more sensitive to outliers or class imbalance than others."
    },
    
    // Layer configuration parameters
    'kernel_size': {
      description: "The size of the convolutional filter that slides over the input data.",
      impact: 'medium',
      valueAnalysis: `A kernel size of ${paramValue} determines the receptive field of each neuron in a convolutional layer.`,
      alternatives: [
        {
          value: "3",
          description: "Standard size capturing local features",
          type: "advanced"
        },
        {
          value: "5",
          description: "Larger receptive field for broader feature capture",
          type: "higher"
        }
      ],
      bestPractices: "Smaller kernels (3x3) are often preferred with deeper networks, while larger kernels may be useful in early layers.",
      tradeoffs: "Larger kernels capture more spatial context but increase computational cost and may lead to overfitting."
    },
    
    'conv_filters': {
      description: "The number of different filters (feature detectors) in a convolutional layer.",
      impact: 'medium',
      valueAnalysis: `Using ${paramValue} filters allows the model to detect that many different features at this layer.`,
      alternatives: [
        {
          value: "32",
          description: "Fewer parameters, faster but less expressive",
          type: "lower"
        },
        {
          value: "128",
          description: "More features, potentially better recognition",
          type: "higher"
        }
      ],
      bestPractices: "Typically, the number of filters increases as you go deeper into the network.",
      tradeoffs: "More filters can detect more features but increase computational cost and may lead to overfitting with limited data."
    },
    
    // RNN Parameters
    'bidirectional': {
      description: "Whether a recurrent neural network processes sequences in both forward and backward directions.",
      impact: 'high',
      valueAnalysis: `Bidirectional processing is set to ${paramValue}, affecting how the model captures context from sequences.`,
      alternatives: [
        {
          value: "True",
          description: "Process data in both directions for better context understanding",
          type: "advanced"
        },
        {
          value: "False",
          description: "Process only forward, more suitable for time series or streaming data",
          type: "advanced"
        }
      ],
      bestPractices: "Use bidirectional for tasks where future context is available and relevant (like document classification).",
      tradeoffs: "Bidirectional processing doubles the computational cost but often improves performance for NLP tasks."
    },
    
    // Normalization Parameters
    'normalization_mean': {
      description: "Mean values used for normalizing input data, often channel-wise for images.",
      impact: 'medium',
      valueAnalysis: `The normalization mean values ${paramValue} help center the data distribution.`,
      alternatives: [
        {
          value: "[0.485, 0.456, 0.406]",
          description: "ImageNet RGB means - good for transfer learning",
          type: "advanced"
        },
        {
          value: "[0.5, 0.5, 0.5]",
          description: "Simple centering around zero",
          type: "advanced"
        }
      ],
      bestPractices: "Use dataset-specific means when possible, or standard values for transfer learning.",
      tradeoffs: "Proper normalization improves training stability and convergence speed."
    },
    
    // Output Configuration
    'num_classes': {
      description: "The number of categories or classes the model is designed to predict.",
      impact: 'high',
      valueAnalysis: `The model is configured to predict ${paramValue} different classes.`,
      alternatives: [
        {
          value: "Dataset-specific",
          description: "Must match your specific dataset classes",
          type: "advanced"
        },
        {
          value: "Multi-label",
          description: "Consider multi-label classification if items can belong to multiple classes",
          type: "advanced"
        }
      ],
      bestPractices: "This parameter must match the actual number of categories in your dataset.",
      tradeoffs: "More classes generally require more model capacity and training data."
    },
    
    // Default fallback for numeric parameters
    'numeric': {
      description: "A parameter that controls an aspect of model training or architecture.",
      impact: 'medium',
      valueAnalysis: `The value ${paramValue} is a common setting that provides a balance of performance and generalization.`,
      alternatives: [
        {
          value: isNaN(Number(paramValue)) ? "Lower value" : (Number(paramValue) * 0.5).toString(),
          description: "A lower value may provide better generalization but slower training.",
          type: "lower"
        },
        {
          value: isNaN(Number(paramValue)) ? "Higher value" : (Number(paramValue) * 2).toString(),
          description: "A higher value may lead to faster training but could reduce generalization.",
          type: "higher"
        }
      ],
      bestPractices: "It's typically recommended to start with the default value and adjust based on validation performance.",
      tradeoffs: "Modifying this parameter often involves a tradeoff between training speed, model performance, and generalization ability."
    },
    
    // Default fallback for boolean or string parameters
    'categorical': {
      description: "A configuration choice that affects model behavior.",
      impact: 'medium',
      valueAnalysis: `The choice "${paramValue}" is one of several possible configurations for this setting.`,
      alternatives: [
        {
          value: paramValue === "True" ? "False" : "True",
          description: "The alternative setting may be more appropriate for different use cases.",
          type: "advanced"
        },
        {
          value: "Custom configuration",
          description: "For advanced use cases, a custom configuration might be optimal.",
          type: "extreme"
        }
      ],
      bestPractices: "Consider the specific requirements of your task when choosing this configuration option.",
      tradeoffs: "Different settings offer tradeoffs in terms of model complexity, training speed, and performance."
    }
  };
  
  // Determine if we have a specific description for this parameter type
  let paramType = paramName;
  
  // If we don't have a specific entry, determine if it's numeric or categorical
  if (!paramTypeDescriptions[paramType]) {
    if (!isNaN(Number(paramValue)) || paramValue.includes('[') || paramValue.includes('.')) {
      paramType = 'numeric';
    } else {
      paramType = 'categorical';
    }
  }
  
  // Get the description for this parameter type
  const paramDesc = paramTypeDescriptions[paramType];
  
  // Generate the explanation
  return {
    name: formattedName,
    description: paramDesc.description,
    impact: paramDesc.impact,
    valueAnalysis: paramDesc.valueAnalysis,
    alternatives: paramDesc.alternatives,
    bestPractices: paramDesc.bestPractices,
    tradeoffs: paramDesc.tradeoffs
  };
}