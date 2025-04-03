import { VertexAI } from '@google-cloud/vertexai';
import { setupGoogleCloudCredentials } from './auth';

// Initialize Vertex AI with proper credentials
setupGoogleCloudCredentials();
const vertexAI = new VertexAI({
  project: process.env.GOOGLE_PROJECT_ID || 'capstone-449418',
  location: process.env.GOOGLE_LOCATION || 'us-central1'
});

// Get the Vertex AI generative model
const generativeModel = vertexAI.preview.getGenerativeModel({
  model: 'gemini-pro',
  generationConfig: {
    temperature: 0.2,
    topP: 0.8,
    topK: 40,
    maxOutputTokens: 2048,
  },
});

/**
 * Basic pattern-based hyperparameter detection as a fallback when AI fails
 * 
 * @param code The code to analyze
 * @returns Array of detected hyperparameters with basic information
 */
function detectBasicHyperparameters(code: string): Array<{
  name: string;
  value: string;
  framework: string;
  impact: string;
  explanation: string;
}> {
  const detected = [];
  
  // Common hyperparameter patterns
  const patterns = [
    { 
      regex: /learning_?rate\s*=\s*([0-9.e-]+)/i, 
      name: 'learning_rate',
      framework: 'common',
      impact: 'high',
      explanation: 'Controls the step size during optimization'
    },
    { 
      regex: /batch_?size\s*=\s*(\d+)/i, 
      name: 'batch_size',
      framework: 'common',
      impact: 'medium',
      explanation: 'Number of samples processed before model update'
    },
    { 
      regex: /epochs\s*=\s*(\d+)/i, 
      name: 'epochs',
      framework: 'common',
      impact: 'medium',
      explanation: 'Number of complete passes through the dataset'
    },
    { 
      regex: /dropout\s*=\s*([0-9.]+)/i, 
      name: 'dropout',
      framework: 'common',
      impact: 'medium',
      explanation: 'Regularization technique to prevent overfitting'
    },
    { 
      regex: /optimizer\s*=\s*['"](.*?)['"]/i, 
      name: 'optimizer',
      framework: 'common',
      impact: 'high',
      explanation: 'Algorithm used to update network weights'
    },
    { 
      regex: /kernel_size\s*=\s*[\(\[]?(\d+)[\)\]]?/i, 
      name: 'kernel_size',
      framework: 'common',
      impact: 'medium',
      explanation: 'Size of convolutional filter'
    },
    { 
      regex: /activation\s*=\s*['"](.*?)['"]/i, 
      name: 'activation',
      framework: 'common',
      impact: 'medium',
      explanation: 'Non-linear function applied to layer outputs'
    },
    { 
      regex: /loss\s*=\s*['"](.*?)['"]/i, 
      name: 'loss',
      framework: 'common',
      impact: 'high',
      explanation: 'Function that measures model error'
    },
    { 
      regex: /momentum\s*=\s*([0-9.]+)/i, 
      name: 'momentum',
      framework: 'common',
      impact: 'medium',
      explanation: 'Accelerates gradient descent in relevant direction'
    },
    { 
      regex: /weight_decay\s*=\s*([0-9.e-]+)/i, 
      name: 'weight_decay',
      framework: 'common',
      impact: 'medium',
      explanation: 'L2 regularization parameter'
    }
  ];
  
  // Check for each pattern
  for (const pattern of patterns) {
    const matches = code.match(new RegExp(pattern.regex, 'g'));
    
    if (matches) {
      for (const match of matches) {
        const valueMatch = match.match(pattern.regex);
        if (valueMatch && valueMatch[1]) {
          detected.push({
            name: pattern.name,
            value: valueMatch[1],
            framework: detectFramework(code, pattern.framework),
            impact: pattern.impact,
            explanation: pattern.explanation
          });
        }
      }
    }
  }
  
  return detected;
}

/**
 * Detect the likely framework based on code patterns
 * 
 * @param code The code to analyze
 * @param defaultFramework Default framework to return if none detected
 * @returns Detected framework name
 */
function detectFramework(code: string, defaultFramework: string): string {
  if (code.includes('import torch') || code.includes('import torch.nn')) {
    return 'pytorch';
  }
  if (code.includes('import tensorflow') || code.includes('import tf')) {
    return 'tensorflow';
  }
  if (code.includes('import keras')) {
    return 'keras';
  }
  if (code.includes('import sklearn') || code.includes('from sklearn')) {
    return 'scikit-learn';
  }
  if (code.includes('import jax')) {
    return 'jax';
  }
  if (code.includes('import mxnet')) {
    return 'mxnet';
  }
  if (code.includes('import xgboost')) {
    return 'xgboost';
  }
  if (code.includes('import lightgbm')) {
    return 'lightgbm';
  }
  return defaultFramework;
}

/**
 * Interface for the hyperparameter detection result
 */
interface HyperparameterDetectionResult {
  success: boolean;
  parameters: Array<{
    name: string;
    value: string;
    framework: string;
    impact: string;
    explanation: string;
  }>;
  error?: string;
}

/**
 * Detect hyperparameters and all configurable aspects in code using deep learning
 * 
 * This function uses the Gemini LLM to analyze code and identify all configurable aspects
 * (hyperparameters, model choices, layer configurations, etc.) with their values,
 * likely framework, and importance.
 * 
 * @param code The code to analyze
 * @returns Promise with detected parameters
 */
export async function detectHyperparametersWithAI(code: string): Promise<HyperparameterDetectionResult> {
  try {
    const prompt = `
You are an expert machine learning engineer with deep expertise in all major frameworks.
Analyze the following code and identify ALL configurable aspects - this includes not just
traditional hyperparameters, but also architecture choices, layer configurations, activation
functions, optimizers, and any other values that could be modified to affect model behavior.

For each configurable element, provide:
1. The parameter name
2. The current value
3. The likely framework (e.g., PyTorch, TensorFlow, Keras, etc.)
4. The impact level (high, medium, low)
5. A concise explanation of what this parameter does

Focus only on machine learning related parameters, not general programming variables.
Return your response as a JSON array like this:
[
  {
    "name": "parameter_name",
    "value": "current_value",
    "framework": "framework_name",
    "impact": "high|medium|low",
    "explanation": "Brief explanation of what this parameter does"
  },
  ...more parameters...
]

Code to analyze:
\`\`\`
${code}
\`\`\`

Respond with ONLY the JSON array, no other text.
`;

    // Generate content using Gemini
    const result = await generativeModel.generateContent(prompt);
    const response = await result.response;
    const textContent = response.candidates?.[0]?.content?.parts?.[0]?.text || '';
    
    // Extract JSON from the response
    const jsonMatch = textContent.match(/\[[\s\S]*\]/);
    
    if (!jsonMatch) {
      console.error('Failed to parse AI response as JSON array');
      // Fall back to basic pattern-based detection
      const detectedParams = detectBasicHyperparameters(code);
      return {
        success: true,
        parameters: detectedParams,
      };
    }
    
    const jsonText = jsonMatch[0];
    const parsedData = JSON.parse(jsonText);
    
    if (!Array.isArray(parsedData)) {
      console.error('AI response is not a valid array');
      // Fall back to basic detection
      const detectedParams = detectBasicHyperparameters(code);
      return {
        success: true,
        parameters: detectedParams,
      };
    }
    
    return {
      success: true,
      parameters: parsedData
    };
  } catch (error) {
    console.error('Error detecting hyperparameters with AI:', error);
    
    // Fall back to basic pattern-based detection on AI failure
    try {
      const detectedParams = detectBasicHyperparameters(code);
      return {
        success: true,
        parameters: detectedParams,
      };
    } catch (fallbackError) {
      return {
        success: false,
        parameters: [],
        error: `AI model error: ${error instanceof Error ? error.message : String(error)}`
      };
    }
  }
}

/**
 * Get alternative suggestions for any configurable aspect using AI
 * 
 * @param paramName Parameter name (can be hyperparameter, model choice, layer config, etc.)
 * @param paramValue Current value
 * @param framework Framework name
 * @returns Promise with alternative suggestions
 */
export async function getAlternativeHyperparameters(
  paramName: string,
  paramValue: string,
  framework: string
): Promise<any> {
  try {
    const prompt = `
You are an expert machine learning engineer specializing in model optimization.
Suggest alternative values for the "${paramName}" parameter (current value: ${paramValue}) 
in the ${framework} framework.

The parameter could be any configurable aspect of a machine learning model, including:
- Traditional hyperparameters (learning rate, batch size, etc.)
- Model architecture choices
- Layer configurations
- Activation functions
- Loss functions
- Regularization settings
- Input/output configurations
- Optimization algorithms
- Data preprocessing options
- Any other configurable value that affects model behavior

For each suggested alternative, provide:
1. The alternative value (be specific and use correct syntax)
2. A brief explanation of why this might be better and how it affects model behavior
3. The type of change (higher, lower, advanced, extreme)
   - "higher": for increasing numeric values
   - "lower": for decreasing numeric values
   - "advanced": for different but common options
   - "extreme": for unusual or experimental options

Return your response in JSON format with the following structure:
{
  "alternatives": [
    {
      "value": "alternative_value",
      "explanation": "Brief explanation of why this might be better",
      "type": "higher|lower|advanced|extreme"
    },
    ...
  ]
}

Provide 3-4 meaningful alternatives that represent different approaches or strategies.
Do not include any other text besides the JSON response.
`;

    // Generate content using Gemini
    const result = await generativeModel.generateContent(prompt);
    const response = await result.response;
    const textContent = response.candidates?.[0]?.content?.parts?.[0]?.text || '';
    
    // Extract JSON from the response
    const jsonMatch = textContent.match(/\{[\s\S]*\}/);
    
    if (!jsonMatch) {
      console.error('Failed to parse AI response as JSON');
      return {
        success: false,
        alternatives: [],
        error: 'Failed to parse AI response as JSON'
      };
    }
    
    const jsonText = jsonMatch[0];
    const parsedData = JSON.parse(jsonText);
    
    if (!parsedData.alternatives || !Array.isArray(parsedData.alternatives)) {
      console.error('AI response did not contain alternatives array');
      return {
        success: false,
        alternatives: [],
        error: 'AI response did not contain alternatives array'
      };
    }
    
    return {
      success: true,
      alternatives: parsedData.alternatives
    };
  } catch (error) {
    console.error('Error getting alternative hyperparameters with AI:', error);
    return {
      success: false,
      alternatives: [],
      error: `AI model error: ${error instanceof Error ? error.message : String(error)}`
    };
  }
}