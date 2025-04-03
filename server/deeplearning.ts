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
You are an expert machine learning engineer specializing in code analysis and optimization.
Please analyze the following code and identify ALL configurable aspects, including:
1. Traditional hyperparameters (learning rate, batch size, etc.)
2. Model architecture choices (model type, architecture)
3. Layer configurations (kernel size, filters, strides, padding)
4. Activation functions
5. Loss functions
6. Regularization methods
7. Input/output shapes
8. Data preprocessing options
9. Callbacks and training configurations
10. Any other parameters that could be modified to change model behavior

CODE:
\`\`\`
${code}
\`\`\`

For each configurable aspect found, provide:
1. The parameter name (use snake_case format)
2. The parameter value (as it appears in the code)
3. The framework it belongs to (e.g., tensorflow, pytorch, scikit-learn, etc.)
4. The impact level (high, medium, or low)
5. A brief explanation of its purpose and how it affects the model

Return your response in JSON format with the following structure:
{
  "parameters": [
    {
      "name": "parameter_name",
      "value": "parameter_value",
      "framework": "framework_name",
      "impact": "high|medium|low",
      "explanation": "Brief explanation"
    },
    ...
  ]
}

Be comprehensive and thorough. Identify ALL configurable aspects, not just traditional hyperparameters.
Do not include any other text besides the JSON response.
`;

    // Basic pattern-based hyperparameter detection as fallback
    const basicDetection = [
      // Common hyperparameters with regex patterns
      ...detectBasicHyperparameters(code)
    ];

    let textContent = '';
    
    try {
      // Generate content using Gemini
      const result = await generativeModel.generateContent(prompt);
      const response = await result.response;
      textContent = response.candidates?.[0]?.content?.parts?.[0]?.text || '';
      
      // Add additional logging for diagnostics
      console.log(`AI response status: ${response.candidates?.[0]?.finishReason || 'unknown'}`);
    } catch (apiError) {
      console.error('API call to Gemini failed:', apiError);
      // Use basic detection as fallback
      return {
        success: true,
        parameters: basicDetection,
        error: `API call failed, using basic detection: ${apiError instanceof Error ? apiError.message : String(apiError)}`
      };
    }
    
    if (!textContent) {
      console.warn('Empty response from AI model, using fallback detection');
      return {
        success: true,
        parameters: basicDetection,
        error: 'Empty response from AI model, using fallback detection'
      };
    }
    
    // Extract JSON from the response (handle potential text wrapping by the model)
    const jsonMatch = textContent.match(/\{[\s\S]*\}/);
    
    if (!jsonMatch) {
      console.error('Failed to parse AI response as JSON');
      return {
        success: false,
        parameters: [],
        error: 'Failed to parse AI response as JSON'
      };
    }
    
    const jsonText = jsonMatch[0];
    const parsedData = JSON.parse(jsonText);
    
    if (!parsedData.parameters || !Array.isArray(parsedData.parameters)) {
      console.error('AI response did not contain parameters array');
      return {
        success: false,
        parameters: [],
        error: 'AI response did not contain parameters array'
      };
    }
    
    return {
      success: true,
      parameters: parsedData.parameters
    };
  } catch (error) {
    console.error('Error detecting hyperparameters with AI:', error);
    return {
      success: false,
      parameters: [],
      error: `AI model error: ${error instanceof Error ? error.message : String(error)}`
    };
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