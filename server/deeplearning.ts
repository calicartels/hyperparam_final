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

    // Generate content using Gemini
    const result = await generativeModel.generateContent(prompt);
    const response = await result.response;
    const textContent = response.candidates?.[0]?.content?.parts?.[0]?.text || '';
    
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