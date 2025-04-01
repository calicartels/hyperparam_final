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
 * Detect hyperparameters in code using deep learning
 * 
 * This function uses the Gemini LLM to analyze code and identify hyperparameters
 * with their values, likely framework, and importance.
 * 
 * @param code The code to analyze
 * @returns Promise with detected hyperparameters
 */
export async function detectHyperparametersWithAI(code: string): Promise<HyperparameterDetectionResult> {
  try {
    const prompt = `
You are an expert machine learning engineer specializing in hyperparameter optimization.
Please analyze the following code and identify all hyperparameters with their values.

CODE:
\`\`\`
${code}
\`\`\`

For each hyperparameter found, provide:
1. The parameter name
2. The parameter value
3. The framework it belongs to (e.g., tensorflow, pytorch, scikit-learn, etc.)
4. The impact level (high, medium, or low)
5. A brief explanation of its purpose

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
 * Get alternative hyperparameter suggestions using AI
 * 
 * @param paramName Hyperparameter name
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
You are an expert machine learning engineer specializing in hyperparameter optimization.
Suggest alternative values for the "${paramName}" hyperparameter (current value: ${paramValue}) 
in the ${framework} framework.

For each suggested alternative, provide:
1. The alternative value
2. A brief explanation of why this might be better
3. The type of change (higher, lower, advanced)

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