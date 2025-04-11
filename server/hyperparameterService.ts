// server/hyperparameterService.ts
import express, { Request, Response, Router } from 'express';
import { VertexAI } from '@google-cloud/vertexai';

interface ExplainHyperparameterRequest extends Request {
  body: {
    paramName: string;
    paramValue: string;
    framework?: string;
    codeContext?: string;
  }
}

interface DetectHyperparametersRequest extends Request {
  body: {
    code: string;
  }
}

const router: Router = express.Router();

// Do NOT initialize VertexAI here at the top level
let vertexai: VertexAI | null = null;

// Helper function to get or initialize the VertexAI client on demand
function getVertexClient(): VertexAI | null {
  // Return the existing client if already initialized
  if (vertexai) {
    return vertexai;
  }

  // Attempt to initialize only if credentials are now available
  try {
    // Check if auth setup has run and set the necessary env vars
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS && process.env.GOOGLE_PROJECT_ID) {
      vertexai = new VertexAI({
        project: process.env.GOOGLE_PROJECT_ID!,
        location: process.env.GOOGLE_LOCATION || 'us-central1' // Use location from .env if available
      });
      console.log('VertexAI initialized successfully (on demand)');
      return vertexai;
    } else {
      // Log if initialization is attempted but credentials aren't ready
      console.log('Google Cloud credentials not available when initializing VertexAI (on demand)');
      return null;
    }
  } catch (error) {
    console.error('Error initializing VertexAI (on demand):', error);
    vertexai = null; // Ensure client is null on error
    return null;
  }
}

// Check LLM status
router.get('/api/llm/status', (req: Request, res: Response) => {
  const client = getVertexClient(); // Get or initialize client
  res.json({
    available: client !== null,
    provider: 'Google Vertex AI',
    model: 'gemini-pro',
    requiresAuth: true // Indicate that credentials are required for full functionality
  });
});

// Analyze hyperparameter
router.post('/api/llm/explain-hyperparameter', async (req: ExplainHyperparameterRequest, res: Response) => {
  const client = getVertexClient(); // Get or initialize client

  try {
    const { paramName, paramValue, framework, codeContext } = req.body;
    
    if (!paramName) {
      return res.status(400).json({ 
        success: false, 
        error: 'Parameter name is required' 
      });
    }
    
    // If client failed to initialize, use fallback
    if (!client) {
      console.log('Explain API: LLM client not available, using fallback.');
      return res.status(200).json({
        success: false,
        error: 'LLM service not available',
        fallbackAvailable: true // Indicate client can generate its own fallback
      });
    }
    
    try {
      // Create the Gemini model
      const generativeModel = client.getGenerativeModel({
        model: 'gemini-pro',
        generationConfig: {
          maxOutputTokens: 1024,
          temperature: 0.2,
          topP: 0.95,
          topK: 40,
        },
      });
      
      // Construct the prompt
      const prompt = `
You are an expert in machine learning, specifically focusing on model configuration and optimization.
Provide a detailed explanation of the configurable parameter '${paramName}' with value '${paramValue}'
${framework ? `in the ${framework} framework` : ''}.

${codeContext ? `Here is the code context where this parameter appears:\\n\\\`\\\`\\\`\\n${codeContext}\\n\\\`\\\`\\\`\\n` : ''}

This could be any configurable aspect of a machine learning model, including:
- Traditional hyperparameters (learning rate, batch size, etc.)
- Model architecture choices (model type, number of layers)
- Layer configurations (kernel size, filters, strides)
- Activation functions (ReLU, Sigmoid, etc.)
- Loss functions (categorical cross-entropy, MSE)
- Regularization methods (dropout, weight decay)
- Any other configurable value that affects model behavior

Provide the following information:
1. A clear definition of the parameter and its role
2. How this specific value (${paramValue}) impacts the model's behavior and performance
3. Common alternative values and when they would be appropriate
4. Best practices for configuring this parameter
5. Trade-offs to consider when adjusting this parameter

Format your response as JSON with the following structure:
{
  "name": "Formatted name of the parameter",
  "description": "Detailed explanation of what the parameter controls",
  "impact": "high/medium/low - how much this affects model performance",
  "valueAnalysis": "Analysis of the current value ${paramValue}",
  "alternatives": [
    {
      "value": "Alternative value 1",
      "description": "When and why to use this value",
      "type": "higher/lower/advanced/extreme"
    },
    ...more alternatives
  ],
  "bestPractices": "Tips for setting this parameter",
  "tradeoffs": "Key tradeoffs to consider"
}
`;
      
      // Get response from the model
      const result = await generativeModel.generateContent(prompt);
      const response = await result.response;
      
      // Extract the text (which should be JSON)
      let responseText = '';
      if (response.candidates && response.candidates.length > 0 &&
          response.candidates[0].content &&
          response.candidates[0].content.parts) {
        responseText = response.candidates[0].content.parts
          .map(part => part.text || '')
          .join('');
      }

      try {
        // Improved JSON extraction:
        // 1. Look for ```json ... ``` block
        // 2. Look for ``` ... ``` block
        // 3. Look for the first { and last } and extract content between
        let jsonText = '';
        const jsonBlockMatch = responseText.match(/```(?:json)?([\s\S]*?)```/s);
        if (jsonBlockMatch && jsonBlockMatch[1]) {
            jsonText = jsonBlockMatch[1].trim();
        } else {
            // Fallback: Find first '{' and last '}'
            const firstBrace = responseText.indexOf('{');
            const lastBrace = responseText.lastIndexOf('}');
            if (firstBrace !== -1 && lastBrace > firstBrace) {
                jsonText = responseText.substring(firstBrace, lastBrace + 1).trim();
            } else {
                // If no braces found, assume the whole response might be JSON (less likely)
                jsonText = responseText.trim();
            }
        }

        // Attempt to parse the extracted text
        const jsonResponse = JSON.parse(jsonText);

        // Basic validation: check if it's an object with expected keys (optional but good)
        if (typeof jsonResponse === 'object' && jsonResponse !== null && jsonResponse.name && jsonResponse.description) {
            return res.json({
                success: true,
                explanation: jsonResponse
            });
        } else {
            console.warn('LLM explanation response parsed but lacks expected structure. Raw text:', responseText, 'Parsed object:', jsonResponse);
            throw new Error('Parsed JSON structure is invalid'); // Force fallback
        }

      } catch (jsonError) {
        console.error('Error parsing LLM explanation response:', jsonError, 'Raw response text:', responseText);
        return res.status(200).json({ // Return 200 OK with fallback indication
          success: false,
          error: 'Error parsing LLM response',
          rawResponse: responseText, // Send raw response back for potential debugging client-side
          fallbackAvailable: true
        });
      }
    } catch (llmError) {
      console.error('Error calling LLM API for explanation:', llmError);
      return res.status(200).json({ // Return 200 OK with fallback indication
        success: false,
        error: 'Error generating explanation with LLM',
        details: llmError instanceof Error ? llmError.message : String(llmError),
        fallbackAvailable: true
      });
    }
  } catch (error) {
    console.error('Server error in explain route:', error);
    return res.status(500).json({
      success: false,
      error: 'Server error',
      details: error instanceof Error ? error.message : String(error),
      fallbackAvailable: true
    });
  }
});

// Detect hyperparameters in code
router.post('/api/llm/detect-hyperparameters', async (req: DetectHyperparametersRequest, res: Response) => {
  const client = getVertexClient(); // Get or initialize client
  const { code } = req.body;

  if (!code) {
    return res.status(400).json({ 
      success: false, 
      error: 'Missing code to analyze' 
    });
  }
  
  // If client is not available, use fallback regex detection
  if (!client) {
    console.log('Detect API: LLM client not available, using fallback regex.');
    const params = detectBasicHyperparameters(code);
    return res.json({
      success: true,
      parameters: params,
      source: 'fallback'
    });
  }
  
  // Proceed with LLM detection if client is available
  try {
    // Create the Gemini model
    const generativeModel = client.getGenerativeModel({
      model: 'gemini-pro',
      generationConfig: {
        maxOutputTokens: 2048,
        temperature: 0.2,
        topP: 0.8,
        topK: 40,
      },
    });
    
    // Construct the prompt
    const prompt = `
You are an expert machine learning engineer with deep expertise in all major frameworks.
Analyze the following code and identify ALL configurable aspects - this includes not just
traditional hyperparameters, but also architecture choices, layer configurations, activation
functions, optimizers, and any other values that could be modified to affect model behavior.

For each configurable element, provide:
1. The parameter name
2. The current value (as a string)
3. The likely framework (e.g., PyTorch, TensorFlow, Keras, scikit-learn, etc.)
4. The impact level (high, medium, low)
5. A concise explanation of what this parameter does
6. The line number where it appears (if possible, otherwise null)

Focus only on machine learning related parameters, not general programming variables.
Return your response as a JSON array like this:
[
  {
    "name": "parameter_name",
    "value": "current_value",
    "framework": "framework_name",
    "impact": "high|medium|low",
    "explanation": "Brief explanation of what this parameter does",
    "line": <line_number | null>
  },
  ...more parameters...
]

Code to analyze:
\\\`\\\`\\\`
${code}
\\\`\\\`\\\`

Respond with ONLY the JSON array, no other text. If no parameters are found, return an empty array [].
`;
    
    // Get response from the model
    const result = await generativeModel.generateContent(prompt);
    const response = await result.response;
    
    // Extract the text (which should be JSON)
    let responseText = '';
    if (response.candidates && response.candidates.length > 0 &&
        response.candidates[0].content &&
        response.candidates[0].content.parts) {
      responseText = response.candidates[0].content.parts
        .map(part => part.text || '')
        .join('');
    }

    try {
       // Improved JSON Array extraction:
      // 1. Look for ```json ... ``` block
      // 2. Look for ``` ... ``` block
      // 3. Look for the first [ and last ] and extract content between
      let jsonText = '';
      const jsonBlockMatch = responseText.match(/```(?:json)?([\s\S]*?)```/s);
      if (jsonBlockMatch && jsonBlockMatch[1]) {
          jsonText = jsonBlockMatch[1].trim();
      } else {
          // Fallback: Find first '[' and last ']'
          const firstBracket = responseText.indexOf('[');
          const lastBracket = responseText.lastIndexOf(']');
          if (firstBracket !== -1 && lastBracket > firstBracket) {
              jsonText = responseText.substring(firstBracket, lastBracket + 1).trim();
          } else {
               // If no brackets found, assume the whole response might be JSON array (less likely)
              jsonText = responseText.trim();
          }
      }

      // Attempt to parse the extracted text
      const jsonResponse = JSON.parse(jsonText);

      // Basic validation: check if it's an array
      if (Array.isArray(jsonResponse)) {
        return res.json({
          success: true,
          parameters: jsonResponse,
          source: 'llm'
        });
      } else {
         console.warn('LLM detection response parsed but is not an array. Raw text:', responseText, 'Parsed object:', jsonResponse);
         throw new Error('Parsed JSON is not an array'); // Force fallback
      }

    } catch (jsonError) {
      console.error('Error parsing LLM detection response:', jsonError, 'Raw response text:', responseText);
      // Use regex fallback on JSON parsing error
      const params = detectBasicHyperparameters(code);
      return res.json({
        success: true,
        parameters: params,
        source: 'fallback',
        warning: 'LLM response parsing error, used fallback detection.',
        rawResponse: responseText // Send raw response back for debugging
      });
    }
  } catch (llmError) {
    console.error('Error calling LLM API for detection:', llmError);
    // Use regex fallback on LLM error
    const params = detectBasicHyperparameters(code);
    return res.json({
      success: true,
      parameters: params,
      source: 'fallback',
      warning: 'LLM API error, used fallback detection.',
      details: llmError instanceof Error ? llmError.message : String(llmError)
    });
  }
});

// Basic pattern-based hyperparameter detection as a fallback when AI fails
function detectBasicHyperparameters(code: string): { name: string; value: string; framework: string; impact: string; explanation: string; line: number | null }[] {
  const detectedParams: { name: string; value: string; framework: string; impact: string; explanation: string; line: number | null }[] = [];
  const lines = code.split('\\n');

  const framework = detectFramework(code);

  // More comprehensive regex patterns
  const HYPERPARAMETER_PATTERNS: { [key: string]: RegExp } = {
    // Common ML frameworks and parameters
    'learning_rate': /learning_rate\s*=\s*([\d.eE+-]+)/g,
    'batch_size': /batch_size\s*=\s*(\d+)/g,
    'epochs': /\b(?:epochs|n_epochs|num_epochs)\s*=\s*(\d+)/g,
    'optimizer': /optimizer\s*=\s*['\"]?(\w+)['\"]?/g,
    'loss': /loss\s*=\s*['\"]?([\w_]+)['\"]?/g,
    'activation': /activation\s*=\s*['\"]?(\w+)['\"]?/g,
    'dropout': /dropout(?:_rate)?\s*=\s*([\d.]+)/g,
    'kernel_size': /kernel_size\s*=\s*(?:(\(\s*\d+\s*,\s*\d+\s*\))|(\d+))/g,
    'filters': /(?:filters|out_channels)\s*=\s*(\d+)/g,
    'layers': /layers\s*=\s*(\d+)/g, // Simple layer count
    'units': /units\s*=\s*(\d+)/g, // Dense layers
    'n_estimators': /n_estimators\s*=\s*(\d+)/g, // Tree-based models
    'max_depth': /max_depth\s*=\s*(\d+)/g, // Tree-based models
    'C': /\bC\s*=\s*([\d.eE+-]+)/g, // SVM regularization
    'gamma': /gamma\s*=\s*(?:['\"](\w+)['\"]|([\d.eE+-]+))/g, // SVM kernel coefficient
    'kernel': /kernel\s*=\s*['\"](\w+)['\"]/g, // SVM kernel
    'n_neighbors': /n_neighbors\s*=\s*(\d+)/g, // KNN
    'momentum': /momentum\s*=\s*([\d.]+)/g, // Optimizers
    'weight_decay': /weight_decay\s*=\s*([\d.eE+-]+)/g, // Regularization / Optimizers
    'num_classes': /num_classes\s*=\s*(\d+)/g,
    'input_shape': /input_shape\s*=\s*\(.*?\)/g,
    'model_type': /(?:tf\.keras\.models|torch\.nn|sklearn\.\w+)\.(\w+)/g, // Basic model type detection
    'tf.keras.layers.Dense': /tf\.keras\.layers\.Dense\s*\(\s*(\d+)/g,
    'torch.nn.Linear': /torch\.nn\.Linear\s*\(\s*\d+,\s*(\d+)/g,
  };

  for (const [name, pattern] of Object.entries(HYPERPARAMETER_PATTERNS)) {
    let match;
    // Reset lastIndex for global regex
    pattern.lastIndex = 0;
    while ((match = pattern.exec(code)) !== null) {
      const value = match[1] || match[2] || match[0]; // Capture value, handling different capture groups

      // Find line number
      let lineNumber: number | null = null;
      let charIndex = match.index;
      let currentLine = 0;
      for(let i = 0; i < lines.length; i++) {
        if (charIndex < lines[i].length + 1) { // +1 for newline character
          lineNumber = i + 1;
          break;
        }
        charIndex -= (lines[i].length + 1);
        currentLine++;
      }

      // Avoid duplicates based on name and line number
      if (!detectedParams.some(p => p.name === name && p.line === lineNumber)) {
        detectedParams.push({
          name: name,
          value: String(value).trim(),
          framework: framework,
          impact: 'medium', // Default impact for basic detection
          explanation: `Detected configuration for ${name}`,
          line: lineNumber
        });
      }
    }
  }
  return detectedParams;
}

// Detect the likely framework based on code patterns
function detectFramework(code: string, defaultFramework: string = 'Unknown'): string {
  if (code.includes('import torch') || code.includes('torch.nn')) return 'PyTorch';
  if (code.includes('import tensorflow') || code.includes('tf.keras')) return 'TensorFlow/Keras';
  if (code.includes('import sklearn') || code.includes('from sklearn')) return 'Scikit-learn';
  if (code.includes('import jax') || code.includes('jax.numpy')) return 'JAX';
  // Add more framework detection logic if needed
  return defaultFramework;
}

export default router;