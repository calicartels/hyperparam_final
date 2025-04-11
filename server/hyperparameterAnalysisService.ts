// server/hyperparameterAnalysisService.ts
import express, { Request, Response, Router } from 'express';
import { VertexAI } from '@google-cloud/vertexai';

// Interface for the parameter analysis request
interface AnalyzeHyperparameterRequest extends Request {
  body: {
    paramName: string;
    paramValue: string;
    framework?: string;
    codeContext?: string;
  }
}

// Create the router
const router: Router = express.Router();

// Initialize Vertex AI on demand
let vertexai: VertexAI | null = null;

// Helper function to get or initialize the VertexAI client
function getVertexClient(): VertexAI | null {
  // Return existing client if already initialized
  if (vertexai) {
    return vertexai;
  }

  // Attempt to initialize only if credentials are available
  try {
    // Check if auth setup has run and set the necessary env vars
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS && process.env.GOOGLE_PROJECT_ID) {
      vertexai = new VertexAI({
        project: process.env.GOOGLE_PROJECT_ID!,
        location: process.env.GOOGLE_LOCATION || 'us-central1'
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

// Route to check LLM status
router.get('/api/llm/status', (req: Request, res: Response) => {
  const client = getVertexClient();
  res.json({
    available: client !== null,
    provider: 'Google Vertex AI',
    model: 'gemini-pro',
    requiresAuth: true
  });
});

// Route to analyze hyperparameters in detail
router.post('/api/llm/analyze-hyperparameter', async (req: AnalyzeHyperparameterRequest, res: Response) => {
  const client = getVertexClient();
  
  try {
    const { paramName, paramValue, framework, codeContext } = req.body;
    
    if (!paramName) {
      return res.status(400).json({ 
        success: false, 
        error: 'Parameter name is required' 
      });
    }
    
    // If client failed to initialize, return error
    if (!client) {
      console.log('Analysis API: LLM client not available, using fallback.');
      return res.status(200).json({
        success: false,
        error: 'LLM service not available',
        fallbackAvailable: true
      });
    }
    
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
      
      // Construct the prompt for hyperparameter analysis
      const prompt = `
You are an expert machine learning engineer who specializes in hyperparameter tuning and neural network optimization.

Analyze the hyperparameter '${paramName}' with value '${paramValue}'
${framework ? `in the ${framework} framework.` : '.'}

${codeContext ? `Here is the code context where this parameter appears:\n\`\`\`\n${codeContext}\n\`\`\`\n` : ''}

I need a detailed analysis with two parts:
1. General parameter information and explanation
2. Visualization configuration to help understand the impact of this parameter

Respond with a detailed JSON object with the following structure:

\`\`\`json
{
  "name": "Readable parameter name",
  "type": "Parameter type (e.g., learning rate, batch size)",
  "defaultValue": "${paramValue}",
  "range": [minimum_value, maximum_value],
  "impact": "high/medium/low",
  "description": "Detailed explanation of what this parameter does and how it affects the model",
  "affectsTraining": true/false,
  "affectsArchitecture": true/false,
  "affectsRegularization": true/false,
  
  "visualizationConfig": {
    "chartType": "line/bar/radar/scatter",
    "title": "Chart title",
    "xAxisLabel": "X-axis label",
    "yAxisLabel": "Y-axis label",
    "datasets": [
      {
        "label": "Dataset label",
        "data": [values for the chart],
        "borderColor": "rgba color string",
        "backgroundColor": "rgba color string",
        "fill": true/false
      }
    ],
    "labels": ["Label 1", "Label 2", ...],
    "description": "Description of what the visualization shows",
    "insights": ["Key insight 1", "Key insight 2", ...],
    "recommendations": [
      {
        "value": "Recommended value",
        "reason": "Reason for recommendation"
      }
    ]
  },
  
  "networkConfig": {
    "layers": [neurons_in_layer_1, neurons_in_layer_2, ...],
    "activationFunction": "relu/sigmoid/tanh",
    "dropoutRate": dropout_value,
    "connectionStrength": connection_strength,
    "signalSpeed": signal_speed,
    "description": "Description of network visualization"
  }
}
\`\`\`

For the visualization configuration:
- Chart data should realistically represent how this parameter affects model performance
- Network configuration should show how this parameter affects neural network behavior
- Include realistic values that would be appropriate for the parameter type
- For learning rate: use log scale and show optimal range, too small/large effects
- For batch size: show impact on memory, training speed, and generalization
- For dropout: show effect on overfitting/training vs. validation performance
- For epochs: show learning curve with potential overfitting
- For optimizers: compare convergence profiles
- For activations: show impacts on gradient flow

Provide comprehensive insights and practical recommendations based on your expertise.
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
        // Extract JSON from the response
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
            jsonText = responseText.trim();
          }
        }

        // Parse the JSON response
        const analysisResponse = JSON.parse(jsonText);

        // Validate required fields
        if (typeof analysisResponse === 'object' && 
            analysisResponse !== null && 
            analysisResponse.name && 
            analysisResponse.visualizationConfig && 
            analysisResponse.networkConfig) {
          
          return res.json({
            success: true,
            analysis: analysisResponse
          });
        } else {
          console.warn('LLM analysis response parsed but lacks expected structure:', analysisResponse);
          throw new Error('Parsed JSON structure is invalid');
        }
      } catch (jsonError) {
        console.error('Error parsing LLM analysis response:', jsonError);
        console.error('Raw response text:', responseText);
        
        return res.status(200).json({
          success: false,
          error: 'Error parsing LLM response',
          rawResponse: responseText.substring(0, 500), // Send partial raw response for debugging
          fallbackAvailable: true
        });
      }
    } catch (llmError) {
      console.error('Error calling LLM API for analysis:', llmError);
      
      return res.status(200).json({
        success: false,
        error: 'Error generating analysis with LLM',
        details: llmError instanceof Error ? llmError.message : String(llmError),
        fallbackAvailable: true
      });
    }
  } catch (error) {
    console.error('Server error in analysis route:', error);
    
    return res.status(500).json({
      success: false,
      error: 'Server error',
      details: error instanceof Error ? error.message : String(error),
      fallbackAvailable: true
    });
  }
});

export default router;