import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
import { VertexAI } from "@google-cloud/vertexai";

// Initialize Vertex AI
let vertexai: VertexAI | null = null;

// This function will initialize Vertex AI if credentials are available
function initializeVertexAI() {
  try {
    // Check if we have the required environment variables
    if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      const projectId = process.env.GOOGLE_PROJECT_ID || "";
      const location = process.env.GOOGLE_LOCATION || "us-central1";
      
      if (!projectId) {
        console.log("Google Cloud Project ID not provided");
        return false;
      }
      
      vertexai = new VertexAI({
        project: projectId,
        location: location
      });
      
      console.log("VertexAI initialized successfully");
      return true;
    } else {
      console.log("Google Cloud credentials not provided");
      return false;
    }
  } catch (error) {
    console.error("Error initializing VertexAI:", error);
    return false;
  }
}

export async function registerRoutes(app: Express): Promise<Server> {
  // API route to get all hyperparameters
  app.get("/api/hyperparameters", async (req, res) => {
    try {
      const hyperparameters = await storage.getAllHyperparameters();
      res.json(hyperparameters);
    } catch (error) {
      console.error("Error fetching hyperparameters:", error);
      res.status(500).json({ message: "Failed to fetch hyperparameters" });
    }
  });

  // API route to get a specific hyperparameter by key
  app.get("/api/hyperparameters/:paramKey", async (req, res) => {
    try {
      const paramKey = req.params.paramKey;
      const hyperparameter = await storage.getHyperparameterByKey(paramKey);
      
      if (!hyperparameter) {
        return res.status(404).json({ message: "Hyperparameter not found" });
      }
      
      res.json(hyperparameter);
    } catch (error) {
      console.error(`Error fetching hyperparameter with key ${req.params.paramKey}:`, error);
      res.status(500).json({ message: "Failed to fetch hyperparameter" });
    }
  });

  // API route to get hyperparameters by framework
  app.get("/api/frameworks/:framework/hyperparameters", async (req, res) => {
    try {
      const framework = req.params.framework;
      const hyperparameters = await storage.getHyperparametersByFramework(framework);
      res.json(hyperparameters);
    } catch (error) {
      console.error(`Error fetching hyperparameters for framework ${req.params.framework}:`, error);
      res.status(500).json({ message: "Failed to fetch framework hyperparameters" });
    }
  });

  // API route to get all supported frameworks
  app.get("/api/frameworks", async (req, res) => {
    try {
      const frameworks = await storage.getAllFrameworks();
      res.json(frameworks);
    } catch (error) {
      console.error("Error fetching frameworks:", error);
      res.status(500).json({ message: "Failed to fetch frameworks" });
    }
  });

  // Route to check if LLM integration is available
  app.get("/api/llm/status", (_req: Request, res: Response) => {
    // Try to initialize if not already done
    if (!vertexai) {
      initializeVertexAI();
    }
    
    res.json({
      available: vertexai !== null,
      provider: "Google Vertex AI",
      model: "gemini-pro",
      requiresAuth: true
    });
  });

  // Route to get hyperparameter explanation from LLM
  app.post("/api/llm/explain-hyperparameter", async (req: Request, res: Response) => {
    try {
      const { paramName, paramValue, framework, codeContext } = req.body;
      
      if (!paramName) {
        return res.status(400).json({ error: "Parameter name is required" });
      }
      
      // Check if Vertex AI is initialized
      if (!vertexai) {
        const initialized = initializeVertexAI();
        if (!initialized) {
          return res.status(503).json({
            error: "LLM service not available. Please provide Google Cloud credentials.",
            fallbackAvailable: true
          });
        }
      }
      
      try {
        // Create the Gemini model
        const generativeModel = vertexai!.getGenerativeModel({
          model: "gemini-pro",
          generationConfig: {
            maxOutputTokens: 1024,
            temperature: 0.2,
            topP: 0.95,
            topK: 40,
          },
        });
        
        // Construct the prompt
        const prompt = `
You are an expert in machine learning, specifically focusing on hyperparameters.
Provide a detailed explanation of the hyperparameter '${paramName}' with value '${paramValue}'
${framework ? `in the ${framework} framework` : ""}.

${codeContext ? `Here is the code context where this hyperparameter appears:\n\`\`\`\n${codeContext}\n\`\`\`\n` : ""}

Provide the following information:
1. A clear definition of the hyperparameter
2. How this specific value (${paramValue}) impacts model training and performance
3. Common alternative values and when they would be appropriate
4. Best practices for tuning this hyperparameter
5. Trade-offs to consider when adjusting this hyperparameter

Format your response as JSON with the following structure:
{
  "name": "Formatted name of the hyperparameter",
  "description": "Detailed explanation of what the hyperparameter controls",
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
  "bestPractices": "Tips for setting this hyperparameter",
  "tradeoffs": "Key tradeoffs to consider"
}
`;
        
        // Get response from the model
        const result = await generativeModel.generateContent(prompt);
        const response = await result.response;
        
        // Extract the text (which should be JSON)
        // In Vertex AI, we get the text from the response parts
        let responseText = "";
        if (response.candidates && response.candidates.length > 0 && 
            response.candidates[0].content && 
            response.candidates[0].content.parts) {
          responseText = response.candidates[0].content.parts
            .map(part => part.text || "")
            .join("");
        }
        
        let jsonResponse;
        
        try {
          // Parse the JSON from the response
          // Sometimes the model might include markdown code blocks, so we need to handle that
          const jsonMatch = responseText.match(/```(?:json)?(.*?)```/s) || responseText.match(/({.*})/s);
          const jsonText = jsonMatch ? jsonMatch[1] : responseText;
          jsonResponse = JSON.parse(jsonText.trim());
          
          return res.json({
            success: true,
            explanation: jsonResponse
          });
        } catch (jsonError) {
          console.error("Error parsing JSON response:", jsonError as Error);
          return res.status(500).json({
            error: "Error parsing LLM response",
            rawResponse: responseText,
            fallbackAvailable: true
          });
        }
      } catch (llmError: unknown) {
        console.error("Error calling LLM API:", llmError);
        return res.status(500).json({
          error: "Error generating explanation with LLM",
          details: llmError instanceof Error ? llmError.message : String(llmError),
          fallbackAvailable: true
        });
      }
    } catch (error: unknown) {
      console.error("Server error:", error);
      return res.status(500).json({
        error: "Server error",
        details: error instanceof Error ? error.message : String(error),
        fallbackAvailable: true
      });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
