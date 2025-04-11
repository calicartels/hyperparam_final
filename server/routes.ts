import type { Express, Request, Response } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";
// VertexAI import is now handled within hyperparameterService
// import { VertexAI } from "@google-cloud/vertexai";
import hyperparameterRouter from './hyperparameterService';

// Initialization logic is now handled within hyperparameterService
// let vertexai: VertexAI | null = null;
// function initializeVertexAI() { ... }

export async function registerRoutes(app: Express): Promise<Server> {
  // Vertex AI initialization is handled by its own service
  // const vertexInitialized = initializeVertexAI();
  // console.log(`Vertex AI initialization status at startup: ${vertexInitialized ? 'SUCCESS' : 'FAILED'}`);

  // Use the hyperparameter service router, which includes LLM routes
  app.use(hyperparameterRouter);

  // API route to get all hyperparameters (presumably from storage, not LLM)
  app.get("/api/hyperparameters", async (req, res) => {
    try {
      const hyperparameters = await storage.getAllHyperparameters();
      res.json(hyperparameters);
    } catch (error) {
      console.error("Error fetching hyperparameters:", error);
      res.status(500).json({ message: "Failed to fetch hyperparameters" });
    }
  });

  // API route to get a specific hyperparameter by key (from storage)
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

  // API route to get hyperparameters by framework (from storage)
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

  // API route to get all supported frameworks (from storage)
  app.get("/api/frameworks", async (req, res) => {
    try {
      const frameworks = await storage.getAllFrameworks();
      res.json(frameworks);
    } catch (error) {
      console.error("Error fetching frameworks:", error);
      res.status(500).json({ message: "Failed to fetch frameworks" });
    }
  });

  // REMOVED: Duplicate LLM status route (/api/llm/status)
  // app.get("/api/llm/status", ...) {
  // });

  // REMOVED: Duplicate LLM explain route (/api/llm/explain-hyperparameter)
  // app.post("/api/llm/explain-hyperparameter", ...) {
  // });

  // REMOVED: Duplicate LLM detect route (/api/llm/detect-hyperparameters)
  // app.post("/api/llm/detect-hyperparameters", ...) {
  // });

  // API route to get a code explanation
  app.post("/api/explain-code", async (req: Request, res: Response) => {
    // ... existing code ...
  });

  const httpServer = createServer(app);
  return httpServer;
}
