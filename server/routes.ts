import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { z } from "zod";

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

  const httpServer = createServer(app);
  return httpServer;
}
