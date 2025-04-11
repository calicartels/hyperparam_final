import React from "react";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";

// Import the analyzer component (assuming it exports HyperparameterAnalyzer)
import HyperparameterAnalyzer from "@/components/HyperparameterAnalyzer";
// import DynamicHyperparameterAnalyzer from "@/components/HyperparameterAnalyzer"; // Correcting the import name

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* Render the analyzer component directly */}
      <HyperparameterAnalyzer />
      
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;