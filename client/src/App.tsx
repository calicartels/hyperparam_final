import React from "react";
// Remove wouter imports
// import { Switch, Route, Link, useLocation, useRoute } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
// Remove page imports
// import NotFound from "@/pages/not-found";
// import PopupPage from "@/pages/PopupPage";
// import OptionsPage from "@/pages/OptionsPage";
// import TestAPIPage from "@/pages/TestAPIPage";

// Import the main analyzer component
import HyperparameterAnalyzer from "@/components/HyperparameterAnalyzer"; 
// Import Playground if you want a separate route for it, or integrate it differently
// import { HyperparameterPlayground } from "@/components/HyperparameterPlayground";

// Removed ParameterDetails component

// Removed Router component

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* Remove the <nav> element entirely */}
      {/* 
      <nav className="bg-gray-100 dark:bg-gray-800 p-4">
        <div className="container flex gap-4">
          <Link href="/">
            <span className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">Popup</span>
          </Link>
          <Link href="/options">
            <span className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">Options</span>
          </Link>
          <Link href="/test-api">
            <span className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">Test API</span>
          </Link>
          <Link href="/playground">
            <span className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer">Playground</span>
          </Link>
        </div>
      </nav>
      */}

      {/* Render the main analyzer component directly */}
      <HyperparameterAnalyzer />
      {/* You might want a simple router later if needed, but for now, just the analyzer */}
      {/* 
      <Switch>
        <Route path="/">
          <HyperparameterAnalyzer />
        </Route>
        Add other routes here if necessary
        <Route path="/playground">
           <div className="container mx-auto p-4 max-w-6xl"><HyperparameterPlayground /></div>
        </Route>
        <Route>
          <NotFound />
        </Route>
      </Switch>
      */}
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
