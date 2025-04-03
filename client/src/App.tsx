import React, { useEffect } from "react";
import { Switch, Route, Link, useLocation, useRoute } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import PopupPage from "@/pages/PopupPage";
import OptionsPage from "@/pages/OptionsPage";
import TestAPIPage from "@/pages/TestAPIPage";
import { HyperparameterPlayground } from "@/components/HyperparameterPlayground";

// Parameter Details wrapper component
function ParameterDetails() {
  const [locationPath] = useLocation();
  const searchParams = new URLSearchParams(window.location.search);
  
  // Get parameters from URL
  const paramName = searchParams.get('param') || '';
  const paramValue = searchParams.get('value') || '';
  const framework = searchParams.get('framework') || 'unknown';
  
  return (
    <TestAPIPage initialParam={paramName} initialValue={paramValue} initialFramework={framework} />
  );
}

function Router() {
  const [locationPath] = useLocation();
  const locationSearch = window.location.search;
  const params = new URLSearchParams(locationSearch);
  const page = params.get('page');
  
  // Handle query parameters to direct to specific pages
  useEffect(() => {
    if (page) {
      let path = '/';
      
      switch (page) {
        case 'options':
          path = '/options';
          break;
        case 'test':
          path = '/parameter-details';
          break;
        default:
          path = '/';
      }
      
      // Keep the query parameters
      window.history.replaceState({}, '', `${path}${locationSearch}`);
    }
  }, [page, locationSearch]);
  
  // Check if we're on the parameter details page
  const [isOnParameterDetails] = useRoute('/parameter-details');
  
  return (
    <>
      {!isOnParameterDetails && (
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
      )}
      
      <Switch>
        <Route path="/">
          {() => <PopupPage />}
        </Route>
        <Route path="/options">
          {() => <OptionsPage />}
        </Route>
        <Route path="/test-api">
          {() => <TestAPIPage />}
        </Route>
        <Route path="/playground">
          {() => <div className="container mx-auto p-4 max-w-6xl"><HyperparameterPlayground /></div>}
        </Route>
        <Route path="/parameter-details">
          {() => {
            const searchParams = new URLSearchParams(window.location.search);
            const paramName = searchParams.get('param') || '';
            const paramValue = searchParams.get('value') || '';
            const framework = searchParams.get('framework') || 'unknown';
            return <TestAPIPage initialParam={paramName} initialValue={paramValue} initialFramework={framework} />;
          }}
        </Route>
        <Route>
          {() => <NotFound />}
        </Route>
      </Switch>
    </>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router />
      <Toaster />
    </QueryClientProvider>
  );
}

export default App;
