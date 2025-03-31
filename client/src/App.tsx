import { Switch, Route, Link } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import PopupPage from "@/pages/PopupPage";
import OptionsPage from "@/pages/OptionsPage";
import TestAPIPage from "@/pages/TestAPIPage";

function Router() {
  return (
    <>
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
        </div>
      </nav>
      
      <Switch>
        <Route path="/" component={PopupPage} />
        <Route path="/options" component={OptionsPage} />
        <Route path="/test-api" component={TestAPIPage} />
        <Route component={NotFound} />
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
