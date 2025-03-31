import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import NotFound from "@/pages/not-found";
import PopupPage from "@/pages/PopupPage";
import OptionsPage from "@/pages/OptionsPage";

function Router() {
  return (
    <Switch>
      <Route path="/" component={PopupPage} />
      <Route path="/options" component={OptionsPage} />
      <Route component={NotFound} />
    </Switch>
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
