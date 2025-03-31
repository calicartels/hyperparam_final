import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { Eye, Settings } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const PopupPage = () => {
  const { toast } = useToast();
  const [isAnalysisActive, setIsAnalysisActive] = useState(false);
  
  const { data: hyperparameters, isLoading } = useQuery({
    queryKey: ['/api/hyperparameters'],
    enabled: false, // Only load when needed
  });

  const toggleAnalysis = () => {
    const newState = !isAnalysisActive;
    setIsAnalysisActive(newState);
    
    // Send message to content script to toggle analysis
    chrome.tabs?.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, { 
          action: newState ? 'activateAnalysis' : 'deactivateAnalysis'
        });
        
        toast({
          title: newState ? "Analysis Activated" : "Analysis Deactivated",
          description: newState 
            ? "HyperExplainer will now analyze code on this page." 
            : "HyperExplainer is now inactive.",
          duration: 3000,
        });
      }
    });
  };

  const openOptions = () => {
    chrome.runtime?.openOptionsPage?.();
  };
  
  return (
    <div className="w-80 p-4">
      <header className="mb-4">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <Eye className="h-5 w-5 text-primary" />
          <span className="bg-gradient-to-r from-primary-600 to-primary-400 bg-clip-text text-transparent">HyperExplainer</span>
        </h1>
        <p className="text-sm text-muted-foreground">
          Explains hyperparameters in LLM-generated code
        </p>
      </header>
      
      <Card className="mb-4">
        <CardContent className="p-4">
          <div className="flex flex-col items-center">
            <Button 
              className={`w-full ${isAnalysisActive ? 'bg-primary-700' : ''}`}
              size="lg"
              onClick={toggleAnalysis}
            >
              {isAnalysisActive ? 'Deactivate Analysis' : 'Activate Analysis'}
            </Button>
            
            <p className="text-xs text-muted-foreground mt-2 text-center">
              {isAnalysisActive 
                ? "HyperExplainer is active and will analyze code blocks on this page" 
                : "Click to enable hyperparameter analysis for code on this page"}
            </p>
          </div>
        </CardContent>
      </Card>
      
      <Card className="mb-4">
        <CardContent className="p-4">
          <h2 className="text-sm font-medium mb-2">Supported Frameworks</h2>
          <div className="flex flex-wrap gap-1">
            <span className="text-xs bg-primary-100 text-primary-800 px-2 py-1 rounded-md">PyTorch</span>
            <span className="text-xs bg-primary-100 text-primary-800 px-2 py-1 rounded-md">TensorFlow</span>
            <span className="text-xs bg-primary-100 text-primary-800 px-2 py-1 rounded-md">Keras</span>
            <span className="text-xs bg-primary-100 text-primary-800 px-2 py-1 rounded-md">scikit-learn</span>
          </div>
        </CardContent>
      </Card>
      
      <div className="flex justify-between">
        <Button 
          variant="outline" 
          className="text-xs" 
          size="sm"
          onClick={openOptions}
        >
          <Settings className="h-3 w-3 mr-1" />
          Options
        </Button>
        <a 
          href="https://github.com/your-repo/hyperexplainer" 
          target="_blank" 
          rel="noreferrer"
          className="text-xs text-primary-600 hover:underline"
        >
          v1.0.0
        </a>
      </div>
    </div>
  );
};

export default PopupPage;
