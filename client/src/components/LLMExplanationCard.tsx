import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { HyperparameterExplanation, LLMStatusResponse, checkLLMStatus } from '@/lib/llmService';
import { Button } from '@/components/ui/button';
import { BadgeCustom } from '@/components/ui/badge-custom';
import { Skeleton } from '@/components/ui/skeleton';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { useToast } from '@/hooks/use-toast';

type LLMExplanationCardProps = {
  explanation: HyperparameterExplanation;
  isLoading?: boolean;
};

export function LLMExplanationCard({ explanation, isLoading = false }: LLMExplanationCardProps) {
  const [llmStatus, setLlmStatus] = useState<LLMStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    async function getLLMStatus() {
      try {
        const status = await checkLLMStatus();
        setLlmStatus(status);
        
        if (!status.available && status.requiresAuth) {
          console.log("LLM service not available but requires auth");
        }
      } catch (error) {
        console.error("Failed to get LLM status:", error);
        toast({
          title: "Warning",
          description: "Could not connect to AI service. Using fallback explanations.",
          variant: "destructive"
        });
      } finally {
        setStatusLoading(false);
      }
    }

    getLLMStatus();
  }, [toast]);

  if (isLoading || statusLoading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <Skeleton className="h-8 w-2/3 mb-2" />
          <Skeleton className="h-4 w-1/2" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full mb-4" />
          <Skeleton className="h-4 w-full mb-2" />
          <Skeleton className="h-4 w-full mb-2" />
          <Skeleton className="h-4 w-3/4" />
        </CardContent>
      </Card>
    );
  }

  return (
    <>
      <Card className="w-full">
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="text-xl font-bold">
              {explanation.name}
            </CardTitle>
            {llmStatus?.available ? (
              <div className="flex items-center bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 text-xs px-2.5 py-1 rounded-full">
                <span className="mr-1">AI Powered</span>
                <span className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></span>
              </div>
            ) : (
              <div className="flex items-center bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 text-xs px-2.5 py-1 rounded-full">
                <span className="mr-1">Fallback Mode</span>
                <span className="w-2 h-2 bg-gray-500 rounded-full"></span>
              </div>
            )}
          </div>
          <CardDescription className="flex flex-wrap items-center gap-2">
            <span className="text-sm font-medium">Impact: </span>
            <span className={`text-sm px-2 py-0.5 rounded-full ${
              explanation.impact === 'high' 
                ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' 
                : explanation.impact === 'medium'
                  ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                  : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
            }`}>
              {explanation.impact.charAt(0).toUpperCase() + explanation.impact.slice(1)}
            </span>
            
            {!llmStatus?.available && (
              <div className="w-full mt-2">
                <p className="text-xs text-amber-600 dark:text-amber-400">
                  Using local knowledge base. For personalized analysis, configure API key.
                </p>
              </div>
            )}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <h3 className="text-sm font-medium mb-1">Definition</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {explanation.description}
            </p>
          </div>
          
          <div>
            <h3 className="text-sm font-medium mb-1">Current Value Analysis</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {explanation.valueAnalysis}
            </p>
          </div>
          
          <div>
            <h3 className="text-sm font-medium mb-1">Alternative Values</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {explanation.alternatives.map((alt, index) => (
                <div key={index} className="flex flex-col p-2 border rounded-lg">
                  <div className="flex justify-between items-center mb-1">
                    <code className="text-sm font-mono bg-gray-100 dark:bg-gray-800 px-1 rounded">
                      {alt.value}
                    </code>
                    <BadgeCustom type={alt.type} className="text-xs">
                      {alt.type}
                    </BadgeCustom>
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {alt.description}
                  </p>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="text-sm font-medium mb-1">Best Practices</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {explanation.bestPractices}
            </p>
          </div>
          
          <div>
            <h3 className="text-sm font-medium mb-1">Trade-offs</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {explanation.tradeoffs}
            </p>
          </div>
        </CardContent>
        <CardFooter className="flex justify-between">
          <div className="text-xs text-gray-500">
            {llmStatus?.available ? 
              `Powered by ${llmStatus.provider} (${llmStatus.model})` : 
              "Using local knowledge base (fallback mode)"
            }
          </div>
          {llmStatus?.requiresAuth && !llmStatus.available && (
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setIsConfigDialogOpen(true)}
            >
              Configure API Key
            </Button>
          )}
        </CardFooter>
      </Card>

      {/* API Configuration Dialog */}
      <Dialog open={isConfigDialogOpen} onOpenChange={setIsConfigDialogOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Enable AI-Powered Analysis</DialogTitle>
            <DialogDescription>
              HyperExplainer can provide personalized ML explanations using Google Vertex AI.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-6 py-4">
            <div className="rounded-lg border bg-card p-3">
              <div className="flex items-center space-x-3">
                <div className="h-10 w-10 bg-purple-100 dark:bg-purple-900 rounded-full flex items-center justify-center">
                  <span className="text-purple-600 dark:text-purple-300 text-xl">🤖</span>
                </div>
                <div>
                  <h3 className="text-sm font-medium">Enhanced Experience</h3>
                  <p className="text-xs text-gray-500">
                    Get personalized analysis specific to your code context
                  </p>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-sm font-medium border-b pb-2">Setup Steps</h3>
              
              <div className="space-y-3">
                <div className="flex items-start">
                  <div className="h-6 w-6 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center text-xs font-medium text-blue-700 dark:text-blue-300 mr-2 mt-0.5">1</div>
                  <div>
                    <h4 className="text-sm font-medium">Create a Google Cloud Project</h4>
                    <p className="text-xs text-gray-500">
                      Visit <a href="https://console.cloud.google.com/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Google Cloud Console</a> to create or select a project
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <div className="h-6 w-6 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center text-xs font-medium text-blue-700 dark:text-blue-300 mr-2 mt-0.5">2</div>
                  <div>
                    <h4 className="text-sm font-medium">Enable Vertex AI API</h4>
                    <p className="text-xs text-gray-500">
                      Go to "APIs & Services" → "Enable APIs and Services" → search for "Vertex AI"
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <div className="h-6 w-6 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center text-xs font-medium text-blue-700 dark:text-blue-300 mr-2 mt-0.5">3</div>
                  <div>
                    <h4 className="text-sm font-medium">Create Service Account & Key</h4>
                    <p className="text-xs text-gray-500">
                      "IAM & Admin" → "Service Accounts" → create account with "Vertex AI User" role → "Keys" tab → "Add Key" → "Create new key" (JSON)
                    </p>
                  </div>
                </div>
                
                <div className="flex items-start">
                  <div className="h-6 w-6 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center text-xs font-medium text-blue-700 dark:text-blue-300 mr-2 mt-0.5">4</div>
                  <div>
                    <h4 className="text-sm font-medium">Configure HyperExplainer</h4>
                    <p className="text-xs text-gray-500">
                      In the settings page, upload your JSON key file and enter your Google Cloud project ID
                    </p>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="rounded-lg border bg-amber-50 dark:bg-amber-900/20 p-3">
              <p className="text-xs text-amber-700 dark:text-amber-400">
                <strong>Note:</strong> You can continue using HyperExplainer with the built-in knowledge base. Setting up Google Cloud provides personalized analysis but is optional.
              </p>
            </div>
          </div>
          
          <DialogFooter className="sm:justify-between">
            <Button
              variant="ghost"
              onClick={() => setIsConfigDialogOpen(false)}
            >
              Maybe Later
            </Button>
            <Button
              variant="default"
              onClick={() => {
                // Open the extension options page
                if (window.chrome && window.chrome.runtime) {
                  window.chrome.runtime.openOptionsPage();
                } else {
                  // Fallback for non-extension environment (development)
                  window.open('/options', '_blank');
                }
                setIsConfigDialogOpen(false);
              }}
            >
              Open Settings
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}