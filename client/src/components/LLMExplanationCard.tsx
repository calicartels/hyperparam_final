import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { HyperparameterExplanation, LLMStatusResponse, checkLLMStatus } from '@/lib/llmService';
import { Button } from '@/components/ui/button';
import { BadgeCustom } from '@/components/ui/badge-custom';
import { Skeleton } from '@/components/ui/skeleton';

type LLMExplanationCardProps = {
  explanation: HyperparameterExplanation;
  isLoading?: boolean;
};

export function LLMExplanationCard({ explanation, isLoading = false }: LLMExplanationCardProps) {
  const [llmStatus, setLlmStatus] = useState<LLMStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);

  useEffect(() => {
    async function getLLMStatus() {
      try {
        const status = await checkLLMStatus();
        setLlmStatus(status);
      } catch (error) {
        console.error("Failed to get LLM status:", error);
      } finally {
        setStatusLoading(false);
      }
    }

    getLLMStatus();
  }, []);

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
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl font-bold">
            {explanation.name}
          </CardTitle>
          {llmStatus?.available && (
            <div className="flex items-center bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 text-xs px-2.5 py-1 rounded-full">
              <span className="mr-1">AI Powered</span>
              <span className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></span>
            </div>
          )}
        </div>
        <CardDescription>
          <div className="flex items-center gap-2">
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
          </div>
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
          {llmStatus?.available ? `Powered by ${llmStatus.provider} (${llmStatus.model})` : "Using local knowledge base"}
        </div>
        {llmStatus?.requiresAuth && !llmStatus.available && (
          <Button variant="outline" size="sm">
            Configure API Key
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}