import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { LLMExplanationCard } from '@/components/LLMExplanationCard';
import { HyperparameterVisualizations } from '@/components/SimpleHyperparameterVisualizations';
import { NetworkVisualization } from '@/components/NetworkVisualization';
import { BenchmarkComparison } from '@/components/BenchmarkComparison';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { checkLLMStatus, getHyperparameterExplanation, generateFallbackExplanation, LLMStatusResponse } from '@/lib/llmService';
import { Switch } from '@/components/ui/switch';
import { Tutorial } from '@/components/Tutorial';
import { useTutorial } from '@/hooks/use-tutorial';

interface TestAPIPageProps {
  initialParam?: string;
  initialValue?: string;
  initialFramework?: string;
}

export default function TestAPIPage({ 
  initialParam, 
  initialValue, 
  initialFramework 
}: TestAPIPageProps) {
  const [llmStatus, setLlmStatus] = useState<LLMStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const [paramName, setParamName] = useState(initialParam || 'learning_rate');
  const [paramValue, setParamValue] = useState(initialValue || '0.001');
  const [framework, setFramework] = useState(initialFramework || 'tensorflow');
  const [codeContext, setCodeContext] = useState(`model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])`);

  const [useFallback, setUseFallback] = useState(false);
  const [explanation, setExplanation] = useState(generateFallbackExplanation(paramName, paramValue, framework));
  
  // Tutorial state management
  const { showTutorial, completeTutorial, dismissTutorial } = useTutorial();
  const [isTutorialVisible, setIsTutorialVisible] = useState(false);
  
  const handleShowTutorial = () => {
    setIsTutorialVisible(true);
    showTutorial();
  };
  
  const handleCompleteTutorial = () => {
    setIsTutorialVisible(false);
    completeTutorial();
  };
  
  const handleDismissTutorial = () => {
    setIsTutorialVisible(false);
    dismissTutorial();
  };

  // Check LLM status on component mount
  React.useEffect(() => {
    async function getLLMStatus() {
      try {
        const status = await checkLLMStatus();
        setLlmStatus(status);
      } catch (error) {
        console.error("Failed to check LLM status:", error);
      }
    }
    
    getLLMStatus();
  }, []);

  // Get explanation from LLM or fallback
  const getExplanation = async () => {
    setIsLoading(true);
    
    try {
      if (useFallback) {
        // Generate fallback explanation
        const fallbackExplanation = generateFallbackExplanation(paramName, paramValue, framework);
        setExplanation(fallbackExplanation);
      } else {
        // Get explanation from LLM service
        const response = await getHyperparameterExplanation({
          paramName,
          paramValue,
          framework,
          codeContext
        });
        
        if (response.success) {
          setExplanation(response.explanation);
        } else {
          // If LLM fails, use fallback
          const fallbackExplanation = generateFallbackExplanation(paramName, paramValue, framework);
          setExplanation(fallbackExplanation);
        }
      }
    } catch (error) {
      console.error("Error getting explanation:", error);
      // On error, use fallback
      const fallbackExplanation = generateFallbackExplanation(paramName, paramValue, framework);
      setExplanation(fallbackExplanation);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container py-10">
      <h1 className="text-3xl font-bold mb-6">Test HyperExplainer API</h1>
      
      <div className="flex justify-end mb-4">
        <div className="relative">
          <Button 
            variant="outline" 
            onClick={handleShowTutorial}
            className="gap-2 group transition-all hover:bg-blue-50 hover:border-blue-200"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-blue-500 group-hover:text-blue-600">
              <path d="M12 16v-4M12 8h.01M22 12c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2s10 4.477 10 10z" 
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span className="text-blue-600 group-hover:text-blue-700">Show Tutorial</span>
          </Button>
          <span className="absolute -top-2 -right-2 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white transform translate-x-1/2 -translate-y-1/2 bg-red-500 rounded-full animate-pulse">
            NEW
          </span>
        </div>
      </div>
      
      <div className="grid md:grid-cols-2 gap-8">
        <div>
          <Card>
            <CardHeader>
              <CardTitle>Hyperparameter Input</CardTitle>
              <CardDescription>
                Enter hyperparameter details below to get an explanation
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="paramName">Parameter Name</Label>
                <Input 
                  id="paramName"
                  value={paramName}
                  onChange={(e) => setParamName(e.target.value)}
                  placeholder="e.g., learning_rate, batch_size"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="paramValue">Parameter Value</Label>
                <Input 
                  id="paramValue"
                  value={paramValue}
                  onChange={(e) => setParamValue(e.target.value)}
                  placeholder="e.g., 0.001, 32"
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="framework">Framework</Label>
                <Select value={framework} onValueChange={setFramework}>
                  <SelectTrigger id="framework">
                    <SelectValue placeholder="Select framework" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tensorflow">TensorFlow</SelectItem>
                    <SelectItem value="pytorch">PyTorch</SelectItem>
                    <SelectItem value="keras">Keras</SelectItem>
                    <SelectItem value="scikit-learn">Scikit-learn</SelectItem>
                    <SelectItem value="xgboost">XGBoost</SelectItem>
                    <SelectItem value="fastai">FastAI</SelectItem>
                    <SelectItem value="huggingface">Hugging Face</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="codeContext">Code Context</Label>
                <Textarea 
                  id="codeContext"
                  value={codeContext}
                  onChange={(e) => setCodeContext(e.target.value)}
                  placeholder="Paste code snippet here..."
                  rows={6}
                  className="font-mono text-sm"
                />
              </div>
              
              <div className="flex items-center space-x-2 pt-2">
                <Switch 
                  id="use-fallback"
                  checked={useFallback}
                  onCheckedChange={setUseFallback}
                />
                <Label htmlFor="use-fallback">Use fallback (no LLM)</Label>
              </div>
              
              <Button 
                onClick={getExplanation}
                disabled={isLoading}
                className="w-full"
              >
                {isLoading ? "Loading..." : "Get Explanation"}
              </Button>
              
              {llmStatus && (
                <div className="mt-2 text-sm text-gray-500">
                  LLM Status: {llmStatus.available ? 
                    <span className="text-green-600">Available ({llmStatus.provider})</span> : 
                    <span className="text-red-600">Not Available</span>
                  }
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        
        <div className="space-y-6">
          <LLMExplanationCard explanation={explanation} isLoading={isLoading} />
          
          <Tabs defaultValue="visualization" className="w-full">
            <TabsList className="mb-4">
              <TabsTrigger value="visualization">Parameter Visualization</TabsTrigger>
              <TabsTrigger value="network">Neural Network</TabsTrigger>
              <TabsTrigger value="benchmark">Benchmark</TabsTrigger>
            </TabsList>
            
            <TabsContent value="visualization" className="space-y-4">
              <HyperparameterVisualizations 
                paramName={paramName} 
                paramValue={paramValue} 
                framework={framework} 
              />
            </TabsContent>
            
            <TabsContent value="network" className="space-y-4">
              <NetworkVisualization 
                paramName={paramName} 
                paramValue={paramValue} 
                framework={framework} 
              />
            </TabsContent>
            
            <TabsContent value="benchmark" className="space-y-4">
              <BenchmarkComparison 
                paramName={paramName} 
                paramValue={paramValue} 
                framework={framework} 
              />
            </TabsContent>
          </Tabs>
        </div>
      </div>
      
      {/* Tutorial Component */}
      <Tutorial 
        onComplete={handleCompleteTutorial} 
        onDismiss={handleDismissTutorial} 
        isVisible={isTutorialVisible} 
      />
    </div>
  );
}