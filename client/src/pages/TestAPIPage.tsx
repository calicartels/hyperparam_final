import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { LLMExplanationCard } from '@/components/LLMExplanationCard';
import { checkLLMStatus, getHyperparameterExplanation, generateFallbackExplanation, LLMStatusResponse } from '@/lib/llmService';
import { Switch } from '@/components/ui/switch';

export default function TestAPIPage() {
  const [llmStatus, setLlmStatus] = useState<LLMStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const [paramName, setParamName] = useState('learning_rate');
  const [paramValue, setParamValue] = useState('0.001');
  const [framework, setFramework] = useState('tensorflow');
  const [codeContext, setCodeContext] = useState(`model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])`);

  const [useFallback, setUseFallback] = useState(false);
  const [explanation, setExplanation] = useState(generateFallbackExplanation(paramName, paramValue, framework));

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
        
        <div>
          <LLMExplanationCard explanation={explanation} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
}