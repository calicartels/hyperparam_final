import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { HyperparameterPlayground } from './HyperparameterPlayground';
import { NetworkVisualization } from './NetworkVisualization';
import { HyperparameterVisualizations } from './HyperparameterVisualizations';
import { BenchmarkComparison } from './BenchmarkComparison';
import { LLMExplanationCard } from './LLMExplanationCard';
import { identifyHyperparameters, detectFramework } from '../lib/hyperparameters';
import { getHyperparameterExplanation, generateFallbackExplanation, checkLLMStatus } from '../lib/llmService';
import { Code, BadgePlus, ChevronDown, ChevronUp, ArrowRight, Eye, Info } from 'lucide-react';

const HyperparameterAnalyzer = () => {
  // State for code input
  const [codeInput, setCodeInput] = useState(
`import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer, 
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10)`);

  // State for detected parameters
  const [detectedParams, setDetectedParams] = useState([]);
  const [framework, setFramework] = useState('');
  const [expandedParams, setExpandedParams] = useState({});
  const [selectedParam, setSelectedParam] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [llmAvailable, setLlmAvailable] = useState(false);

  // Detect hyperparameters on code change
  useEffect(() => {
    if (codeInput.trim()) {
      analyzeCode();
    }
  }, [codeInput]);

  // Check LLM status on component mount
  useEffect(() => {
    async function checkLLM() {
      try {
        const status = await checkLLMStatus();
        setLlmAvailable(status?.available || false);
      } catch (error) {
        console.error("Failed to check LLM status:", error);
        setLlmAvailable(false);
      }
    }
    
    checkLLM();
  }, []);

  // Function to analyze code and detect parameters
  const analyzeCode = () => {
    if (!codeInput.trim()) return;
    
    const params = identifyHyperparameters(codeInput);
    const detectedFramework = detectFramework(codeInput);
    
    setDetectedParams(params);
    setFramework(detectedFramework);
    
    // Reset expanded state for new parameters
    const initialExpandedState = {};
    params.forEach(param => {
      initialExpandedState[param.key] = false;
    });
    setExpandedParams(initialExpandedState);
    
    // Select first parameter by default if available
    if (params.length > 0 && !selectedParam) {
      handleSelectParameter(params[0]);
    }
  };

  // Toggle expanded state for parameter
  const toggleExpanded = (key) => {
    setExpandedParams(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  // Handle parameter selection
  const handleSelectParameter = async (param) => {
    setSelectedParam(param);
    setIsLoading(true);
    
    try {
      if (llmAvailable) {
        // Get explanation from LLM service
        const response = await getHyperparameterExplanation({
          paramName: param.key,
          paramValue: param.value,
          framework: framework,
          codeContext: codeInput
        });
        
        if (response.success && response.explanation) {
          setExplanation(response.explanation);
        } else {
          // If LLM fails, use fallback
          const fallbackExplanation = generateFallbackExplanation(param.key, param.value, framework);
          setExplanation(fallbackExplanation);
        }
      } else {
        // Use fallback explanation if LLM is not available
        const fallbackExplanation = generateFallbackExplanation(param.key, param.value, framework);
        setExplanation(fallbackExplanation);
      }
    } catch (error) {
      console.error("Error getting explanation:", error);
      // On error, use fallback
      const fallbackExplanation = generateFallbackExplanation(param.key, param.value, framework);
      setExplanation(fallbackExplanation);
    } finally {
      setIsLoading(false);
    }
  };

  // Get impact color
  const getImpactColor = (impact) => {
    switch(impact) {
      case 'high':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'medium':
        return 'bg-amber-100 text-amber-800 border-amber-300';
      case 'low':
        return 'bg-green-100 text-green-800 border-green-300';
      default:
        return 'bg-blue-100 text-blue-800 border-blue-300';
    }
  };

  return (
    <div className="container mx-auto py-6 px-4">
      <header className="mb-6">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Eye className="h-8 w-8 text-primary" />
          <span>HyperExplainer</span>
        </h1>
        <p className="text-muted-foreground">
          Automatically detect and understand hyperparameters in machine learning code
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Code Input Section */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="h-5 w-5" />
                Code Input
              </CardTitle>
              <CardDescription>
                Paste machine learning code to analyze
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                value={codeInput}
                onChange={(e) => setCodeInput(e.target.value)}
                className="font-mono text-sm h-96"
                placeholder="Paste your ML code here..."
              />
              <Button 
                onClick={analyzeCode} 
                className="w-full mt-4"
              >
                Analyze Code
              </Button>
              
              {/* Detected Parameters List */}
              {detectedParams.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-sm font-medium mb-2 flex items-center">
                    <BadgePlus className="h-4 w-4 mr-1" />
                    Detected Parameters
                    {framework && (
                      <Badge className="ml-2" variant="outline">
                        {framework}
                      </Badge>
                    )}
                  </h3>
                  <div className="space-y-2 max-h-96 overflow-y-auto pr-2">
                    {detectedParams.map((param, index) => (
                      <div 
                        key={index}
                        className={`border rounded-md overflow-hidden transition-all ${
                          selectedParam?.key === param.key ? 'ring-2 ring-primary' : ''
                        }`}
                      >
                        <div 
                          className="p-3 cursor-pointer flex justify-between items-center hover:bg-gray-50"
                          onClick={() => handleSelectParameter(param)}
                        >
                          <div>
                            <h4 className="font-medium text-sm">{param.key}</h4>
                            <div className="text-sm font-mono text-gray-600">{param.value}</div>
                          </div>
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleExpanded(param.key);
                            }}
                            className="p-1 rounded-full hover:bg-gray-200"
                          >
                            {expandedParams[param.key] ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </button>
                        </div>
                        
                        {expandedParams[param.key] && (
                          <div className="px-3 pb-3 text-xs text-gray-600 border-t pt-2">
                            <div className="flex items-center gap-1 mb-1">
                              <Info className="h-3 w-3" />
                              <span>Position: {param.position.start}-{param.position.end}</span>
                            </div>
                            <Button 
                              size="sm" 
                              variant="ghost" 
                              className="text-xs py-1 h-7 mt-1"
                              onClick={() => handleSelectParameter(param)}
                            >
                              View Details
                              <ArrowRight className="h-3 w-3 ml-1" />
                            </Button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
        
        {/* Hyperparameter Explanation & Visualization */}
        <div className="lg:col-span-2">
          {selectedParam ? (
            <div className="space-y-6">
              {/* LLM Explanation Card */}
              {explanation && (
                <LLMExplanationCard explanation={explanation} isLoading={isLoading} />
              )}
              
              {/* Visualization Tabs */}
              <Tabs defaultValue="playground" className="w-full">
                <TabsList className="mb-4">
                  <TabsTrigger value="playground">Parameter Playground</TabsTrigger>
                  <TabsTrigger value="visualization">Impact Visualization</TabsTrigger>
                  <TabsTrigger value="network">Neural Network</TabsTrigger>
                  <TabsTrigger value="benchmark">Benchmark</TabsTrigger>
                </TabsList>
                
                <TabsContent value="playground" className="space-y-4">
                  <HyperparameterPlayground />
                </TabsContent>
                
                <TabsContent value="visualization" className="space-y-4">
                  <HyperparameterVisualizations 
                    paramName={selectedParam.key} 
                    paramValue={selectedParam.value} 
                    framework={framework} 
                  />
                </TabsContent>
                
                <TabsContent value="network" className="space-y-4">
                  <NetworkVisualization 
                    paramName={selectedParam.key} 
                    paramValue={selectedParam.value} 
                    framework={framework} 
                  />
                </TabsContent>
                
                <TabsContent value="benchmark" className="space-y-4">
                  <BenchmarkComparison 
                    paramName={selectedParam.key} 
                    paramValue={selectedParam.value} 
                    framework={framework} 
                  />
                </TabsContent>
              </Tabs>
            </div>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-10">
                <Info className="h-12 w-12 text-gray-300 mb-4" />
                <h3 className="text-xl font-medium text-gray-600 mb-2">No Parameter Selected</h3>
                <p className="text-gray-500 text-center max-w-md mb-4">
                  Enter your machine learning code and select a parameter to see detailed analysis and visualizations.
                </p>
                <Button onClick={analyzeCode}>Analyze Code</Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export default HyperparameterAnalyzer;