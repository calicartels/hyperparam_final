import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowRight, Info, AlertCircle, HelpCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Tutorial } from '@/components/Tutorial';
import { useTutorial } from '@/hooks/use-tutorial';

// Helper to safely check if we're in a Chrome extension environment
const isChromeExtension = typeof window !== 'undefined' && 
  typeof window.chrome !== 'undefined' && 
  typeof window.chrome.runtime !== 'undefined';

interface Hyperparameter {
  name: string;
  value: string;
  framework: string;
  impact: 'high' | 'medium' | 'low';
  codeContext?: string;
}

export default function PopupPage() {
  const [hyperparameters, setHyperparameters] = useState<Hyperparameter[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentFrameworks, setCurrentFrameworks] = useState<string[]>([]);
  const { isTutorialVisible, completeTutorial, dismissTutorial, resetTutorial, showTutorial } = useTutorial();
  
  // Get hyperparameters from background script on mount
  useEffect(() => {
    if (isChromeExtension) {
      // Try to get data from the active tab via content script
      try {
        window.chrome.tabs.query({ active: true, currentWindow: true }, (tabs: any) => {
          if (tabs[0].id) {
            window.chrome.tabs.sendMessage(
              tabs[0].id, 
              { action: 'GET_DETECTED_PARAMETERS' },
              (response: any) => {
                if (response && response.parameters) {
                  setHyperparameters(response.parameters);
                  if (response.frameworks) {
                    setCurrentFrameworks(response.frameworks);
                  }
                }
                setLoading(false);
              }
            );
          } else {
            setLoading(false);
          }
        });
      } catch (error) {
        console.error('Chrome extension API error:', error);
        setLoading(false);
      }
    } else {
      // We're in web app mode - use example data for demonstration
      setTimeout(() => {
        setLoading(false);
        // Example parameters for demo purposes
        setHyperparameters([
          { name: 'learning_rate', value: '0.001', framework: 'tensorflow', impact: 'high' },
          { name: 'batch_size', value: '32', framework: 'common', impact: 'medium' },
          { name: 'dropout', value: '0.5', framework: 'common', impact: 'medium' }
        ]);
        setCurrentFrameworks(['tensorflow', 'pytorch']);
      }, 500);
    }
  }, []);
  
  const openOptions = () => {
    if (isChromeExtension) {
      try {
        window.chrome.runtime.openOptionsPage();
      } catch (error) {
        console.error('Failed to open options page:', error);
      }
    } else {
      // In web app mode, redirect to options route
      window.location.href = '/?page=options';
    }
  };
  
  const analyzePage = () => {
    setLoading(true);
    
    if (isChromeExtension) {
      try {
        window.chrome.tabs.query({ active: true, currentWindow: true }, (tabs: any) => {
          if (tabs[0].id) {
            window.chrome.tabs.sendMessage(tabs[0].id, { action: 'ANALYZE_PAGE' }, (response: any) => {
              if (response && response.parameters) {
                setHyperparameters(response.parameters);
                if (response.frameworks) {
                  setCurrentFrameworks(response.frameworks);
                }
              }
              setLoading(false);
            });
          } else {
            setLoading(false);
          }
        });
      } catch (error) {
        console.error('Chrome extension API error:', error);
        setLoading(false);
      }
    } else {
      // In web app mode, simulate analyzing a page
      setTimeout(() => {
        setHyperparameters([
          { name: 'learning_rate', value: '0.001', framework: 'tensorflow', impact: 'high' },
          { name: 'batch_size', value: '32', framework: 'common', impact: 'medium' },
          { name: 'dropout', value: '0.5', framework: 'common', impact: 'medium' },
          { name: 'epochs', value: '100', framework: 'pytorch', impact: 'medium' },
          { name: 'optimizer', value: 'adam', framework: 'tensorflow', impact: 'high' }
        ]);
        setCurrentFrameworks(['tensorflow', 'pytorch']);
        setLoading(false);
      }, 800);
    }
  };
  
  const openDetailedAnalysis = (param: Hyperparameter) => {
    if (isChromeExtension) {
      try {
        const url = window.chrome.runtime.getURL(`index.html?param=${param.name}&value=${param.value}&framework=${param.framework}`);
        window.chrome.tabs.create({ url });
      } catch (error) {
        console.error('Failed to open detailed analysis:', error);
        // Fallback for web app
        window.open(`/?param=${param.name}&value=${param.value}&framework=${param.framework}`, '_blank');
      }
    } else {
      // In web app mode, open in a new tab
      window.open(`/?param=${param.name}&value=${param.value}&framework=${param.framework}`, '_blank');
    }
  };
  
  // Group parameters by framework
  const parametersByFramework: Record<string, Hyperparameter[]> = {};
  
  hyperparameters.forEach(param => {
    if (!parametersByFramework[param.framework]) {
      parametersByFramework[param.framework] = [];
    }
    parametersByFramework[param.framework].push(param);
  });
  
  // Get impact color
  const getImpactColor = (impact: string) => {
    switch(impact) {
      case 'high':
        return 'text-red-500';
      case 'medium':
        return 'text-amber-500';
      case 'low':
        return 'text-green-500';
      default:
        return 'text-gray-500';
    }
  };
  
  return (
    <div className="p-4 w-full max-w-sm">
      {/* Tutorial Overlay */}
      {isTutorialVisible && (
        <Tutorial
          onComplete={completeTutorial}
          onDismiss={dismissTutorial}
          isVisible={isTutorialVisible}
        />
      )}
      
      <div className="flex justify-between items-center">
        <h1 className="text-xl font-bold mb-1 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
          HyperExplainer
        </h1>
        <Button 
          variant="ghost" 
          size="sm" 
          className="p-1 h-8 w-8" 
          onClick={showTutorial}
          title="Show tutorial"
        >
          <HelpCircle size={16} className="text-gray-400 hover:text-indigo-500" />
        </Button>
      </div>
      <p className="text-xs text-gray-500 mb-4">AI-powered hyperparameter analysis</p>
      
      {loading ? (
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
        </div>
      ) : hyperparameters.length > 0 ? (
        <div className="space-y-4">
          <div className="flex items-center text-sm text-gray-600">
            <Info size={14} className="mr-1 text-indigo-500" />
            <span>Found {hyperparameters.length} hyperparameters</span>
            
            {currentFrameworks.length > 0 && (
              <div className="ml-auto flex gap-1">
                {currentFrameworks.map(framework => (
                  <Badge key={framework} variant="outline" className="text-xs capitalize">
                    {framework}
                  </Badge>
                ))}
              </div>
            )}
          </div>
          
          <div className="max-h-[300px] overflow-y-auto pr-1 space-y-3">
            {Object.entries(parametersByFramework).map(([framework, params]) => (
              <div key={framework} className="space-y-2">
                {Object.keys(parametersByFramework).length > 1 && (
                  <h3 className="text-xs uppercase tracking-wider text-gray-500 font-semibold">
                    {framework === 'common' ? 'Common Parameters' : framework}
                  </h3>
                )}
                
                {params.map((param, index) => (
                  <Card 
                    key={index} 
                    className="cursor-pointer hover:shadow-md transition-shadow border-l-4"
                    style={{ borderLeftColor: param.impact === 'high' ? '#ef4444' : param.impact === 'medium' ? '#f59e0b' : '#10b981' }}
                    onClick={() => openDetailedAnalysis(param)}
                  >
                    <CardHeader className="p-3 pb-1">
                      <CardTitle className="text-sm flex justify-between items-center">
                        <span className="font-medium">
                          {param.name}
                        </span>
                        <ArrowRight size={16} className="text-gray-400" />
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-3 pt-0 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Value:</span>
                        <span className="font-mono text-indigo-600">{param.value}</span>
                      </div>
                      <div className="flex justify-between items-center mt-1">
                        <span className="text-gray-500">Impact:</span>
                        <span className={`${getImpactColor(param.impact)} font-medium capitalize`}>
                          {param.impact}
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ))}
          </div>
          
          <Button className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700" onClick={openOptions}>
            See detailed analysis
          </Button>
        </div>
      ) : (
        <div className="space-y-4 py-4">
          <div className="flex flex-col items-center justify-center text-center space-y-3">
            <AlertCircle size={40} className="text-gray-300" />
            <p className="text-gray-600">
              No hyperparameters detected on this page.
            </p>
          </div>
          
          <Button className="w-full" onClick={analyzePage}>
            Analyze this page
          </Button>
          
          <Button className="w-full" variant="outline" onClick={openOptions}>
            Open settings
          </Button>
        </div>
      )}
      
      <div className="mt-4 pt-2 border-t border-gray-100">
        <p className="text-[10px] text-center text-gray-400">
          Powered by Google Cloud Vertex AI
        </p>
      </div>
    </div>
  );
}