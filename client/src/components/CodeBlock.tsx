import { useState, useEffect, useRef } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { identifyHyperparameters, detectFramework } from '@/lib/hyperparameters';
import { useToast } from '@/hooks/use-toast';

interface CodeBlockProps {
  code: string;
  onParameterClick: (paramKey: string, value: string) => void;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, onParameterClick }) => {
  const codeRef = useRef<HTMLPreElement>(null);
  const [highlightedCode, setHighlightedCode] = useState("");
  const [framework, setFramework] = useState("");
  const { toast } = useToast();

  useEffect(() => {
    // Simple syntax highlighting for code
    const processedCode = code
      .replace(/\b(import|from|as|def|class|for|in|if|else|return|try|except|with|yield|lambda)\b/g, '<span class="text-blue-300">$1</span>')
      .replace(/\b(self|None|True|False)\b/g, '<span class="text-blue-300">$1</span>')
      .replace(/\b([\w\.]+)\(/g, '<span class="text-yellow-300">$1</span>(')
      .replace(/(["'])(?:\\\1|.)*?\1/g, '<span class="text-green-300">$&</span>')
      .replace(/(#.*)$/gm, '<span class="text-gray-500">$1</span>')
      .replace(/\b([A-Z][A-Za-z]+)\b/g, '<span class="text-green-300">$1</span>');
      
    setHighlightedCode(processedCode);
    
    // Detect the framework
    const detectedFramework = detectFramework(code);
    setFramework(detectedFramework);
  }, [code]);

  const analyzeCode = () => {
    if (!codeRef.current) return;
    
    const params = identifyHyperparameters(code);
    let tempCode = code;
    let offset = 0;
    
    // Sort parameters by position in reverse to not mess up indices
    params.sort((a, b) => b.position.start - a.position.start);
    
    for (const param of params) {
      const start = param.position.start + offset;
      const end = param.position.end + offset;
      
      const replacement = `<span data-param="${param.key}" data-value="${param.value}" class="bg-yellow-200 px-1 rounded cursor-pointer">${tempCode.substring(start, end)}</span>`;
      
      tempCode = tempCode.substring(0, start) + replacement + tempCode.substring(end);
      offset += replacement.length - (end - start);
    }
    
    // Apply our base syntax highlighting
    const highlightedCode = tempCode
      .replace(/\b(import|from|as|def|class|for|in|if|else|return|try|except|with|yield|lambda)\b/g, '<span class="text-blue-300">$1</span>')
      .replace(/\b(self|None|True|False)\b/g, '<span class="text-blue-300">$1</span>')
      .replace(/\b([\w\.]+)(?=\()/g, '<span class="text-yellow-300">$1</span>')
      .replace(/(["'])(?:\\\1|.)*?\1/g, '<span class="text-green-300">$&</span>')
      .replace(/(#.*)$/gm, '<span class="text-gray-500">$1</span>')
      .replace(/\b([A-Z][A-Za-z]+)\b/g, '<span class="text-green-300">$1</span>');
      
    setHighlightedCode(highlightedCode);
    
    // Add event listeners to the highlighted parameters
    setTimeout(() => {
      const paramElements = codeRef.current?.querySelectorAll('[data-param]');
      if (paramElements) {
        paramElements.forEach(element => {
          element.addEventListener('click', () => {
            const paramKey = element.getAttribute('data-param');
            const paramValue = element.getAttribute('data-value');
            if (paramKey && paramValue) {
              onParameterClick(paramKey, paramValue);
            }
          });
        });
      }
      
      toast({
        title: "Code Analyzed",
        description: `${params.length} hyperparameters identified in ${framework} code.`,
        duration: 3000,
      });
    }, 0);
  };

  return (
    <Card className="relative">
      <pre 
        ref={codeRef}
        className="bg-gray-800 text-gray-100 p-4 rounded-lg font-mono text-sm overflow-x-auto"
        dangerouslySetInnerHTML={{ __html: highlightedCode }}
      />
      <div className="absolute top-2 right-2">
        <Button 
          size="sm" 
          className="text-xs bg-primary-500 hover:bg-primary-600 text-white px-2 py-1 rounded"
          onClick={analyzeCode}
        >
          Analyze
        </Button>
      </div>
    </Card>
  );
};

export default CodeBlock;
