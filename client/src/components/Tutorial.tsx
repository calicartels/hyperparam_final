import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { ArrowLeft, ArrowRight, Book, Code, Lightbulb, Settings, X } from 'lucide-react';

type TutorialStep = {
  title: string;
  description: string;
  icon: React.ReactNode;
  image?: string;
  cta?: string;
};

interface TutorialProps {
  onComplete: () => void;
  onDismiss: () => void;
  isVisible: boolean;
}

export function Tutorial({ onComplete, onDismiss, isVisible }: TutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);
  
  const steps: TutorialStep[] = [
    {
      title: "Welcome to HyperExplainer",
      description: "Discover what hyperparameters are and how they affect machine learning models. This extension helps you understand and optimize the parameters in ML code.",
      icon: <Book className="h-6 w-6 text-indigo-500" />,
      image: "tutorial-welcome.svg",
      cta: "Get Started",
    },
    {
      title: "Parameter Detection",
      description: "HyperExplainer automatically detects hyperparameters in ML code on the page. Parameters are highlighted based on their impact level.",
      icon: <Code className="h-6 w-6 text-indigo-500" />,
      image: "tutorial-highlight.svg",
      cta: "Next",
    },
    {
      title: "Detailed Explanations",
      description: "Click on any parameter to see a detailed explanation of what it does, how it affects your model, and best practices for setting it.",
      icon: <Lightbulb className="h-6 w-6 text-indigo-500" />,
      image: "tutorial-explanation.svg",
      cta: "Next",
    },
    {
      title: "Alternative Values",
      description: "Explore alternative parameter values with visualizations showing how each option affects model performance.",
      icon: <ArrowRight className="h-6 w-6 text-indigo-500" />,
      image: "tutorial-alternatives.svg",
      cta: "Next",
    },
    {
      title: "Customize Settings",
      description: "Configure which ML frameworks to support, set highlight colors, and customize other extension behaviors.",
      icon: <Settings className="h-6 w-6 text-indigo-500" />,
      image: "tutorial-settings.svg",
      cta: "Finish",
    },
  ];

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  // Reset to first step when tutorial becomes visible
  useEffect(() => {
    if (isVisible) {
      setCurrentStep(0);
    }
  }, [isVisible]);

  if (!isVisible) return null;

  const currentStepData = steps[currentStep];

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg max-w-md w-full shadow-xl relative overflow-hidden flex flex-col">
        {/* Close button */}
        <button 
          onClick={onDismiss}
          className="absolute right-3 top-3 text-gray-400 hover:text-gray-600 z-10"
        >
          <X size={20} />
        </button>
        
        {/* Tutorial step indicator */}
        <div className="flex space-x-1 absolute top-4 left-1/2 transform -translate-x-1/2 z-10">
          {steps.map((_, index) => (
            <div
              key={index}
              className={`h-1.5 rounded-full ${
                index === currentStep ? 'bg-indigo-500 w-5' : 'bg-gray-200 w-2'
              } transition-all duration-300`}
            />
          ))}
        </div>
        
        {/* Step content */}
        <div className="flex flex-col items-center px-6 pt-12 pb-6">
          {currentStepData.image && (
            <div className="w-full h-48 flex items-center justify-center mb-6">
              <img 
                src={`tutorial-images/${currentStepData.image}`} 
                alt={currentStepData.title}
                className="max-h-full max-w-full object-contain"
              />
            </div>
          )}
          
          <div className="bg-indigo-50 p-3 rounded-full mb-4">
            {currentStepData.icon}
          </div>
          
          <h3 className="text-lg font-bold text-gray-900 mb-2 text-center">
            {currentStepData.title}
          </h3>
          
          <p className="text-gray-600 text-center mb-6">
            {currentStepData.description}
          </p>
        </div>
        
        {/* Navigation buttons */}
        <div className="flex justify-between items-center px-6 pb-6">
          <Button
            variant="ghost"
            onClick={handlePrevious}
            disabled={currentStep === 0}
            className={currentStep === 0 ? 'invisible' : ''}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          
          <Button onClick={handleNext} className="bg-indigo-600 hover:bg-indigo-700">
            {currentStepData.cta || "Next"}
            {currentStep < steps.length - 1 && <ArrowRight className="ml-2 h-4 w-4" />}
          </Button>
        </div>
      </div>
    </div>
  );
}