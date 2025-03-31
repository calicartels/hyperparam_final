import React, { useState, useEffect } from 'react';
import { 
  Card,
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Check, ChevronRight, X, AlertCircle, Info, Lightbulb, BookOpen, Settings } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

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

const tutorialSteps: TutorialStep[] = [
  {
    title: "Welcome to HyperExplainer",
    description: "This extension helps you understand machine learning hyperparameters found in code. Let's see how it works!",
    icon: <Info className="w-8 h-8 text-primary" />,
    image: "/tutorial-images/tutorial-welcome.svg",
    cta: "Let's start!"
  },
  {
    title: "Find Hyperparameters",
    description: "When you're viewing code on websites like ChatGPT, HyperExplainer automatically highlights hyperparameters that control machine learning models.",
    icon: <AlertCircle className="w-8 h-8 text-blue-500" />,
    image: "/tutorial-images/tutorial-highlight.svg",
    cta: "Next"
  },
  {
    title: "Get Detailed Explanations",
    description: "Click on any highlighted parameter to see a detailed explanation of what it does and how it affects model training.",
    icon: <BookOpen className="w-8 h-8 text-amber-500" />,
    image: "/tutorial-images/tutorial-explanation.svg",
    cta: "Next"
  },
  {
    title: "Explore Alternatives",
    description: "Each explanation includes alternative values you might want to try, with insights on when to use them.",
    icon: <Lightbulb className="w-8 h-8 text-yellow-500" />,
    image: "/tutorial-images/tutorial-alternatives.svg",
    cta: "Next"
  },
  {
    title: "Customize Settings",
    description: "Access the extension options to customize how HyperExplainer works. You can change highlighting colors, control which frameworks are detected, and more.",
    icon: <Settings className="w-8 h-8 text-violet-500" />,
    image: "/tutorial-images/tutorial-settings.svg",
    cta: "Got it!"
  }
];

export function Tutorial({ onComplete, onDismiss, isVisible }: TutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [exiting, setExiting] = useState(false);
  const progress = ((currentStep + 1) / tutorialSteps.length) * 100;

  const handleNext = () => {
    if (currentStep < tutorialSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      setExiting(true);
      setTimeout(() => {
        onComplete();
      }, 300);
    }
  };

  const handleDismiss = () => {
    setExiting(true);
    setTimeout(() => {
      onDismiss();
    }, 300);
  };

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      {!exiting && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <motion.div
            className="w-full max-w-md p-2"
            initial={{ scale: 0.9, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.9, y: 20 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
          >
            <Card className="w-full overflow-hidden border-2 border-primary/20 shadow-lg">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {tutorialSteps[currentStep].icon}
                    <CardTitle className="text-xl">
                      {tutorialSteps[currentStep].title}
                    </CardTitle>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="icon" 
                    className="rounded-full h-8 w-8" 
                    onClick={handleDismiss}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                <Progress value={progress} className="h-1 mt-2" />
              </CardHeader>
              
              <CardContent className="px-6 py-4">
                <div className="min-h-[100px] flex flex-col items-center gap-4">
                  <CardDescription className="text-base text-center">
                    {tutorialSteps[currentStep].description}
                  </CardDescription>
                  
                  {tutorialSteps[currentStep].image && (
                    <div className="w-full h-48 bg-muted rounded-md flex items-center justify-center overflow-hidden">
                      <img 
                        src={tutorialSteps[currentStep].image} 
                        alt={tutorialSteps[currentStep].title}
                        className="object-cover w-full h-full"
                      />
                    </div>
                  )}
                </div>
              </CardContent>
              
              <CardFooter className="flex justify-between border-t bg-muted/50 px-6 py-3">
                <div className="flex gap-1">
                  {tutorialSteps.map((_, index) => (
                    <div
                      key={index}
                      className={cn(
                        "w-2 h-2 rounded-full transition-all",
                        currentStep === index 
                          ? "bg-primary w-4" 
                          : index < currentStep 
                            ? "bg-primary/40" 
                            : "bg-muted-foreground/20"
                      )}
                    />
                  ))}
                </div>
                <Button onClick={handleNext} className="gap-1">
                  {tutorialSteps[currentStep].cta || "Next"}
                  {currentStep < tutorialSteps.length - 1 ? (
                    <ChevronRight className="h-4 w-4" />
                  ) : (
                    <Check className="h-4 w-4" />
                  )}
                </Button>
              </CardFooter>
            </Card>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}