import { useState, useEffect } from 'react';

// Local storage key for tutorial state
const TUTORIAL_STORAGE_KEY = 'hyperexplainer_tutorial_completed';

export function useTutorial() {
  // State to track if tutorial has been shown
  const [shouldShowTutorial, setShouldShowTutorial] = useState(false);
  const [isTutorialVisible, setIsTutorialVisible] = useState(false);
  const [tutorialLoaded, setTutorialLoaded] = useState(false);

  // Load tutorial state on initial mount
  useEffect(() => {
    const hasCompletedTutorial = localStorage.getItem(TUTORIAL_STORAGE_KEY) === 'true';
    setShouldShowTutorial(!hasCompletedTutorial);
    setTutorialLoaded(true);
  }, []);

  // Show tutorial when appropriate (only after loaded and if not previously completed)
  useEffect(() => {
    if (tutorialLoaded && shouldShowTutorial) {
      // Small delay to show tutorial after component mounts
      const timer = setTimeout(() => {
        setIsTutorialVisible(true);
      }, 1000);
      
      return () => clearTimeout(timer);
    }
  }, [tutorialLoaded, shouldShowTutorial]);

  const completeTutorial = () => {
    localStorage.setItem(TUTORIAL_STORAGE_KEY, 'true');
    setShouldShowTutorial(false);
    setIsTutorialVisible(false);
  };

  const dismissTutorial = () => {
    localStorage.setItem(TUTORIAL_STORAGE_KEY, 'true');
    setShouldShowTutorial(false);
    setIsTutorialVisible(false);
  };

  const resetTutorial = () => {
    localStorage.removeItem(TUTORIAL_STORAGE_KEY);
    setShouldShowTutorial(true);
    setIsTutorialVisible(true);
  };

  const showTutorial = () => {
    setIsTutorialVisible(true);
  };

  return {
    isTutorialVisible,
    completeTutorial,
    dismissTutorial,
    resetTutorial,
    showTutorial
  };
}