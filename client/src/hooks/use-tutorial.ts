import { useState, useEffect } from 'react';

const TUTORIAL_SHOWN_KEY = 'hyperexplainer-tutorial-shown';

export function useTutorial() {
  const [isTutorialVisible, setIsTutorialVisible] = useState(false);
  const [hasSeenTutorial, setHasSeenTutorial] = useState(true); // Default to true until we check storage

  useEffect(() => {
    // Check if the tutorial has been shown before
    const tutorialShown = localStorage.getItem(TUTORIAL_SHOWN_KEY);
    if (tutorialShown !== 'true') {
      setHasSeenTutorial(false);
      setIsTutorialVisible(true);
    }
  }, []);

  const completeTutorial = () => {
    localStorage.setItem(TUTORIAL_SHOWN_KEY, 'true');
    setHasSeenTutorial(true);
    setIsTutorialVisible(false);
  };

  const dismissTutorial = () => {
    localStorage.setItem(TUTORIAL_SHOWN_KEY, 'true');
    setHasSeenTutorial(true);
    setIsTutorialVisible(false);
  };

  const resetTutorial = () => {
    localStorage.removeItem(TUTORIAL_SHOWN_KEY);
    setHasSeenTutorial(false);
    setIsTutorialVisible(true);
  };

  const showTutorial = () => {
    setIsTutorialVisible(true);
  };

  const hideTutorial = () => {
    setIsTutorialVisible(false);
  };

  return {
    isTutorialVisible,
    hasSeenTutorial,
    completeTutorial,
    dismissTutorial,
    resetTutorial,
    showTutorial,
    hideTutorial
  };
}