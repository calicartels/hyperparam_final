// Type definitions for Chrome extension APIs
// These are simplified definitions for the purposes of our project

interface Chrome {
  tabs: {
    query: (
      queryInfo: { active: boolean; currentWindow: boolean },
      callback: (tabs: { id?: number }[]) => void
    ) => void;
    sendMessage: (
      tabId: number,
      message: any,
      callback?: (response: any) => void
    ) => void;
    create: (createProperties: { url: string }) => void;
  };
  runtime: {
    openOptionsPage: () => void;
    getURL: (path: string) => string;
  };
}

// Add Chrome namespace to Window interface
interface Window {
  chrome: Chrome;
}