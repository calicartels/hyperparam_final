# HyperExplainer Chrome Extension

A Chrome extension that identifies, explains, and suggests alternatives for hyperparameters in LLM-generated code.

## Features

- Automatically detects hyperparameters in code blocks on websites like ChatGPT
- Provides detailed explanations of hyperparameters and their impacts
- Suggests alternative values with visual comparisons
- Works with major ML frameworks including TensorFlow, PyTorch, Keras, and more
- Uses Google Cloud Vertex AI for advanced AI-powered explanations (optional)
- Works offline with fallback explanations when no API keys are available

## Setting Up Local Development

### Prerequisites

- Node.js and npm
- A Google Cloud account with Vertex AI API access (optional)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```

3. Set up environment variables by creating a `.env` file:
   ```
   GOOGLE_PROJECT_ID=your-google-project-id
   GOOGLE_LOCATION=us-central1
   ```

4. Place your Google Cloud service account key in the `credentials` directory with the name:
   ```
   695116221974-hqg9ap5bh4ok0nc23bbbkh8h30nhqa8d.apps.googleusercontent.com
   ```

5. Start the development server:
   ```
   npm run dev
   ```

### Testing the API

1. After starting the development server, navigate to `http://localhost:5000/test-api`
2. Use the test form to try different hyperparameters and see the explanations
3. Toggle between using the LLM API and the fallback mechanism

## Loading the Extension in Chrome

1. Build the extension:
   ```
   npm run build
   ```

2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" in the top-right corner
4. Click "Load unpacked" and select the `dist` directory
5. The extension should now be active in your browser

## Using with ChatGPT

1. Go to ChatGPT (or another supported site)
2. Ask for code examples that include hyperparameters (e.g., "Show me how to train a neural network with TensorFlow")
3. When code blocks appear, HyperExplainer will automatically identify and highlight hyperparameters
4. Click on the highlighted parameters to see explanations and alternatives

## Structure

- `client/` - Frontend code for the Chrome extension
  - `public/` - Static files including content scripts
  - `src/` - React components and application logic
- `server/` - Backend API for LLM integration
- `shared/` - Shared types and schemas

## Configuration Options

The extension can be configured through the options page, accessible by right-clicking the extension icon and selecting "Options".

## Working Without API Keys

HyperExplainer works even without API keys but provides more detailed explanations when connected to Google Cloud Vertex AI.