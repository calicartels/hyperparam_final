import { apiRequest } from "./queryClient";

/**
 * Interface for the hyperparameter explanation request
 */
export interface ExplainHyperparameterRequest {
  paramName: string;
  paramValue: string;
  framework?: string;
  codeContext?: string;
}

/**
 * Interface for the alternative value suggestion
 */
export interface AlternativeValue {
  value: string;
  description: string;
  type: 'higher' | 'lower' | 'advanced' | 'extreme';
}

/**
 * Interface for the structured LLM explanation response
 */
export interface HyperparameterExplanation {
  name: string;
  description: string;
  impact: 'high' | 'medium' | 'low';
  valueAnalysis: string;
  alternatives: AlternativeValue[];
  bestPractices: string;
  tradeoffs: string;
}

/**
 * Interface for the full LLM API response
 */
export interface LLMExplanationResponse {
  success: boolean;
  explanation: HyperparameterExplanation;
  error?: string;
  rawResponse?: string;
  fallbackAvailable?: boolean;
}

/**
 * Interface for the LLM status response
 */
export interface LLMStatusResponse {
  available: boolean;
  provider: string;
  model: string;
  requiresAuth: boolean;
}

/**
 * Check if LLM service is available
 * @returns Promise with LLM status
 */
export async function checkLLMStatus(): Promise<LLMStatusResponse> {
  const response = await apiRequest<LLMStatusResponse>({
    method: "GET",
    url: "/api/llm/status",
    on401: "returnNull",
  });
  
  return response;
}

/**
 * Get a hyperparameter explanation from the LLM service
 * @param request Hyperparameter explanation request
 * @returns Promise with explanation
 */
export async function getHyperparameterExplanation(
  request: ExplainHyperparameterRequest
): Promise<LLMExplanationResponse> {
  const response = await apiRequest<LLMExplanationResponse>({
    method: "POST",
    url: "/api/llm/explain-hyperparameter",
    body: request,
    on401: "returnNull",
  });
  
  return response;
}

/**
 * Generate fallback content if LLM service is unavailable
 * @param paramName Hyperparameter name
 * @param paramValue Parameter value
 * @param framework Optional framework name
 * @returns Fallback explanation
 */
export function generateFallbackExplanation(
  paramName: string,
  paramValue: string,
  framework?: string
): HyperparameterExplanation {
  // Convert paramName to a more readable format
  const formattedName = paramName
    .replace(/_/g, ' ')
    .replace(/(\w)(\w*)/g, (_, first, rest) => first.toUpperCase() + rest.toLowerCase());
  
  // Generate a generic explanation
  return {
    name: formattedName,
    description: `${formattedName} is a hyperparameter that controls an aspect of model training or architecture.`,
    impact: 'medium',
    valueAnalysis: `The value ${paramValue} is a common setting that provides a balance of performance and generalization.`,
    alternatives: [
      {
        value: Number(paramValue) * 0.5 + '',
        description: "A lower value may provide better generalization but slower training.",
        type: "lower"
      },
      {
        value: Number(paramValue) * 2 + '',
        description: "A higher value may lead to faster training but could reduce generalization.",
        type: "higher"
      }
    ],
    bestPractices: "It's typically recommended to start with the default value and adjust based on validation performance.",
    tradeoffs: "Modifying this parameter often involves a tradeoff between training speed, model performance, and generalization ability."
  };
}