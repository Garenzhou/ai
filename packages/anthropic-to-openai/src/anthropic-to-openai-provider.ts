import {
  LanguageModelV4,
  NoSuchModelError,
  ProviderV4,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  loadApiKey,
  withoutTrailingSlash,
  withUserAgentSuffix,
} from '@ai-sdk/provider-utils';
import { AnthropicToOpenAIChatLanguageModel } from './anthropic-to-openai-chat-language-model';
import { AnthropicToOpenAIChatModelId, AnthropicToOpenAILanguageModelChatOptions } from './anthropic-to-openai-chat-options';
import { VERSION } from './version';

export interface AnthropicToOpenAIProviderSettings {
  baseURL: string;
  apiKey?: string;
  headers?: Record<string, string>;
  fetch?: FetchFunction;
}

export interface AnthropicToOpenAIProvider extends ProviderV4 {
  (modelId: AnthropicToOpenAIChatModelId): LanguageModelV4;
  languageModel(modelId: AnthropicToOpenAIChatModelId): LanguageModelV4;
  chat(modelId: AnthropicToOpenAIChatModelId): LanguageModelV4;
}

export function createAnthropicToOpenAI(
  options: AnthropicToOpenAIProviderSettings,
): AnthropicToOpenAIProvider {
  if (!options.baseURL) {
    throw new Error(
      'AnthropicToOpenAIProvider requires a baseURL to be specified',
    );
  }

  const baseURL = withoutTrailingSlash(options.baseURL);

  const getHeaders = () => {
    const headers: Record<string, string> = {
      ...options.headers,
    };
    if (options.apiKey) {
      headers['Authorization'] = `Bearer ${options.apiKey}`;
    }
    return withUserAgentSuffix(
      headers,
      `ai-sdk/anthropic-to-openai/${VERSION}`,
    );
  };

  const createLanguageModel = (modelId: AnthropicToOpenAIChatModelId) => {
    return new AnthropicToOpenAIChatLanguageModel(modelId, {
      baseURL,
      headers: getHeaders,
      fetch: options.fetch,
    });
  };

  const provider = (modelId: AnthropicToOpenAIChatModelId) =>
    createLanguageModel(modelId);

  provider.specificationVersion = 'v4' as const;
  provider.languageModel = createLanguageModel;
  provider.chat = createLanguageModel;

  provider.embeddingModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: 'embeddingModel' });
  };
  provider.textEmbeddingModel = provider.embeddingModel;
  provider.imageModel = (modelId: string) => {
    throw new NoSuchModelError({ modelId, modelType: 'imageModel' });
  };

  return provider;
}

export const anthropicToOpenAI = createAnthropicToOpenAI({
  baseURL: '',
});
