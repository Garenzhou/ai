import { JSONValue } from '@ai-sdk/provider';

export type AnthropicToOpenAIPrompt = Array<AnthropicToOpenAIMessage>;

export type AnthropicToOpenAIMessage =
  | AnthropicToOpenAISystemMessage
  | AnthropicToOpenAIUserMessage
  | AnthropicToOpenAIAssistantMessage
  | AnthropicToOpenAIToolMessage;

type JsonRecord<T = never> = Record<
  string,
  JSONValue | JSONValue[] | T | T[] | undefined
>;

export interface AnthropicToOpenAISystemMessage extends JsonRecord {
  role: 'system';
  content: string;
}

export type AnthropicToOpenAIContentPart =
  | AnthropicToOpenAITextContentPart
  | AnthropicToOpenAIImageContentPart
  | AnthropicToOpenAIDocumentContentPart
  | AnthropicToOpenAIToolResultContentPart;

export interface AnthropicToOpenAITextContentPart extends JsonRecord {
  type: 'text';
  text: string;
}

export interface AnthropicToOpenAIImageContentPart extends JsonRecord {
  type: 'image';
  source: {
    type: 'base64' | 'url';
    media_type?: string;
    data?: string;
    url?: string;
  };
}

export interface AnthropicToOpenAIDocumentContentPart extends JsonRecord {
  type: 'document';
  source: {
    type: 'base64' | 'url';
    media_type?: string;
    data?: string;
    url?: string;
  };
  title?: string;
}

export interface AnthropicToOpenAIToolResultContentPart extends JsonRecord {
  type: 'tool_result';
  tool_use_id: string;
  content: string | Array<AnthropicToOpenAITextContentPart>;
  is_error?: boolean;
}

export interface AnthropicToOpenAIUserMessage extends JsonRecord<AnthropicToOpenAIContentPart> {
  role: 'user';
  content: string | Array<AnthropicToOpenAIContentPart>;
}

export interface AnthropicToOpenAIThinkingContent extends JsonRecord {
  type: 'thinking';
  thinking: string;
  signature?: string;
}

export interface AnthropicToOpenAIToolCallContent extends JsonRecord {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

export interface AnthropicToOpenAIAssistantMessage extends JsonRecord {
  role: 'assistant';
  content?: string | null;
  reasoning_content?: string;
  tool_calls?: Array<AnthropicToOpenAIMessageToolCall>;
}

export interface AnthropicToOpenAIMessageToolCall extends JsonRecord {
  type: 'function';
  id: string;
  function: {
    arguments: string;
    name: string;
  };
}

export interface AnthropicToOpenAIToolMessage extends JsonRecord {
  role: 'tool';
  content: string;
  tool_call_id: string;
}

export interface AnthropicToOpenAIFunctionTool extends JsonRecord {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface AnthropicToOpenAIChatOptions {
  model?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  stream?: boolean;
  tools?: Array<AnthropicToOpenAIFunctionTool>;
  tool_choice?: AnthropicToOpenAIToolChoice;
  thinking?: AnthropicToOpenAIThinkingConfig;
  system?: string;
}

export type AnthropicToOpenAIToolChoice =
  | { type: 'none' }
  | { type: 'auto' }
  | { type: 'required' }
  | { type: 'function'; function: { name: string } };

export interface AnthropicToOpenAIThinkingConfig {
  type: 'enabled' | 'disabled' | 'adaptive';
  budget_tokens?: number;
}

export interface AnthropicToOpenAIResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: string | null;
      reasoning_content?: string;
      tool_calls?: Array<AnthropicToOpenAIMessageToolCall>;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface AnthropicToOpenAIStreamChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: 'assistant';
      content?: string;
      reasoning_content?: string;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: 'function';
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}
