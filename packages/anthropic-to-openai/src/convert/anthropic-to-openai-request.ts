import {
  AnthropicToOpenAIPrompt,
  AnthropicToOpenAISystemMessage,
  AnthropicToOpenAIUserMessage,
  AnthropicToOpenAIAssistantMessage,
  AnthropicToOpenAIToolMessage,
  AnthropicToOpenAIChatOptions,
  AnthropicToOpenAIFunctionTool,
  AnthropicToOpenAIToolChoice,
  AnthropicToOpenAIThinkingConfig,
  AnthropicToOpenAIMessageToolCall,
} from './types';

interface AnthropicContentBlock {
  type: 'text' | 'image' | 'document' | 'tool_use' | 'tool_result' | 'thinking';
  text?: string;
  source?: {
    type: 'base64' | 'url';
    media_type?: string;
    data?: string;
    url?: string;
  };
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: string | unknown;
  is_error?: boolean;
  thinking?: string;
  signature?: string;
}

interface AnthropicMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | Array<AnthropicContentBlock>;
  tool_calls?: Array<{
    id: string;
    type: 'tool_use';
    name: string;
    input: Record<string, unknown>;
  }>;
  tool_call_id?: string;
}

function convertAnthropicContentToOpenAI(
  content: string | Array<AnthropicContentBlock>,
): string | Array<unknown> {
  if (typeof content === 'string') {
    return content;
  }

  const convertedParts: Array<unknown> = [];

  for (const block of content) {
    switch (block.type) {
      case 'text':
        convertedParts.push({
          type: 'text',
          text: block.text ?? '',
        });
        break;

      case 'image':
        if (block.source) {
          if (block.source.type === 'base64') {
            convertedParts.push({
              type: 'image_url',
              image_url: {
                url: `data:${block.source.media_type};base64,${block.source.data}`,
              },
            });
          } else if (block.source.type === 'url') {
            convertedParts.push({
              type: 'image_url',
              image_url: { url: block.source.url },
            });
          }
        }
        break;

      case 'document':
        if (block.source) {
          const docPart: Record<string, unknown> = {
            type: 'file',
            file: {},
          };
          if (block.source.type === 'base64') {
            docPart.file = {
              filename: block.title ?? 'document',
              file_data: `data:${block.source.media_type};base64,${block.source.data}`,
            };
          } else if (block.source.type === 'url') {
            docPart.file = {
              filename: block.title ?? 'document',
              file_data: block.source.url,
            };
          }
          convertedParts.push(docPart);
        }
        break;

      case 'tool_use':
        break;

      case 'tool_result':
        break;
    }
  }

  return convertedParts.length === 1 ? (convertedParts[0] as string) : convertedParts;
}

export function convertAnthropicToOpenAIMessages(
  messages: AnthropicMessage[],
  systemPrompt?: string,
): {
  messages: Array<{
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string | Array<unknown>;
    tool_calls?: Array<AnthropicToOpenAIMessageToolCall>;
    tool_call_id?: string;
    reasoning_content?: string;
  }>;
  system?: string;
} {
  const openAIMessages: Array<{
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string | Array<unknown>;
    tool_calls?: Array<AnthropicToOpenAIMessageToolCall>;
    tool_call_id?: string;
    reasoning_content?: string;
  }> = [];

  let systemContent = systemPrompt ?? '';

  for (const msg of messages) {
    switch (msg.role) {
      case 'system':
        if (typeof msg.content === 'string') {
          systemContent = msg.content;
        } else {
          for (const block of msg.content) {
            if (block.type === 'text') {
              systemContent = block.text ?? '';
              break;
            }
          }
        }
        break;

      case 'user':
        openAIMessages.push({
          role: 'user',
          content: convertAnthropicContentToOpenAI(msg.content),
        });
        break;

      case 'assistant':
        const assistantMsg: {
          role: 'assistant';
          content: string | Array<unknown>;
          tool_calls?: Array<AnthropicToOpenAIMessageToolCall>;
          reasoning_content?: string;
        } = {
          role: 'assistant',
          content: '',
        };

        if (typeof msg.content === 'string') {
          assistantMsg.content = msg.content;
        } else {
          const contentParts: Array<unknown> = [];
          for (const block of msg.content) {
            if (block.type === 'text') {
              contentParts.push({ type: 'text', text: block.text ?? '' });
            } else if (block.type === 'thinking') {
              assistantMsg.reasoning_content = block.thinking;
            } else if (block.type === 'tool_use') {
              contentParts.push({
                type: 'tool_use',
                id: block.id,
                name: block.name,
                input: block.input,
              });
            }
          }
          assistantMsg.content = contentParts;
        }

        if (msg.tool_calls && msg.tool_calls.length > 0) {
          assistantMsg.tool_calls = msg.tool_calls.map((tc) => ({
            type: 'function',
            id: tc.id,
            function: {
              name: tc.name,
              arguments: JSON.stringify(tc.input),
            },
          }));
        }

        openAIMessages.push(assistantMsg);
        break;

      case 'tool':
        openAIMessages.push({
          role: 'tool',
          content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content),
          tool_call_id: msg.tool_call_id ?? '',
        });
        break;
    }
  }

  return {
    messages: openAIMessages,
    system: systemContent,
  };
}

export function convertAnthropicToolsToOpenAI(
  tools: Array<{ name: string; description?: string; input_schema?: Record<string, unknown> }>,
): Array<AnthropicToOpenAIFunctionTool> {
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
    },
  }));
}

export function convertAnthropicToolChoiceToOpenAI(
  toolChoice: AnthropicToOpenAIToolChoice | undefined,
): { type?: string; function?: { name: string } } | undefined {
  if (!toolChoice) return undefined;

  switch (toolChoice.type) {
    case 'none':
      return { type: 'none' };
    case 'auto':
      return { type: 'auto' };
    case 'required':
      return { type: 'required' };
    case 'function':
      return { type: 'function', function: { name: toolChoice.function.name } };
    default:
      return undefined;
  }
}

export function convertAnthropicThinkingToOpenAI(
  thinking: AnthropicToOpenAIThinkingConfig | undefined,
): { type?: string; max_tokens?: number } | undefined {
  if (!thinking) return undefined;

  switch (thinking.type) {
    case 'disabled':
      return { type: 'disabled' };
    case 'enabled':
      return { type: 'enabled', max_tokens: thinking.budget_tokens };
    case 'adaptive':
      return { type: 'enabled', max_tokens: thinking.budget_tokens };
    default:
      if (thinking.budget_tokens) {
        return { type: 'enabled', max_tokens: thinking.budget_tokens };
      }
      return undefined;
  }
}

export function convertAnthropicRequestToOpenAI(
  model: string,
  messages: AnthropicMessage[],
  options: AnthropicToOpenAIChatOptions = {},
): Record<string, unknown> {
  const { messages: openAIMessages, system } = convertAnthropicToOpenAIMessages(
    messages,
    options.system,
  );

  const request: Record<string, unknown> = {
    model,
    messages: openAIMessages,
  };

  if (system) {
    request.system = system;
  }

  if (options.max_tokens !== undefined) {
    request.max_tokens = options.max_tokens;
  }

  if (options.temperature !== undefined) {
    request.temperature = options.temperature;
  }

  if (options.top_p !== undefined) {
    request.top_p = options.top_p;
  }

  if (options.stop !== undefined) {
    request.stop = options.stop;
  }

  if (options.tools && options.tools.length > 0) {
    request.tools = convertAnthropicToolsToOpenAI(
      options.tools.map((t) => ({
        name: t.function.name,
        description: t.function.description,
        input_schema: t.function.parameters,
      })),
    );
  }

  if (options.tool_choice) {
    request.tool_choice = convertAnthropicToolChoiceToOpenAI(options.tool_choice);
  }

  const thinking = convertAnthropicThinkingToOpenAI(options.thinking);
  if (thinking) {
    if (thinking.type === 'disabled') {
      request.reasoning_effort = 'none';
    } else if (thinking.max_tokens) {
      request.max_completion_tokens = thinking.max_tokens;
    }
  }

  return request;
}
