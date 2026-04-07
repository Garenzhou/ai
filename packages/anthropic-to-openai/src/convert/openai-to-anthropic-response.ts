import {
  AnthropicToOpenAIResponse,
  AnthropicToOpenAIStreamChunk,
} from './types';

interface AnthropicStreamEvent {
  type: string;
  index?: number;
  content_block?: {
    type: string;
    id?: string;
    name?: string;
  };
  delta?: {
    type: string;
    text?: string;
    thinking?: string;
    partial_json?: string;
  };
  message?: {
    id?: string;
    type?: string;
    role?: string;
    content?: string;
    model?: string;
  };
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
}

function mapOpenAIStopReasonToAnthropic(
  reason: string | null,
): 'end_turn' | 'tool_use' | 'max_tokens' | 'stop_sequence' | null {
  switch (reason) {
    case 'stop':
      return 'end_turn';
    case 'tool_calls':
      return 'tool_use';
    case 'length':
      return 'max_tokens';
    case null:
      return null;
    default:
      return 'end_turn';
  }
}

export function convertOpenAIResponseToAnthropic(
  openAIResponse: AnthropicToOpenAIResponse,
): {
  id: string;
  type: 'message';
  role: 'assistant';
  content: Array<{
    type: 'text' | 'tool_use' | 'thinking';
    text?: string;
    id?: string;
    name?: string;
    input?: Record<string, unknown>;
    thinking?: string;
    signature?: string;
  }>;
  model: string;
  stop_reason: 'end_turn' | 'tool_use' | 'max_tokens' | 'stop_sequence' | null;
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
} {
  const choice = openAIResponse.choices[0];
  const content: Array<{
    type: 'text' | 'tool_use' | 'thinking';
    text?: string;
    id?: string;
    name?: string;
    input?: Record<string, unknown>;
    thinking?: string;
    signature?: string;
  }> = [];

  if (choice.message.reasoning_content) {
    content.push({
      type: 'thinking',
      thinking: choice.message.reasoning_content,
    });
  }

  if (choice.message.content) {
    content.push({
      type: 'text',
      text: choice.message.content,
    });
  }

  if (choice.message.tool_calls) {
    for (const tc of choice.message.tool_calls) {
      let input: Record<string, unknown> = {};
      try {
        input = JSON.parse(tc.function.arguments);
      } catch {
        input = {};
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
      });
    }
  }

  const usage = openAIResponse.usage ?? {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
  };

  return {
    id: openAIResponse.id,
    type: 'message',
    role: 'assistant',
    content,
    model: openAIResponse.model,
    stop_reason: mapOpenAIStopReasonToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens,
      output_tokens: usage.completion_tokens,
    },
  };
}

export function convertOpenAIStreamChunkToAnthropicEvents(
  chunk: AnthropicToOpenAIStreamChunk,
  isFirst: boolean,
): AnthropicStreamEvent[] {
  const events: AnthropicStreamEvent[] = [];
  const choice = chunk.choices[0];

  if (isFirst) {
    events.push({
      type: 'message_start',
      message: {
        id: chunk.id,
        type: 'message',
        role: 'assistant',
        model: chunk.model,
      },
    });
  }

  if (choice.delta.role) {
    events.push({
      type: 'content_block_start',
      content_block: {
        type: 'text',
      },
      index: 0,
    });
  }

  if (choice.delta.reasoning_content) {
    events.push({
      type: 'content_block_start',
      content_block: {
        type: 'thinking',
      },
      index: 0,
    });
    events.push({
      type: 'content_block_delta',
      index: 0,
      delta: {
        type: 'thinking_delta',
        thinking: choice.delta.reasoning_content,
      },
    });
  }

  if (choice.delta.content !== undefined) {
    if (choice.delta.content) {
      events.push({
        type: 'content_block_delta',
        index: 0,
        delta: {
          type: 'text_delta',
          text: choice.delta.content,
        },
      });
    }
  }

  if (choice.delta.tool_calls && choice.delta.tool_calls.length > 0) {
    for (const tc of choice.delta.tool_calls) {
      const tcIndex = tc.index ?? 0;

      if (tc.id && tc.function?.name) {
        events.push({
          type: 'content_block_start',
          index: tcIndex,
          content_block: {
            type: 'tool_use',
            id: tc.id,
            name: tc.function.name,
          },
        });
      }

      if (tc.function?.arguments) {
        events.push({
          type: 'content_block_delta',
          index: tcIndex,
          delta: {
            type: 'input_json_delta',
            partial_json: tc.function.arguments,
          },
        });
      }
    }
  }

  if (choice.finish_reason) {
    events.push({
      type: 'message_delta',
      delta: {
        type: mapOpenAIStopReasonToAnthropic(choice.finish_reason) ?? undefined,
      },
      usage: {
        input_tokens: chunk.usage?.prompt_tokens ?? 0,
        output_tokens: chunk.usage?.completion_tokens ?? 0,
      },
    });

    events.push({
      type: 'content_block_stop',
      index: 0,
    });

    events.push({
      type: 'message_stop',
    });
  }

  return events;
}

export function formatAnthropicStreamEvent(event: AnthropicStreamEvent): string {
  return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
}
