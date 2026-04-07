import {
  APICallError,
  LanguageModelV4,
  LanguageModelV4CallOptions,
  LanguageModelV4Content,
  LanguageModelV4FinishReason,
  LanguageModelV4GenerateResult,
  LanguageModelV4StreamPart,
  LanguageModelV4StreamResult,
  SharedV4ProviderMetadata,
  SharedV4Warning,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonErrorResponseHandler,
  createJsonResponseHandler,
  FetchFunction,
  generateId,
  parseProviderOptions,
  postJsonToApi,
  ResponseHandler,
  Resolvable,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import {
  AnthropicToOpenAIChatModelId,
  anthropicToOpenAILanguageModelChatOptions,
} from './anthropic-to-openai-chat-options';
import {
  convertAnthropicRequestToOpenAI,
  convertAnthropicToolsToOpenAI,
  convertAnthropicToolChoiceToOpenAI,
  convertAnthropicThinkingToOpenAI,
  convertAnthropicToOpenAIMessages,
} from './convert/anthropic-to-openai-request';
import { convertOpenAIResponseToAnthropic } from './convert/openai-to-anthropic-response';

const CHUNK_SCHEMA = z.object({
  id: z.string(),
  object: z.literal('chat.completion.chunk'),
  created: z.number(),
  model: z.string(),
  choices: z.array(
    z.object({
      index: z.number(),
      delta: z
        .object({
          role: z.literal('assistant').optional(),
          content: z.string().optional(),
          reasoning_content: z.string().optional(),
          tool_calls: z
            .array(
              z.object({
                index: z.number(),
                id: z.string().optional(),
                type: z.literal('function').optional(),
                function: z
                  .object({
                    name: z.string().optional(),
                    arguments: z.string().optional(),
                  })
                  .optional(),
              }),
            )
            .optional(),
        })
        .optional(),
      finish_reason: z.string().nullish(),
    }),
  ),
  usage: z
    .object({
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
      total_tokens: z.number().optional(),
    })
    .optional(),
});

const RESPONSE_SCHEMA = z.object({
  id: z.string(),
  object: z.literal('chat.completion'),
  created: z.number(),
  model: z.string(),
  choices: z.array(
    z.object({
      index: z.number(),
      message: z.object({
        role: z.literal('assistant'),
        content: z.string().nullable(),
        reasoning_content: z.string().optional(),
        tool_calls: z
          .array(
            z.object({
              id: z.string(),
              type: z.literal('function'),
              function: z.object({
                name: z.string(),
                arguments: z.string(),
              }),
            }),
          )
          .optional(),
      }),
      finish_reason: z.string().nullish(),
    }),
  ),
  usage: z
    .object({
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
      total_tokens: z.number().optional(),
    })
    .optional(),
});

export interface AnthropicToOpenAIChatConfig {
  baseURL: string;
  headers: Resolvable<Record<string, string | undefined>>;
  fetch?: FetchFunction;
}

export class AnthropicToOpenAIChatLanguageModel implements LanguageModelV4 {
  readonly specificationVersion = 'v4';

  readonly modelId: AnthropicToOpenAIChatModelId;
  private readonly config: AnthropicToOpenAIChatConfig;
  private readonly failedResponseHandler: ResponseHandler<APICallError>;

  constructor(
    modelId: AnthropicToOpenAIChatModelId,
    config: AnthropicToOpenAIChatConfig,
  ) {
    this.modelId = modelId;
    this.config = config;
    this.failedResponseHandler = createJsonErrorResponseHandler({
      errorSchema: z.object({
        message: z.string(),
        type: z.string(),
        code: z.string().optional(),
      }),
      errorType: 'provider',
    });
  }

  get provider(): string {
    return 'anthropic-to-openai';
  }

  get supportedUrls() {
    return {};
  }

  async doGenerate(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4GenerateResult> {
    const { args, warnings } = await this.getArgs(options);

    const body = convertAnthropicRequestToOpenAI(args.model, args.messages as any, {
      max_tokens: args.max_tokens,
      temperature: args.temperature,
      top_p: args.top_p,
      stop: args.stop as any,
      tools: args.tools as any,
      tool_choice: args.tool_choice as any,
      thinking: args.thinking as any,
      system: args.system,
    });

    const {
      responseHeaders,
      value: responseBody,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: `${this.config.baseURL.replace(/\/$/, '')}/chat/completions`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: this.failedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(RESPONSE_SCHEMA),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const anthropicResponse = convertOpenAIResponseToAnthropic(responseBody as any);
    const content: Array<LanguageModelV4Content> = [];

    for (const block of anthropicResponse.content) {
      if (block.type === 'thinking') {
        content.push({
          type: 'reasoning',
          text: block.thinking ?? '',
        });
      } else if (block.type === 'text') {
        content.push({
          type: 'text',
          text: block.text ?? '',
        });
      } else if (block.type === 'tool_use') {
        content.push({
          type: 'tool-call',
          toolCallId: block.id ?? generateId(),
          toolName: block.name ?? '',
          input: block.input ?? {},
        });
      }
    }

    return {
      content,
      finishReason: {
        unified: this.mapFinishReason(anthropicResponse.stop_reason),
        raw: anthropicResponse.stop_reason ?? undefined,
      },
      usage: {
        promptTokens: anthropicResponse.usage.input_tokens,
        completionTokens: anthropicResponse.usage.output_tokens,
        totalTokens:
          anthropicResponse.usage.input_tokens +
          anthropicResponse.usage.output_tokens,
      },
      providerMetadata: {
        'anthropic-to-openai': {},
      },
      request: { body: JSON.stringify(body) },
      response: {
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
    };
  }

  async doStream(
    options: LanguageModelV4CallOptions,
  ): Promise<LanguageModelV4StreamResult> {
    const { args, warnings } = await this.getArgs(options);

    const body = convertAnthropicRequestToOpenAI(args.model, args.messages as any, {
      max_tokens: args.max_tokens,
      temperature: args.temperature,
      top_p: args.top_p,
      stop: args.stop as any,
      tools: args.tools as any,
      tool_choice: args.tool_choice as any,
      thinking: args.thinking as any,
      system: args.system,
    });

    body.stream = true;
    (body as any).stream_options = { include_usage: true };

    const { responseHeaders, value: response } = await postJsonToApi({
      url: `${this.config.baseURL.replace(/\/$/, '')}/chat/completions`,
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: this.failedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(CHUNK_SCHEMA),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    let finishReason: LanguageModelV4FinishReason = {
      unified: 'other',
      raw: undefined,
    };

    let usage: { promptTokens: number; completionTokens: number; totalTokens: number } | undefined;
    let isFirstChunk = true;
    let isActiveReasoning = false;
    let isActiveText = false;
    const toolCalls: Array<{
      id: string;
      name: string;
      arguments: string;
      hasFinished: boolean;
    }> = [];

    return {
      stream: response.pipeThrough(
        new TransformStream<
          z.infer<typeof CHUNK_SCHEMA>,
          LanguageModelV4StreamPart
        >({
          start(controller) {
            controller.enqueue({ type: 'stream-start', warnings });
          },

          transform(chunk, controller) {
            if (!chunk.success) {
              finishReason = { unified: 'error', raw: undefined };
              controller.enqueue({ type: 'error', error: chunk.error });
              return;
            }

            const value = chunk.value;

            if (isFirstChunk) {
              isFirstChunk = false;
              controller.enqueue({
                type: 'response-metadata',
                responseId: value.id,
                timestamp: new Date(value.created * 1000),
                model: value.model,
              });
            }

            if (value.usage) {
              usage = {
                promptTokens: value.usage.prompt_tokens ?? 0,
                completionTokens: value.usage.completion_tokens ?? 0,
                totalTokens: value.usage.total_tokens ?? 0,
              };
            }

            const choice = value.choices[0];

            if (choice?.finish_reason != null) {
              finishReason = {
                unified: this.mapFinishReasonFromOpenAI(choice.finish_reason),
                raw: choice.finish_reason ?? undefined,
              };
            }

            if (choice?.delta == null) {
              return;
            }

            const delta = choice.delta;

            const reasoningContent = delta.reasoning_content;
            if (reasoningContent) {
              if (!isActiveReasoning) {
                controller.enqueue({
                  type: 'reasoning-start',
                  id: 'reasoning-0',
                });
                isActiveReasoning = true;
              }
              controller.enqueue({
                type: 'reasoning-delta',
                id: 'reasoning-0',
                delta: reasoningContent,
              });
            }

            if (delta.content) {
              if (isActiveReasoning) {
                controller.enqueue({
                  type: 'reasoning-end',
                  id: 'reasoning-0',
                });
                isActiveReasoning = false;
              }

              if (!isActiveText) {
                controller.enqueue({ type: 'text-start', id: 'txt-0' });
                isActiveText = true;
              }

              controller.enqueue({
                type: 'text-delta',
                id: 'txt-0',
                delta: delta.content,
              });
            }

            if (delta.tool_calls != null) {
              for (const toolCallDelta of delta.tool_calls) {
                const index = toolCallDelta.index ?? toolCalls.length;

                if (toolCalls[index] == null) {
                  if (toolCallDelta.id == null) {
                    continue;
                  }

                  toolCalls[index] = {
                    id: toolCallDelta.id,
                    name: toolCallDelta.function?.name ?? '',
                    arguments: '',
                    hasFinished: false,
                  };

                  controller.enqueue({
                    type: 'tool-call-start',
                    toolCallId: toolCallDelta.id,
                    toolName: toolCallDelta.function?.name ?? '',
                  });
                }

                if (toolCallDelta.function?.arguments) {
                  toolCalls[index].arguments += toolCallDelta.function.arguments;

                  controller.enqueue({
                    type: 'tool-call-delta',
                    toolCallId: toolCalls[index].id,
                    delta: toolCallDelta.function.arguments,
                  });
                }
              }
            }

            if (choice?.finish_reason != null) {
              if (isActiveReasoning) {
                controller.enqueue({
                  type: 'reasoning-end',
                  id: 'reasoning-0',
                });
              }
              if (isActiveText) {
                controller.enqueue({
                  type: 'text-end',
                  id: 'txt-0',
                });
              }

              for (const tc of toolCalls) {
                if (!tc.hasFinished) {
                  tc.hasFinished = true;
                  controller.enqueue({
                    type: 'tool-call-end',
                    toolCallId: tc.id,
                    result: tc.arguments,
                  });
                }
              }

              controller.enqueue({
                type: 'finish',
                finishReason,
                usage,
                providerMetadata: {
                  'anthropic-to-openai': {},
                },
              });
            }
          },
        }),
      ),
      response: {
        headers: responseHeaders,
      },
    };
  }

  private async getArgs(options: LanguageModelV4CallOptions) {
    const warnings: SharedV4Warning[] = [];

    const compatibleOptions = await parseProviderOptions({
      provider: 'anthropic-to-openai',
      providerOptions: options.providerOptions,
      schema: anthropicToOpenAILanguageModelChatOptions,
    });

    const prompt = options.prompt;

    const systemMessages = prompt.filter(
      (p) => p.role === 'system',
    ) as Array<{ role: 'system'; content: Array<{ type: string; text: string }> }>;
    const nonSystemMessages = prompt.filter((p) => p.role !== 'system');

    let systemContent: string | undefined;
    if (systemMessages.length > 0) {
      const systemMsg = systemMessages[0];
      if (systemMsg.content.length > 0 && systemMsg.content[0].type === 'text') {
        systemContent = systemMsg.content[0].text;
      }
    }

    const args: any = {
      model: this.modelId,
      messages: nonSystemMessages,
      system: systemContent,
    };

    if (options.maxOutputTokens) {
      args.max_tokens = options.maxOutputTokens;
    }

    if (options.temperature) {
      args.temperature = options.temperature;
    }

    if (options.topP) {
      args.top_p = options.topP;
    }

    if (options.stopSequences) {
      args.stop = options.stopSequences;
    }

    if (options.reasoning) {
      const thinking = convertAnthropicThinkingToOpenAI(options.reasoning as any);
      if (thinking) {
        if (thinking.type === 'disabled') {
          args.reasoning_effort = 'none';
        } else if (thinking.max_tokens) {
          args.max_completion_tokens = thinking.max_tokens;
        }
      }
    }

    if (options.tools && options.tools.length > 0) {
      const openaiTools = options.tools.map((tool) => ({
        type: 'function' as const,
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.parameters,
        },
      }));
      args.tools = openaiTools;
    }

    if (options.toolChoice) {
      const toolChoice = options.toolChoice;
      if (toolChoice.type === 'auto') {
        args.tool_choice = { type: 'auto' };
      } else if (toolChoice.type === 'required') {
        args.tool_choice = { type: 'required' };
      } else if (toolChoice.type === 'none') {
        args.tool_choice = { type: 'none' };
      } else if (toolChoice.type === 'tool') {
        args.tool_choice = { type: 'function', function: { name: toolChoice.toolName } };
      }
    }

    return { args, warnings };
  }

  private mapFinishReason(
    reason: 'end_turn' | 'tool_use' | 'max_tokens' | 'stop_sequence' | null,
  ): LanguageModelV4FinishReason['unified'] {
    switch (reason) {
      case 'end_turn':
      case 'stop_sequence':
        return 'stop';
      case 'tool_use':
        return 'tool-calls';
      case 'max_tokens':
        return 'length';
      default:
        return 'other';
    }
  }

  private mapFinishReasonFromOpenAI(
    reason: string | null,
  ): LanguageModelV4FinishReason['unified'] {
    switch (reason) {
      case 'stop':
        return 'stop';
      case 'tool_calls':
        return 'tool-calls';
      case 'length':
        return 'length';
      default:
        return 'other';
    }
  }
}
