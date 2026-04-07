import { z } from 'zod/v4';

export type AnthropicToOpenAIChatModelId = string;

export const anthropicToOpenAILanguageModelChatOptions = z.object({
  user: z.string().optional(),
  model: z.string().optional(),
});

export type AnthropicToOpenAILanguageModelChatOptions = z.infer<
  typeof anthropicToOpenAILanguageModelChatOptions
>;
