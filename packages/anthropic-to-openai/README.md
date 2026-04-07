# @ai-sdk/anthropic-to-openai

Protocol adapter for converting Anthropic API requests to OpenAI-compatible format.

## Installation

```bash
npm install @ai-sdk/anthropic-to-openai
```

## Usage

```typescript
import { createAnthropicToOpenAI } from '@ai-sdk/anthropic-to-openai';
import { generateText } from 'ai';

const provider = createAnthropicToOpenAI({
  baseURL: 'https://your-proxy-provider.com/v1',
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const model = provider('claude-sonnet-4-20250514');

const result = await generateText({
  model,
  messages: [
    {
      role: 'user',
      content: 'Hello, how are you?',
    },
  ],
});
```

## Configuration

| Setting | Type | Description |
|---------|------|-------------|
| `baseURL` | `string` | The base URL of the OpenAI-compatible upstream provider. |
| `apiKey` | `string` | Optional API key for authentication. |
| `headers` | `Record<string, string>` | Optional custom headers. |
| `fetch` | `FetchFunction` | Optional custom fetch implementation. |

## Features

- Request conversion from Anthropic to OpenAI format
- Response conversion (streaming and non-streaming)
- Tool calling support
- System prompt handling
- Thinking/reasoning support

## License

Apache-2.0
