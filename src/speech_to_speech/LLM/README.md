# Language models

The LLM runs locally by default. After [installation](../../../README.md#installation), `speech-to-speech local` picks a model for your platform:

| Platform | Backend | Model | Device |
|---|---|---|---|
| Apple Silicon macOS | `mlx-lm` | `mlx-community/Qwen3.5-4B-4bit` | MLX |
| Linux / Windows | `transformers` | `Qwen/Qwen3.5-4B` | Available accelerator, otherwise CPU (slow) |

Both backends use `LanguageModelHandler` in `language_model.py`. `--llm_backend` selects a backend and its default model; `--model_name` replaces the model. Use MLX conversions with `mlx-lm` and Transformers checkpoints with `transformers`:

```bash
speech-to-speech local --model_name mlx-community/Qwen3.5-9B-4bit   # Apple Silicon
speech-to-speech local --model_name Qwen/Qwen3.5-9B                 # Linux / Windows
```

`--llm_device` changes only the LLM's device (for example `cuda` or `cpu`); `--device` changes every stage. `--llm_torch_dtype` sets the Transformers dtype. `--mac-optimal-settings` is optional on Mac and only sets MPS device defaults.

## Local generation options

Local backends use the `--llm_*` arguments. For example, cap reply length:

```bash
speech-to-speech local --llm_gen_max_new_tokens 128
```

Transformers also accepts `--llm_gen_temperature`, `--llm_gen_do_sample`, and `--llm_gen_min_new_tokens`. The MLX text path forwards the maximum token count but not those sampling settings. Shared conversation settings include `--chat_size`, `--init_chat_prompt`, `--user_role`, and `--stream_batch_sentences`. See the [argument definitions](../arguments_classes/language_model_arguments.py) for defaults.

For image input, select a vision-language model and pass `--llm_is_vlm` to use `VisionLanguageModelHandler`. MLX vision inference needs `mlx-vlm`, included in `speech-to-speech[mlx-lm]`.

## API backends

Use an API backend when the LLM runs in a separate server or at a hosted provider. Neither is the default.

| Backend | Endpoint | Handler |
|---|---|---|
| `responses-api` | `/v1/responses` | `ResponsesApiModelHandler` |
| `chat-completions` | `/v1/chat/completions` | `ChatCompletionsApiModelHandler` |

Both use the `--responses_api_*` connection flags plus the shared settings above; `--llm_*` generation flags do not apply.

A local inference server:

```bash
speech-to-speech local \
  --llm_backend chat-completions \
  --model_name YOUR_SERVED_MODEL \
  --responses_api_base_url http://127.0.0.1:8000/v1 \
  --responses_api_api_key ""
```

A hosted provider, here OpenAI with its default `gpt-5.6-terra`:

```bash
export OPENAI_API_KEY=...
speech-to-speech local --llm_backend responses-api
```

Speech recognition and synthesis stay local either way. With a hosted provider, transcribed text and conversation history are sent to it. See [LLM backends](../../../README.md#llm-backends) for the llama.cpp setup, provider examples, and direct audio input.

### Responses without new user input

Realtime `response.create` can request another reply after a completed assistant message. Some Chat Completions providers instead treat that last message as unfinished text to continue (assistant prefill), which can produce an empty completion.

To test fresh assistant generation with llama.cpp, use [`--no-prefill-assistant`](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage). For Ollama 0.40.6 using its llama.cpp runner, set `LLAMA_ARG_PREFILL_ASSISTANT=0` in the **Ollama server's environment** before starting it. Changing the speech-to-speech process's environment does not configure an already-running provider.

Some models still repeat the last answer or speak when instructed to remain silent after prefill is disabled. Check those behaviors with your model before adopting the setting.

For Chat Completions, `--responses_api_instruction_only_format transcript` offers an alternative: send completed history as JSON request data alongside the current `response.instructions`. This gives the model a fresh response boundary and distinguishes the current request from completed turns. Text and function calls still use the provider's normal output channels; stored conversation history stays unchanged.

The option applies only to explicit, nonempty `response.instructions` when the selected history ends with a completed assistant message and contains text and completed tool exchanges. New user messages, tool-result turns, pending calls, media history, and omitted or cleared instructions keep the normal message format. The default is `messages`.

A tested configuration uses [Gemma 4 E4B Q4_0](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/a69f018d7678a2b9f5e705f8903d6fb29b9b25dc/gemma-4-E4B-it-Q4_0.gguf) with llama.cpp **b10098** (`0278d836`), its bundled chat template, and reasoning enabled:

```bash
llama-server -m /path/to/gemma-4-E4B-it-Q4_0.gguf \
  --alias local-gemma --host 127.0.0.1 --port 19740 \
  -ngl 99 -np 1 -c 8192 -t 4 -tb 4 -fa on \
  --jinja --reasoning on --no-mmproj --no-cache-prompt --prefill-assistant
```

Add these LLM settings to your usual speech-to-speech command:

```bash
speech-to-speech serve --llm_backend chat-completions \
  --model_name local-gemma --responses_api_base_url http://127.0.0.1:19740/v1 \
  --responses_api_disable_thinking false \
  --responses_api_instruction_only_format transcript
```

Compatibility depends on the model and its template. Validate requested speech, deliberate silence, and tool calls before enabling this format with another setup. It adds prompt tokens and does not make every model follow instructions reliably.

## Language behavior

`--enable_lang_prompt` adds a per-turn instruction asking the LLM to reply in the language reported by STT. It is off by default. Parakeet chooses its transcription language automatically; the Whisper backends accept `--language auto` or a fixed code.

`--detect_llm_output_language` separately detects the assistant's output language for TTS. See [Multi-language support](../../../README.md#multi-language-support) for how these options interact.
