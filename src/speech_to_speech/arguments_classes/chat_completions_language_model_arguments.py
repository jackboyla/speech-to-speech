from dataclasses import dataclass, field
from typing import Literal, Optional

from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)


@dataclass
class ChatCompletionsLanguageModelHandlerArguments(ResponsesApiLanguageModelHandlerArguments):
    """Arguments for the ``chat-completions`` LLM backend.

    Inherits the OpenAI-compatible connection fields from the Responses-API
    arguments (``responses_api_base_url`` / ``responses_api_api_key`` /
    ``responses_api_stream`` / ``responses_api_disable_thinking``) so the same
    CLI flags and launcher env vars drive both backends. Chat Completions keeps
    the reasoning-effort default unset so provider-specific disable-thinking
    behavior remains unchanged.
    """

    responses_api_reasoning_effort: Optional[str] = field(
        default=None,
        metadata={
            "help": "Provider-specific reasoning level sent as extra_body={'reasoning_effort': <value>} on the "
            "Chat Completions request. Use to disable reasoning on providers where "
            "chat_template_kwargs.enable_thinking is ignored (e.g. 'none' / 'low'). When unset, falls back to "
            "the disable_thinking behaviour (chat_template_kwargs.enable_thinking=false). Default is None."
        },
    )
    responses_api_instruction_only_format: Literal["messages", "transcript"] = field(
        default="messages",
        metadata={
            "help": "Chat Completions only: use 'transcript' to send completed text history as request data when "
            "response.instructions requests another reply without new input. Requires a compatible model; "
            "ordinary user/tool turns and media history keep their native message format. Default is messages."
        },
    )
