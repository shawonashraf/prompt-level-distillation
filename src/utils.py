from dataclasses import dataclass, field
import json
import re
import threading
from typing import Optional

import litellm
from src.config import ModelConfig

_stats_lock = threading.Lock()


@dataclass
class LLMStats:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def log(self, response):
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            with _stats_lock:
                self.requests += 1
                self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
                details = getattr(usage, "completion_tokens_details", None)
                if details:
                    self.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


_stats = LLMStats()


def get_stats() -> LLMStats:
    return _stats


def reset_stats():
    global _stats
    _stats = LLMStats()


def chat_completion(
    system: str,
    user: str,
    config: ModelConfig,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    model = f"openai/{config.model}"
    kwargs = {}
    effective_max = max_tokens if max_tokens is not None else config.max_tokens
    if effective_max:  # null/0 = let the server use the model's allowed maximum
        kwargs["max_tokens"] = effective_max
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        api_base=config.base_url,
        api_key=config.api_key or "not-needed",
        temperature=temperature if temperature is not None else config.temperature,
        **kwargs,
    )
    _stats.log(response)
    return response.choices[0].message.content or ""


def parse_json_response(text: str) -> dict:
    # ponytail: models wrap JSON in markdown fences or add prose; grab first {...} block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return json.loads(brace.group(0))
    return json.loads(text)
