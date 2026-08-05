from openai import OpenAI
from src.config import ModelConfig


def create_client(config: ModelConfig) -> OpenAI:
    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key or "not-needed",
    )


def chat_completion(
    client: OpenAI,
    system: str,
    user: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
