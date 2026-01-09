import os
from typing import Any, List, Mapping, Optional

from openai import OpenAI


class UnifiedLLMClient:
    """Simple wrapper around the unified OpenAI-compatible endpoint."""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        url = api_url or os.getenv("OPENAI_BASE_URL", "")
        if not key:
            raise ValueError("An API key is required to create the unified LLM client")
        self.client = OpenAI(api_key=key, base_url=url)

    def chat(self, messages: List[Mapping[str, Any]], model: str, **kwargs: Any):
        return self.client.chat.completions.create(model=model, messages=messages, **kwargs)


def build_unified_client(api_key: Optional[str] = None, api_url: Optional[str] = None) -> UnifiedLLMClient:
    return UnifiedLLMClient(api_key=api_key, api_url=api_url)
