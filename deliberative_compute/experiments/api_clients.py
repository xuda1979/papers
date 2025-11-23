"""Lightweight API clients for OpenAI and Gemini models.

These wrappers expose a uniform interface so experiment scripts can call
multiple backends without duplicating request boilerplate. The functions
are intentionally conservative (no streaming) to make result logging and
replay straightforward.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

# Optional imports: callers should ensure the corresponding packages are
# installed before invoking a client.
try:  # pragma: no cover - exercised in integration, not unit tests
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:  # pragma: no cover - exercised in integration, not unit tests
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore


@dataclass
class GenerationResult:
    """Container for text generations and token accounting."""

    texts: List[str]
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OpenAIChatModel:
    """Minimal OpenAI Chat Completions wrapper."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError("openai package is required for OpenAIChatModel")
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to call OpenAI APIs")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        texts = [c.message.content or "" for c in resp.choices]
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        return GenerationResult(texts=texts, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class GeminiChatModel:
    """Minimal Gemini wrapper using the google-generativeai SDK."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
    ) -> None:
        if genai is None:
            raise ImportError("google-generativeai package is required for GeminiChatModel")
        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required to call Gemini APIs")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: Optional[List[str]] = None,
    ) -> GenerationResult:
        texts: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for _ in range(n):
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    stop_sequences=stop or None,
                ),
            )
            texts.append(resp.text or "")
            meta = getattr(resp, "usage_metadata", None)
            prompt_tokens += getattr(meta, "prompt_token_count", 0) or 0
            completion_tokens += getattr(meta, "candidates_token_count", 0) or 0
        return GenerationResult(texts=texts, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
