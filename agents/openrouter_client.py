"""OpenRouter API client — drop-in replacement for anthropic.Anthropic.

Exposes the same interface used by AnnotationAgent:

    client.messages.create(model=..., max_tokens=..., messages=[...])
    → returns obj with .content[0].text

Environment variables:
    OPENROUTER_API_KEY  — required, your OpenRouter secret key
    OPENROUTER_MODEL    — model id to use
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

DEFAULT_MODEL: str = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5:free")


class _ContentBlock:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _Message:
    __slots__ = ("content",)

    def __init__(self, text: str) -> None:
        self.content = [_ContentBlock(text)]


class _MessagesResource:
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def create(
        self,
        model: str,
        max_tokens: int,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> _Message:
        try:
            import requests as _requests
        except ImportError as exc:
            raise ImportError("Run: pip install requests") from exc

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        payload.update(kwargs)

        resp = _requests.post(
            self.BASE_URL,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data:
            raise RuntimeError(
                f"OpenRouter API returned unexpected response (no 'choices'):\n"
                f"{json.dumps(data, ensure_ascii=False, indent=2)}"
            )
        text: str = data["choices"][0]["message"].get("content") or ""
        return _Message(text)


class OpenRouterClient:
    """Minimal OpenRouter client compatible with the anthropic.Anthropic interface.

    Usage::

        client = OpenRouterClient(api_key="sk-or-...")
        msg = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(msg.content[0].text)
    """

    def __init__(self, api_key: str) -> None:
        self.messages = _MessagesResource(api_key)
