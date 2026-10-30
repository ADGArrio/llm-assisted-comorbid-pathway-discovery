"""Swappable LLM client for any OpenAI-compatible /v1/chat/completions endpoint: local Ollama by default, or a hosted model
configured through .env (LLM_BASE_URL, LLM_MODEL, LLM_API_KEY). Also provides a stub client for tests.
"""
import os
from typing import List

import requests


# Client for one chat-completions endpoint.
class LLMClient:
    # Set up the client. Input: optional model, base URL, API key and timeout, otherwise read from .env. Output: none.
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None, timeout: int = 300):
        self.model = model or os.environ.get("LLM_MODEL", "gemma4:12b")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.timeout = timeout
        # finish_reason of the most recent reply ("stop" = finished, "length" = cut off by the output budget or context
        # window), so callers can record a truncated reply.
        self.last_finish_reason = None

    # Send one prompt and return the reply. Input: prompt text and sampling temperature. Output: the model's reply text;
    # raises RuntimeError if the server returns empty content.
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "reasoning_effort": "none",  # Ask the server to skip its thinking pass; servers that do not support it ignore this.
        }
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        self.last_finish_reason = data["choices"][0].get("finish_reason")
        if self.last_finish_reason == "length":
            print(
                f"WARNING: LLM reply was cut off (finish_reason=length) after {len(message.get('content') or '')} characters "
                f"for a {len(prompt)}-character prompt -- the model's context window is probably too small; see "
                f"the Stage 4 statistics fix and judge retry note.",
                flush=True,
            )
        content = (message.get("content") or "").strip()

        if not content:
            reasoning = (message.get("reasoning") or "").strip()
            hint = (
                f" The model's `reasoning` field was non-empty ({len(reasoning)} chars), "
                "suggesting a thinking-model response landed there instead of `content` -- "
                "see the KNOWN ISSUE note in this file's module docstring."
                if reasoning
                else " `reasoning` was also empty, so this doesn't look like the known "
                "thinking-mode issue -- check the model/server directly."
            )
            raise RuntimeError(
                f"LLMClient.generate(): model '{self.model}' returned empty content."
                + hint
            )

        return content


# Test double that returns fixed or queued replies without any network call.
class StubLLMClient:

    # Input: either fixed_response (returned on every call) or responses (an ordered queue, one entry consumed per
    # call); at least one is required. Output: none.
    def __init__(self, fixed_response: str = None, responses: List[str] = None):
        if fixed_response is None and responses is None:
            raise ValueError("StubLLMClient needs fixed_response or responses.")
        self.fixed_response = fixed_response
        self._responses = list(responses) if responses is not None else None
        self.last_prompt = None
        self.last_finish_reason = None
        self.prompts: List[str] = []  # Every prompt seen, in call order, for test assertions.

    # Record the prompt and return the next reply. Input: prompt text (temperature is ignored). Output: the next queued
    # reply if responses was given, else the fixed one.
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        self.last_prompt = prompt
        self.prompts.append(prompt)
        if self._responses is not None:
            if not self._responses:
                raise RuntimeError(
                    "StubLLMClient: responses queue exhausted -- this test needs to provide "
                    "more responses (one per generate() call, e.g. one per pairwise debate)."
                )
            return self._responses.pop(0)
        return self.fixed_response
