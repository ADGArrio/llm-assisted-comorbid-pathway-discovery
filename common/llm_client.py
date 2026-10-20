"""
Swappable LLM client.

Service: any OpenAI-compatible /v1/chat/completions endpoint -- local
Ollama by default (LLM_BASE_URL=http://localhost:11434, no key needed),
or any hosted model via .env (LLM_BASE_URL/LLM_MODEL/LLM_API_KEY).
Data: request/response are the standard OpenAI chat-completions JSON shape.
Used by: PipelineA.pipeline_a (narrative generation) and PipelineB.agent2_chain (chain selection).

KNOWN ISSUE (gemma4 via Ollama): a "thinking" model can put its reasoning
in a separate `reasoning` field and leave `content` empty. We send
"reasoning_effort": "none" to ask the server to skip that, and raise
instead of silently returning "" if content is still empty. Details:
https://github.com/ollama/ollama/issues/15288, /15293, /15635.
"""
import os
from typing import List

import requests


class LLMClient:
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None, timeout: int = 300):
        # Input: optional overrides, else read from .env (LLM_MODEL/LLM_BASE_URL/LLM_API_KEY).
        self.model = model or os.environ.get("LLM_MODEL", "gemma4:12b")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.timeout = timeout
        # finish_reason of the most recent reply ("stop" = finished; "length" = cut off because the output budget or the
        # model's context window ran out -- Ollama's default context is small, and a long prompt leaves almost no room
        # for the reply). Callers record it so a truncated reply is visible instead of looking like a model verdict.
        self.last_finish_reason = None

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        # Input: prompt text + sampling temperature. Output: the model's response text.
        # Raises RuntimeError if the server returns empty content (see KNOWN ISSUE above).
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "reasoning_effort": "none",  # ask the server to skip its thinking pass; ignored by servers that don't support it
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
                f"claude/agent4_stats_fix_and_stage1_retry.md.",
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


class StubLLMClient:
    """Fixed or queued responses, no network call. For tests.

    Added `responses` 2026-09-27, alongside Agent 2's pairwise debate tournament (see
    agent2_chain.py's module docstring): a test that exercises more than one LLM call per run
    (the stage-1 independent filter, then one or more stage-2 debates) needs a different response
    for each call, which a single `fixed_response` can't provide. `responses` is consumed in
    order, one per `generate()` call; `fixed_response` (unchanged) still works exactly as before
    for every existing single/repeated-response test."""

    def __init__(self, fixed_response: str = None, responses: List[str] = None):
        # Input: either fixed_response (returned on every call) or responses (an ordered queue,
        # one entry consumed per call) -- at least one is required.
        if fixed_response is None and responses is None:
            raise ValueError("StubLLMClient needs fixed_response or responses.")
        self.fixed_response = fixed_response
        self._responses = list(responses) if responses is not None else None
        self.last_prompt = None
        self.last_finish_reason = None
        self.prompts: List[str] = []  # every prompt seen, in call order -- useful when a test needs to
        # inspect more than just the most recent one (e.g. asserting on a specific debate's prompt).

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        # Input: prompt (recorded for test assertions). Output: the next queued response if
        # `responses` was given, else the fixed one.
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
