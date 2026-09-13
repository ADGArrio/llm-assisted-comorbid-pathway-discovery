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
import requests


class LLMClient:
    def __init__(self, model: str = None, base_url: str = None, api_key: str = None, timeout: int = 300):
        # Input: optional overrides, else read from .env (LLM_MODEL/LLM_BASE_URL/LLM_API_KEY).
        self.model = model or os.environ.get("LLM_MODEL", "gemma4:12b")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.timeout = timeout

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
    """Fixed response, no network call. For tests."""

    def __init__(self, fixed_response: str):
        self.fixed_response = fixed_response
        self.last_prompt = None

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        # Input: prompt (recorded to self.last_prompt for test assertions). Output: self.fixed_response.
        self.last_prompt = prompt
        return self.fixed_response
