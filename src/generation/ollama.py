from __future__ import annotations

import httpx

from src.plugins.registry import generator_registry

from .base import GenerationResult, Generator


DEFAULT_SYSTEM_PROMPT = """
You are a helpful RAG assistant.

Answer the user's question using only the provided context.

Rules:
- Do not invent facts.
- If the context does not contain enough information, say so.
- Cite supporting information using [Source N].
""".strip()


@generator_registry.register("ollama")
class OllamaGenerator(Generator):

    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout: float = 120.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        if not model.strip():
            raise ValueError(
                "model cannot be empty"
            )

        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0 and 2"
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0"
            )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.timeout = timeout

    def generate(
        self,
        query: str,
        context: str,
        sources: list[dict] | None = None,
    ) -> GenerationResult:

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not context.strip():
            raise ValueError(
                "context cannot be empty"
            )

        prompt = (
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:"
        )

        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        with httpx.Client(
            timeout=self.timeout
        ) as client:

            response = client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )

            response.raise_for_status()

            data = response.json()

        answer = data.get("response", "").strip()

        if not answer:
            raise RuntimeError(
                "Ollama returned an empty response"
            )

        return GenerationResult(
            answer=answer,
            model=self.model,
            prompt=prompt,
            sources=sources or [],
            metadata={
                "provider": "ollama",
                "temperature": self.temperature,
                "done": data.get("done"),
            },
        )